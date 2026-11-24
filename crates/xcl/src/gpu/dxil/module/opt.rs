//! Cheap cleanup of the entry function before it is written: unreachable
//! blocks emptied, trivial phis replaced by their single value, common
//! subexpressions merged (pure instructions along the dominator tree, wave
//! intrinsics within a block), dead instructions removed.
//!
//! The translator emits each guest instruction's per-lane expansion as is
//! (repeated lane-bit / ballot / extract sequences, operands of discarded
//! results); this pass removes the redundancy without changing what any
//! remaining instruction computes.

use std::collections::HashMap;

use super::{BinOp, CastOp, ConstDef, FnAttrs, Module, Op, Pred, Ty, TypeDef, Value};

impl Op {
    /// Calls `f` on every value operand.
    pub(super) fn for_each_operand(&self, mut f: impl FnMut(Value)) {
        match self {
            Op::Bin(_, a, b, _) | Op::Cmp(_, a, b) | Op::Store(a, b, _) | Op::AtomicRmw(_, a, b) => {
                f(*a);
                f(*b)
            }
            Op::Cast(_, v, _) | Op::Extract(v, _) | Op::Load(v, _) | Op::Alloca(_, v, _) | Op::CondBr(v, ..) => f(*v),
            Op::Select(a, b, c) | Op::CmpXchg(a, b, c) => {
                f(*a);
                f(*b);
                f(*c)
            }
            Op::Call(_, args) => args.iter().copied().for_each(f),
            Op::Phi(inc) => inc.iter().for_each(|p| f(p.0)),
            Op::Gep(p, idx) => {
                f(*p);
                idx.iter().copied().for_each(f)
            }
            Op::Switch(v, _, cases) => {
                f(*v);
                cases.iter().for_each(|c| f(c.0))
            }
            Op::Br(_) | Op::Ret | Op::Unreachable => {}
        }
    }

    /// Rewrites every value operand through `f`.
    fn map_operands(&mut self, mut f: impl FnMut(Value) -> Value) {
        match self {
            Op::Bin(_, a, b, _) | Op::Cmp(_, a, b) | Op::Store(a, b, _) | Op::AtomicRmw(_, a, b) => {
                *a = f(*a);
                *b = f(*b)
            }
            Op::Cast(_, v, _) | Op::Extract(v, _) | Op::Load(v, _) | Op::Alloca(_, v, _) | Op::CondBr(v, ..) => *v = f(*v),
            Op::Select(a, b, c) | Op::CmpXchg(a, b, c) => {
                *a = f(*a);
                *b = f(*b);
                *c = f(*c)
            }
            Op::Call(_, args) => args.iter_mut().for_each(|a| *a = f(*a)),
            Op::Phi(inc) => inc.iter_mut().for_each(|p| p.0 = f(p.0)),
            Op::Gep(p, idx) => {
                *p = f(*p);
                idx.iter_mut().for_each(|v| *v = f(*v))
            }
            Op::Switch(v, _, cases) => {
                *v = f(*v);
                cases.iter_mut().for_each(|c| c.0 = f(c.0))
            }
            Op::Br(_) | Op::Ret | Op::Unreachable => {}
        }
    }
}

/// How an instruction may be merged with an identical one.
#[derive(PartialEq)]
enum Cse {
    No,
    /// pure: any dominating copy will do
    Dom,
    /// a wave intrinsic: same result only under the same active lanes (same block)
    Block,
}

impl Module {
    fn wave_call(&self, f: u32) -> bool {
        let n = &self.funcs[f as usize].name;
        n.starts_with("dx.op.wave") || n.starts_with("dx.op.quad")
    }

    fn is_const_global_ptr(&self, v: Value) -> bool {
        match v {
            Value::Global(g) => self.globals[g as usize].constant,
            Value::Inst(i) => match &self.insts[i as usize].op {
                Op::Gep(base, _) => self.is_const_global_ptr(*base),
                _ => false,
            },
            _ => false,
        }
    }

    fn cse_kind(&self, op: &Op) -> Cse {
        match op {
            Op::Bin(..) | Op::Cast(..) | Op::Cmp(..) | Op::Select(..) | Op::Extract(..) | Op::Gep(..) => Cse::Dom,
            Op::Load(p, _) if self.is_const_global_ptr(*p) => Cse::Dom,
            Op::Call(f, _) if self.funcs[*f as usize].attrs == FnAttrs::ReadNone => Cse::Dom,
            Op::Call(f, _) if self.wave_call(*f) && !self.funcs[*f as usize].name.starts_with("dx.op.waveIsFirstLane") => Cse::Block,
            _ => Cse::No,
        }
    }

    fn zext_src(&self, v: Value) -> Option<(Value, u32)> {
        let Value::Inst(j) = v else { return None };
        let Op::Cast(CastOp::ZExt, inner, _) = self.insts[j as usize].op else { return None };
        self.int_width(self.value_ty(inner)).map(|sw| (inner, sw))
    }

    /// Reinterprets `v` as `to` for free: inverse `BitCast`, dominating `BitCast` in CSE,
    /// or finite constant bitcast (excluding NaN and Inf in both directions).
    fn free_bitcast(&mut self, v: Value, to: Ty, table: &HashMap<(Op, Ty), Value>) -> Option<Value> {
        let from = self.value_ty(v);
        if self.types[from.0 as usize] == TypeDef::Double || self.types[to.0 as usize] == TypeDef::Double {
            return None;
        }
        if let Value::Inst(j) = v {
            if let Op::Cast(CastOp::BitCast, inner, _) = self.insts[j as usize].op {
                if self.value_ty(inner) == to {
                    return Some(inner);
                }
            }
        }
        if let Some(&hit) = table.get(&(Op::Cast(CastOp::BitCast, v, to), to)) {
            return Some(hit);
        }
        let fbits = |v| match v {
            Value::Const(c) => match &self.consts[c as usize] {
                ConstDef::Float(_, b) => Some(*b),
                ConstDef::Null(t) if self.is_float(*t) => Some(0),
                _ => None,
            },
            _ => None,
        };
        match (&self.types[from.0 as usize], &self.types[to.0 as usize]) {
            (TypeDef::Int(32), TypeDef::Float) => self.const_u64(v).filter(|&b| (b as u32 & 0x7FFF_FFFF) < 0x7F80_0000).map(|b| self.const_float(to, b & 0xFFFF_FFFF)),
            (TypeDef::Float, TypeDef::Int(32)) => fbits(v).filter(|&b| (b as u32 & 0x7FFF_FFFF) < 0x7F80_0000).map(|b| self.const_int(to, b & 0xFFFF_FFFF)),
            (TypeDef::Int(16), TypeDef::Half) => self.const_u64(v).filter(|&b| (b as u16 & 0x7FFF) < 0x7C00).map(|b| self.const_float(to, b & 0xFFFF)),
            (TypeDef::Half, TypeDef::Int(16)) => fbits(v).filter(|&b| (b as u16 & 0x7FFF) < 0x7C00).map(|b| self.const_int(to, b & 0xFFFF)),
            _ => None,
        }
    }

    /// Pushes `bitcast(select(c, a, b), to)` into `select` in place when `select` has no
    /// non-bitcast uses (so the original select dies in DCE) and both arms convert for free,
    /// returning the original `(op, ty)` so CSE can still reuse the bitcast key in dominated blocks.
    fn simplify_bitcast_select(&mut self, op: &mut Op, repl: &mut [Option<Value>], table: &HashMap<(Op, Ty), Value>, bc_only: &[Option<Option<Ty>>]) -> Option<(Op, Ty)> {
        let Op::Cast(CastOp::BitCast, Value::Inst(j), to) = *op else { return None };
        if bc_only[j as usize] != Some(Some(to)) {
            return None;
        }
        let Op::Select(c, t, f) = self.insts[j as usize].op else { return None };
        let nt = self.free_bitcast(resolve(repl, t), to, table)?;
        let nf = self.free_bitcast(resolve(repl, f), to, table)?;
        let orig = (op.clone(), to);
        *op = Op::Select(resolve(repl, c), resolve(repl, nt), resolve(repl, nf));
        Some(orig)
    }

    /// Collapses nested same-condition selects (`select(c, t, select(c, _, f)) -> select(c, t, f)`).
    fn simplify_select(&self, op: &mut Op, ty: Ty, repl: &mut [Option<Value>]) {
        let Op::Select(c, t, f) = op else { return };
        if self.types[ty.0 as usize] == TypeDef::Double {
            return;
        }
        if let Value::Inst(j) = *f {
            if let Op::Select(c2, _, f2) = self.insts[j as usize].op {
                if resolve(repl, c2) == *c {
                    *f = resolve(repl, f2);
                }
            }
        }
        if let Value::Inst(j) = *t {
            if let Op::Select(c2, t2, _) = self.insts[j as usize].op {
                if resolve(repl, c2) == *c {
                    *t = resolve(repl, t2);
                }
            }
        }
    }

    /// Canonicalizes operand order on commutative integer ops and integer compares for CSE.
    fn canonicalize_commutative(&self, op: &mut Op, ty: Ty) {
        match op {
            Op::Bin(BinOp::Add | BinOp::Mul | BinOp::And | BinOp::Or | BinOp::Xor, a, b, _)
                if self.int_width(ty).is_some() && *b < *a => std::mem::swap(a, b),
            Op::Cmp(p, a, b) if self.int_width(self.value_ty(*a)).is_some() && *b < *a => {
                *p = match *p {
                    Pred::Eq => Pred::Eq, Pred::Ne => Pred::Ne,
                    Pred::UGt => Pred::ULt, Pred::ULt => Pred::UGt,
                    Pred::UGe => Pred::ULe, Pred::ULe => Pred::UGe,
                    Pred::SGt => Pred::SLt, Pred::SLt => Pred::SGt,
                    Pred::SGe => Pred::SLe, Pred::SLe => Pred::SGe,
                    _ => return,
                };
                std::mem::swap(a, b);
            }
            _ => {}
        }
    }

    /// Cheap algebraic & constant simplification on already-resolved operands.
    fn simplify_op(&mut self, op: &Op, ty: Ty, table: &HashMap<(Op, Ty), Value>) -> Option<Value> {
        // Keep f64 instructions intact so Doubles / DoubleExtensions shader flags stay consistent.
        if matches!(self.types[ty.0 as usize], TypeDef::Double) {
            return None;
        }
        match *op {
            Op::Select(c, t, f) => {
                if t == f {
                    return Some(t);
                }
                match self.const_u64(c) {
                    Some(1) => Some(t),
                    Some(0) => Some(f),
                    _ => (self.int_width(ty) == Some(1) && self.const_u64(t) == Some(1) && self.const_u64(f) == Some(0))
                        .then_some(c),
                }
            }
            Op::Bin(bop, a, b, 0) => {
                let w = self.int_width(ty)?;
                if w > 64 {
                    return None;
                }
                let mask = if w >= 64 { u64::MAX } else { (1u64 << w) - 1 };
                let (ca, cb) = (self.const_u64(a), self.const_u64(b));
                if let (Some(x), Some(y)) = (ca, cb) {
                    let sx = if w >= 64 { x as i64 } else { ((x << (64 - w)) as i64) >> (64 - w) };
                    let r = match bop {
                        BinOp::Add => Some(x.wrapping_add(y)),
                        BinOp::Sub => Some(x.wrapping_sub(y)),
                        BinOp::Mul => Some(x.wrapping_mul(y)),
                        BinOp::And => Some(x & y),
                        BinOp::Or => Some(x | y),
                        BinOp::Xor => Some(x ^ y),
                        BinOp::Shl if y < w as u64 => Some(x << y),
                        BinOp::LShr if y < w as u64 => Some(x >> y),
                        BinOp::AShr if y < w as u64 => Some((sx >> (y as u32)) as u64),
                        BinOp::UDiv if y != 0 => Some(x / y),
                        BinOp::URem if y != 0 => Some(x % y),
                        _ => None,
                    };
                    if let Some(r) = r {
                        return Some(self.const_int(ty, r & mask));
                    }
                }
                match (bop, cb) {
                    (BinOp::Add | BinOp::Sub | BinOp::Or | BinOp::Xor | BinOp::Shl | BinOp::LShr | BinOp::AShr, Some(0)) => return Some(a),
                    (BinOp::And, Some(m)) if m == mask => return Some(a),
                    (BinOp::Or, Some(m)) if m == mask => return Some(b),
                    (BinOp::And | BinOp::Mul, Some(0)) => return Some(b),
                    (BinOp::Mul, Some(1)) => return Some(a),
                    _ => {}
                }
                match (bop, ca) {
                    (BinOp::Add | BinOp::Or | BinOp::Xor, Some(0)) => return Some(b),
                    (BinOp::And, Some(m)) if m == mask => return Some(b),
                    (BinOp::Or, Some(m)) if m == mask => return Some(a),
                    (BinOp::And | BinOp::Mul | BinOp::Shl | BinOp::LShr | BinOp::AShr, Some(0)) => return Some(a),
                    (BinOp::Mul, Some(1)) => return Some(b),
                    _ => {}
                }
                if a == b {
                    match bop {
                        BinOp::And | BinOp::Or => return Some(a),
                        BinOp::Sub | BinOp::Xor => return Some(self.const_int(ty, 0)),
                        _ => {}
                    }
                }
                if bop == BinOp::LShr {
                    if let Some(sh) = cb.filter(|&sh| sh < w as u64) {
                        if self.zext_src(a).is_some_and(|(_, sw)| sh >= sw as u64) {
                            return Some(self.const_int(ty, 0));
                        }
                        if let Value::Inst(j) = a {
                            if let Op::Bin(BinOp::Or, x, y, 0) = self.insts[j as usize].op {
                                for (lo, hi) in [(x, y), (y, x)] {
                                    let lo_cleared = self.const_u64(lo) == Some(0)
                                        || self.zext_src(lo).is_some_and(|(_, sw)| sh >= sw as u64);
                                    if let (true, Value::Inst(hj)) = (lo_cleared, hi) {
                                        if let Op::Bin(BinOp::Shl, z, k, 0) = self.insts[hj as usize].op {
                                            if self.const_u64(k) == Some(sh)
                                                && self.zext_src(z).is_some_and(|(_, sw)| (sw as u64) + sh <= w as u64)
                                            {
                                                return Some(z);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                None
            }
            Op::Cmp(p, a, b) => {
                let at = self.value_ty(a);
                let w = self.int_width(at)?;
                if w > 64 {
                    return None;
                }
                let (ca, cb) = (self.const_u64(a), self.const_u64(b));
                if let (Some(x), Some(y)) = (ca, cb) {
                    let sx = |v: u64| if w >= 64 { v as i64 } else { ((v << (64 - w)) as i64) >> (64 - w) };
                    let r = match p {
                        Pred::Eq => Some(x == y),
                        Pred::Ne => Some(x != y),
                        Pred::UGt => Some(x > y),
                        Pred::UGe => Some(x >= y),
                        Pred::ULt => Some(x < y),
                        Pred::ULe => Some(x <= y),
                        Pred::SGt => Some(sx(x) > sx(y)),
                        Pred::SGe => Some(sx(x) >= sx(y)),
                        Pred::SLt => Some(sx(x) < sx(y)),
                        Pred::SLe => Some(sx(x) <= sx(y)),
                        _ => None,
                    };
                    if let Some(r) = r {
                        return Some(self.const_int(ty, r as u64));
                    }
                }
                if a == b {
                    let r = match p {
                        Pred::Eq | Pred::ULe | Pred::UGe | Pred::SLe | Pred::SGe => Some(true),
                        Pred::Ne | Pred::ULt | Pred::UGt | Pred::SLt | Pred::SGt => Some(false),
                        _ => None,
                    };
                    if let Some(r) = r {
                        return Some(self.const_int(ty, r as u64));
                    }
                }
                if w == 1 {
                    if matches!((p, cb), (Pred::Eq, Some(1)) | (Pred::Ne, Some(0))) {
                        return Some(a);
                    }
                    if matches!((p, ca), (Pred::Eq, Some(1)) | (Pred::Ne, Some(0))) {
                        return Some(b);
                    }
                }
                if let Some((inner, 1)) = self.zext_src(a) {
                    if matches!((p, cb), (Pred::Ne | Pred::UGt, Some(0)) | (Pred::Eq, Some(1))) {
                        return Some(inner);
                    }
                }
                if let Some((inner, 1)) = self.zext_src(b) {
                    if matches!((p, ca), (Pred::Ne | Pred::ULt, Some(0)) | (Pred::Eq, Some(1))) {
                        return Some(inner);
                    }
                }
                None
            }
            Op::Cast(CastOp::BitCast, v, to) => self.free_bitcast(v, to, table),
            Op::Cast(cop @ (CastOp::ZExt | CastOp::SExt | CastOp::Trunc), v, to) => {
                let sw = self.int_width(self.value_ty(v))?;
                let dw = self.int_width(to)?;
                if sw > 64 || dw > 64 {
                    return None;
                }
                if sw == dw {
                    return Some(v);
                }
                if cop == CastOp::Trunc {
                    if let Value::Inst(j) = v {
                        if let Op::Cast(CastOp::ZExt | CastOp::SExt, inner, _) = self.insts[j as usize].op {
                            if self.value_ty(inner) == to {
                                return Some(inner);
                            }
                        }
                    }
                }
                let x = self.const_u64(v)?;
                let dm = if dw >= 64 { u64::MAX } else { (1u64 << dw) - 1 };
                let r = match cop {
                    CastOp::ZExt | CastOp::Trunc => x & dm,
                    CastOp::SExt => {
                        let sx = if sw >= 64 { x as i64 } else { ((x << (64 - sw)) as i64) >> (64 - sw) };
                        (sx as u64) & dm
                    }
                    _ => unreachable!(),
                };
                Some(self.const_int(to, r))
            }
            _ => None,
        }
    }

    /// Whether an unused instruction can go. Double values stay: helpers already set
    /// the Doubles / DoubleExtensions flags for them (`Int64Ops` is refreshed in `infer_type_features`).
    fn removable(&self, op: &Op, ty: Ty, wide: &[Ty]) -> bool {
        if wide.contains(&ty) {
            return false;
        }
        let mut w = false;
        op.for_each_operand(|v| w |= wide.contains(&self.value_ty(v)));
        if w {
            return false;
        }
        match op {
            Op::Bin(..) | Op::Cast(..) | Op::Cmp(..) | Op::Select(..) | Op::Extract(..) | Op::Gep(..) | Op::Phi(_) | Op::Load(..) | Op::Alloca(..) => true,
            // only calls whose presence sets no shader flag (the flags are collected while
            // building and must match the code: no wave ops, f64 math, typed / raw loads)
            Op::Call(f, _) => {
                let d = &self.funcs[*f as usize];
                matches!(d.attrs, FnAttrs::ReadNone | FnAttrs::ReadOnly)
                    && !d.name.ends_with(".f64")
                    && [
                        "dx.op.unary.", "dx.op.unaryBits.", "dx.op.binary.", "dx.op.tertiary.", "dx.op.quaternary.",
                        "dx.op.dot", "dx.op.legacyF16ToF32", "dx.op.legacyF32ToF16", "dx.op.bitcast",
                        "dx.op.isSpecialFloat.", "dx.op.threadId", "dx.op.groupId.", "dx.op.flattenedThreadIdInGroup.",
                        "dx.op.loadInput.", "dx.op.annotateHandle", "dx.op.createHandleFromBinding",
                        "dx.op.cbufferLoadLegacy.",
                    ]
                    .iter()
                    .any(|p| d.name.starts_with(p))
            }
            _ => false,
        }
    }

    /// Whether every non-phi operand of the reachable instructions is written before its
    /// use and has a value (what the writer checks).
    fn uses_well_formed(&self, order: &[u32]) -> bool {
        let mut at = vec![usize::MAX; self.insts.len()];
        let mut k = 0;
        for &b in order {
            for &i in &self.blocks[b as usize] {
                at[i as usize] = k;
                k += 1;
            }
        }
        let mut ok = true;
        for &b in order {
            for &i in &self.blocks[b as usize] {
                let op = &self.insts[i as usize].op;
                if matches!(op, Op::Phi(_)) {
                    continue;
                }
                op.for_each_operand(|v| {
                    if let Value::Inst(j) = v {
                        ok &= at[j as usize] < at[i as usize] && !matches!(self.types[self.insts[j as usize].ty.0 as usize], super::TypeDef::Void);
                    }
                });
            }
        }
        ok
    }

    /// Immediate dominators of the blocks in `order` (reverse postorder), by
    /// Cooper, Harvey and Kennedy's iteration ("A Simple, Fast Dominance Algorithm").
    fn idoms(&self, order: &[u32]) -> Vec<Option<usize>> {
        let n = self.blocks.len();
        let mut rpo = vec![usize::MAX; n];
        for (k, &b) in order.iter().enumerate() {
            rpo[b as usize] = k;
        }
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &b in order {
            for s in self.successors(b as usize) {
                preds[s.0 as usize].push(b as usize);
            }
        }
        let mut idom: Vec<Option<usize>> = vec![None; n];
        idom[0] = Some(0);
        let mut changed = true;
        while changed {
            changed = false;
            for &b in order.iter().skip(1) {
                let b = b as usize;
                let mut new: Option<usize> = None;
                for &p in &preds[b] {
                    if idom[p].is_none() {
                        continue;
                    }
                    new = Some(match new {
                        None => p,
                        Some(mut x) => {
                            let mut y = p;
                            while x != y {
                                while rpo[x] > rpo[y] {
                                    x = idom[x].unwrap();
                                }
                                while rpo[y] > rpo[x] {
                                    y = idom[y].unwrap();
                                }
                            }
                            x
                        }
                    });
                }
                if new.is_some() && idom[b] != new {
                    idom[b] = new;
                    changed = true;
                }
            }
        }
        idom
    }

    /// Cleans up the entry function (see the module docs of `opt`). Values
    /// keep their identities; replaced ones simply lose their uses.
    ///
    /// `XCL_DXIL_NO_OPT=phi,fold,cse,dce` (any subset, or `all`) skips passes, for bisecting.
    pub fn optimize(&mut self) {
        let off = std::env::var("XCL_DXIL_NO_OPT").unwrap_or_default();
        let skip = |p: &str| off.split(',').any(|x| x.trim() == p || x.trim() == "all");
        let order: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();
        // malformed IR (a use before its definition or of a void result) is left for the
        // writer to report, not cleaned away
        if !self.uses_well_formed(&order) {
            return;
        }
        // unreachable blocks: emptied (the writer drops them)
        let mut reach = vec![false; self.blocks.len()];
        for &b in &order {
            reach[b as usize] = true;
        }
        for (b, r) in reach.iter().enumerate() {
            if !r {
                self.blocks[b].clear();
            }
        }
        for &b in &order {
            for k in 0..self.blocks[b as usize].len() {
                let i = self.blocks[b as usize][k] as usize;
                if let Op::Phi(inc) = &mut self.insts[i].op {
                    inc.retain(|p| reach[p.1 .0 as usize]);
                }
            }
        }

        let mut repl: Vec<Option<Value>> = vec![None; self.insts.len()];

        // trivial phis (all incoming values the phi itself or one value), to a fixed point
        while !skip("phi") {
            let mut changed = false;
            for &b in &order {
                for &i in &self.blocks[b as usize] {
                    let i = i as usize;
                    if repl[i].is_some() {
                        continue;
                    }
                    let Op::Phi(inc) = &self.insts[i].op else { break };
                    let me = Value::Inst(i as u32);
                    let mut same: Option<Value> = None;
                    let mut trivial = true;
                    for &(v, _) in inc {
                        let v = resolve(&mut repl, v);
                        if v == me || Some(v) == same {
                            continue;
                        }
                        if same.is_some() {
                            trivial = false;
                            break;
                        }
                        same = Some(v);
                    }
                    if let (true, Some(v)) = (trivial, same) {
                        repl[i] = Some(v);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // CSE along the dominator tree (preorder walk with a scoped table)
        let idom = self.idoms(&order);
        let mut kids: Vec<Vec<usize>> = vec![Vec::new(); self.blocks.len()];
        for &b in order.iter().skip(1) {
            if let Some(d) = idom[b as usize] {
                kids[d].push(b as usize);
            }
        }
        let mut table: HashMap<(Op, Ty), Value> = HashMap::new();
        // (block, undo log length) frames; undo log holds keys to remove / restore
        let mut undo: Vec<((Op, Ty), Option<Value>)> = Vec::new();
        enum Step {
            Enter(usize),
            Leave(usize),
        }
        let (do_fold, do_cse) = (!skip("fold"), !skip("cse"));
        // None = unseen, Some(Some(to)) = only used by BitCast(_, to), Some(None) = other use.
        let mut bc_only: Vec<Option<Option<Ty>>> = vec![None; if do_fold { self.insts.len() } else { 0 }];
        if do_fold {
            for &b in &order {
                for &u in &self.blocks[b as usize] {
                    if repl[u as usize].is_some() {
                        continue;
                    }
                    match self.insts[u as usize].op {
                        Op::Cast(CastOp::BitCast, v, to) => if let Value::Inst(j) = resolve(&mut repl, v) {
                            bc_only[j as usize] = match bc_only[j as usize] {
                                None => Some(Some(to)),
                                Some(Some(prev)) if prev == to => Some(Some(to)),
                                _ => Some(None),
                            };
                        },
                        Op::Select(c, t, f) if self.types[self.insts[u as usize].ty.0 as usize] != TypeDef::Double => {
                            let rc = resolve(&mut repl, c);
                            if let Value::Inst(j) = rc { bc_only[j as usize] = Some(None); }
                            for arm in [t, f] {
                                if let Value::Inst(j) = resolve(&mut repl, arm) {
                                    if !matches!(self.insts[j as usize].op, Op::Select(c2, _, _) if resolve(&mut repl, c2) == rc) {
                                        bc_only[j as usize] = Some(None);
                                    }
                                }
                            }
                        }
                        _ => self.insts[u as usize].op.for_each_operand(|v| {
                            if let Value::Inst(j) = resolve(&mut repl, v) { bc_only[j as usize] = Some(None); }
                        }),
                    }
                }
            }
        }
        let mut stack = if !do_fold && !do_cse { Vec::new() } else { vec![Step::Enter(0)] };
        let mut cse_replaced = false;
        while let Some(step) = stack.pop() {
            let b = match step {
                Step::Leave(mark) => {
                    while undo.len() > mark {
                        let (k, old) = undo.pop().unwrap();
                        match old {
                            Some(v) => table.insert(k, v),
                            None => table.remove(&k),
                        };
                    }
                    continue;
                }
                Step::Enter(b) => b,
            };
            stack.push(Step::Leave(undo.len()));
            let mut local: HashMap<(Op, Ty), Value> = HashMap::new();
            for k in 0..self.blocks[b].len() {
                let i = self.blocks[b][k] as usize;
                let ty = self.insts[i].ty;
                let mut op = std::mem::replace(&mut self.insts[i].op, Op::Ret);
                op.map_operands(|v| resolve(&mut repl, v));
                if repl[i].is_none() {
                    let alias_key = if do_fold {
                        let k = self.simplify_bitcast_select(&mut op, &mut repl, &table, &bc_only);
                        self.simplify_select(&mut op, ty, &mut repl);
                        if let Some(s) = self.simplify_op(&op, ty, &table) {
                            repl[i] = Some(resolve(&mut repl, s));
                            cse_replaced = true;
                        }
                        k
                    } else {
                        None
                    };
                    if repl[i].is_none() && do_cse {
                        self.canonicalize_commutative(&mut op, ty);
                        let kind = self.cse_kind(&op);
                        if kind != Cse::No {
                            let key = (op.clone(), ty);
                            let t = if kind == Cse::Dom { &mut table } else { &mut local };
                            if let Some(&v) = t.get(&key) {
                                repl[i] = Some(v);
                                cse_replaced = true;
                            } else {
                                let old = t.insert(key.clone(), Value::Inst(i as u32));
                                if kind == Cse::Dom {
                                    undo.push((key, old));
                                }
                            }
                        }
                    }
                    if let (true, Some(akey)) = (do_cse, alias_key) {
                        let old = table.insert(akey.clone(), repl[i].unwrap_or(Value::Inst(i as u32)));
                        undo.push((akey, old));
                    }
                }
                self.insts[i].op = op;
            }
            for &c in kids[b].iter().rev() {
                stack.push(Step::Enter(c));
            }
        }
        // second trivial-phi fixed point if CSE/simplification collapsed incoming values
        if cse_replaced && !skip("phi") {
            loop {
                let mut changed = false;
                for &b in &order {
                    for &i in &self.blocks[b as usize] {
                        let i = i as usize;
                        if repl[i].is_some() {
                            continue;
                        }
                        let Op::Phi(inc) = &self.insts[i].op else { break };
                        let me = Value::Inst(i as u32);
                        let mut same: Option<Value> = None;
                        let mut trivial = true;
                        for &(v, _) in inc {
                            let v = resolve(&mut repl, v);
                            if v == me || Some(v) == same {
                                continue;
                            }
                            if same.is_some() {
                                trivial = false;
                                break;
                            }
                            same = Some(v);
                        }
                        if let (true, Some(v)) = (trivial, same) {
                            repl[i] = Some(v);
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        // phis may name values defined later (back edges): resolve once more
        for &b in &order {
            for k in 0..self.blocks[b as usize].len() {
                let i = self.blocks[b as usize][k] as usize;
                self.insts[i].op.map_operands(|v| resolve(&mut repl, v));
            }
        }

        // dead instructions: live = side effects, terminators and their operands, transitively
        let (_, f64t) = self.existing_i64_f64();
        let wide: Vec<Ty> = f64t.into_iter().collect();
        let mut live = vec![false; self.insts.len()];
        let mut work: Vec<u32> = Vec::new();
        for &b in &order {
            for &i in &self.blocks[b as usize] {
                if repl[i as usize].is_none() && (skip("dce") || !self.removable(&self.insts[i as usize].op, self.insts[i as usize].ty, &wide)) {
                    live[i as usize] = true;
                    work.push(i);
                }
            }
        }
        while let Some(i) = work.pop() {
            self.insts[i as usize].op.for_each_operand(|v| {
                if let Value::Inst(j) = v {
                    if !live[j as usize] {
                        live[j as usize] = true;
                        work.push(j);
                    }
                }
            });
        }
        for b in &mut self.blocks {
            b.retain(|&i| live[i as usize]);
        }
    }
}

/// The value `v` stands for after replacements (path-compressing the chain).
fn resolve(repl: &mut [Option<Value>], v: Value) -> Value {
    let mut r = v;
    while let Value::Inst(i) = r {
        match repl[i as usize] {
            Some(n) => r = n,
            None => break,
        }
    }
    let mut c = v;
    while let Value::Inst(i) = c {
        match repl[i as usize] {
            Some(n) if n != r => {
                repl[i as usize] = Some(r);
                c = n;
            }
            _ => break,
        }
    }
    r
}
