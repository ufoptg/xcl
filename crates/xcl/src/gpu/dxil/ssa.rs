//! On-the-fly SSA construction (Braun, Buchwald, Hack, Leißa, Mallon, Zwinkau:
//! "Simple and Efficient Construction of Static Single Assignment Form", CC 2013)
//! over the module's basic blocks. Variables are register dwords (SGPRs, SCC,
//! VCC, M0, EXEC, VGPRs) numbered by the translator.
//!
//! Blocks are registered as they are created; predecessors are added as the
//! branches are emitted. A block whose predecessors are not all known yet is
//! *unsealed*: reads there create incomplete phis, completed on [`Ssa::seal`].
//! A read that reaches a block without predecessors asks the caller for the
//! variable's initial value (the kernel loads it from `io` in the entry block).

use std::collections::HashMap;

use super::module::{Block, Ty, Value};
use super::shader::Shader;

/// Multiplicative hash for the small integer keys of the definition maps (SipHash
/// dominated the reads of long programs).
#[derive(Default, Clone, Copy)]
struct VarHasher(u64);

impl std::hash::Hasher for VarHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
        }
    }
    fn write_u32(&mut self, v: u32) {
        self.0 = (self.0.rotate_left(5) ^ v as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

type VarMap = HashMap<u32, Value, std::hash::BuildHasherDefault<VarHasher>>;

#[derive(Default)]
struct SBlock {
    defs: VarMap,
    preds: Vec<Block>,
    sealed: bool,
    /// predecessors to wait for before sealing automatically (None: sealed explicitly)
    expect: Option<usize>,
    incomplete: Vec<(u32, Value)>,
    /// a region join ([`Ssa::seal_region_join`]): the region's last block and the
    /// variables it writes
    region: Option<(Block, Vec<u32>)>,
}

pub struct Ssa {
    blocks: Vec<SBlock>,
    ty: Ty,
    /// (block, variable) of the sealed joins whose operands are being read, with the
    /// phi created if a read came back around to the join (a loop)
    visiting: HashMap<(u32, u32), Option<Value>>,
    /// open write logs (innermost last)
    logs: Vec<Vec<u32>>,
}

impl Ssa {
    /// All variables have type `ty`.
    pub fn new(ty: Ty) -> Ssa {
        Ssa { blocks: Vec::new(), ty, visiting: HashMap::new(), logs: Vec::new() }
    }

    fn at(&mut self, b: Block) -> &mut SBlock {
        let i = b.0 as usize;
        if self.blocks.len() <= i {
            self.blocks.resize_with(i + 1, SBlock::default);
        }
        &mut self.blocks[i]
    }

    /// `b` seals itself once `n` predecessors have been added.
    pub fn expect_preds(&mut self, s: &mut Shader, b: Block, n: usize, init: &mut dyn FnMut(&mut Shader, u32) -> Value) {
        self.at(b).expect = Some(n);
        if self.at(b).preds.len() >= n && !self.at(b).sealed {
            self.seal(s, b, init);
        }
    }

    pub fn add_pred(&mut self, s: &mut Shader, b: Block, p: Block, init: &mut dyn FnMut(&mut Shader, u32) -> Value) {
        let sb = self.at(b);
        assert!(!sb.sealed, "edge into the sealed block {b:?}");
        sb.preds.push(p);
        if let Some(n) = sb.expect {
            if sb.preds.len() >= n {
                self.seal(s, b, init);
            }
        }
    }

    pub fn is_sealed(&mut self, b: Block) -> bool {
        self.at(b).sealed
    }

    pub fn preds(&mut self, b: Block) -> Vec<Block> {
        self.at(b).preds.clone()
    }

    /// A definition by the program (logged for an open [`Ssa::begin_log`]).
    pub fn write(&mut self, b: Block, var: u32, v: Value) {
        if let Some(log) = self.logs.last_mut() {
            log.push(var);
        }
        self.at(b).defs.insert(var, v);
    }

    /// A read's result remembered in `b` (not a definition).
    fn cache(&mut self, b: Block, var: u32, v: Value) {
        self.at(b).defs.insert(var, v);
    }

    /// Starts logging the variables written (e.g. in a guarded region).
    pub fn begin_log(&mut self) {
        self.logs.push(Vec::new());
    }

    /// Ends the innermost log: the variables written since `begin_log` (also
    /// counted in the enclosing log).
    pub fn end_log(&mut self) -> Vec<u32> {
        let mut log = self.logs.pop().expect("begin_log");
        log.sort_unstable();
        log.dedup();
        if let Some(outer) = self.logs.last_mut() {
            outer.extend_from_slice(&log);
        }
        log
    }

    /// Seals the join `j` of a region `cond → (then … end) → j` / `cond → j` whose
    /// only writes are `written` (sorted): a written variable merges over both edges,
    /// every other one reads through `cond` directly, as if `cond` were `j`'s only
    /// predecessor. Reads then skip the region instead of walking both edges of every
    /// guard (guarded memory accesses make thousands of such joins).
    pub fn seal_region_join(&mut self, j: Block, cond: Block, end: Block, written: Vec<u32>) {
        let sb = self.at(j);
        assert!(!sb.sealed && sb.incomplete.is_empty(), "region join already read");
        sb.preds = vec![cond];
        sb.sealed = true;
        if !written.is_empty() {
            sb.region = Some((end, written));
        }
    }

    /// The predecessors a read of `var` in `b` merges over.
    fn preds_of(&mut self, b: Block, var: u32) -> Vec<Block> {
        let sb = self.at(b);
        match &sb.region {
            Some((end, w)) if w.binary_search(&var).is_ok() => vec![sb.preds[0], *end],
            _ => sb.preds.clone(),
        }
    }

    /// The definition of `var` reaching the current end of block `b`.
    pub fn read(&mut self, s: &mut Shader, b: Block, var: u32, init: &mut dyn FnMut(&mut Shader, u32) -> Value) -> Value {
        struct Frame {
            chain: Vec<Block>,
            b: Block,
            preds: Vec<Block>,
            vals: Vec<Value>,
        }
        let mut stack: Vec<Frame> = Vec::new();
        let mut cur = b;
        loop {
            let resolved = if let Some(&v) = self.at(cur).defs.get(&var) {
                Some(v)
            } else {
                // chains of sealed single-predecessor blocks (guarded memory accesses make long ones)
                // are walked iteratively, not recursively
                let mut chain = Vec::new();
                let mut b = cur;
                let hit = loop {
                    let sb = self.at(b);
                    if !sb.sealed || sb.preds.len() != 1 || sb.region.as_ref().is_some_and(|(_, w)| w.binary_search(&var).is_ok()) {
                        break None;
                    }
                    chain.push(b);
                    b = sb.preds[0];
                    if let Some(&v) = self.at(b).defs.get(&var) {
                        break Some(v);
                    }
                };
                if let Some(v) = hit {
                    for c in chain {
                        self.cache(c, var, v);
                    }
                    Some(v)
                } else if !self.at(b).sealed {
                    let ty = self.ty;
                    let phi = s.m.phi_at(b, ty);
                    self.at(b).incomplete.push((var, phi));
                    self.cache(b, var, phi);
                    for c in chain {
                        self.cache(c, var, phi);
                    }
                    Some(phi)
                } else {
                    let preds = self.preds_of(b, var);
                    match preds.len() {
                        0 => {
                            let v = init(s, var);
                            self.cache(b, var, v);
                            for c in chain {
                                self.cache(c, var, v);
                            }
                            Some(v)
                        }
                        1 => {
                            let next = preds[0];
                            stack.push(Frame { chain, b, preds, vals: Vec::with_capacity(1) });
                            cur = next;
                            continue;
                        }
                        _ => {
                            // the operands first, without a phi: most joins a read walks through
                            // (guarded accesses make thousands) see one value from every edge
                            // (Braun et al.'s trivial phi) and need none. A read that comes back
                            // here around a loop gets a phi, which then stays.
                            let key = (b.0, var);
                            if let Some(slot) = self.visiting.get(&key).copied() {
                                let phi = slot.unwrap_or_else(|| s.m.detached_phi(self.ty));
                                self.visiting.insert(key, Some(phi));
                                for c in chain {
                                    self.cache(c, var, phi);
                                }
                                Some(phi)
                            } else {
                                self.visiting.insert(key, None);
                                let next = preds[0];
                                let cap = preds.len();
                                stack.push(Frame { chain, b, preds, vals: Vec::with_capacity(cap) });
                                cur = next;
                                continue;
                            }
                        }
                    }
                }
            };

            let mut val = resolved.unwrap();
            loop {
                let Some(top) = stack.last_mut() else {
                    return val;
                };
                top.vals.push(val);
                if top.vals.len() < top.preds.len() {
                    cur = top.preds[top.vals.len()];
                    break;
                }
                let frame = stack.pop().unwrap();
                val = if frame.preds.len() == 1 {
                    frame.vals[0]
                } else {
                    let key = (frame.b.0, var);
                    let marker = self.visiting.remove(&key).flatten();
                    if marker.is_none() && frame.vals.iter().all(|&v| v == frame.vals[0]) {
                        frame.vals[0]
                    } else {
                        let phi = marker.unwrap_or_else(|| s.m.detached_phi(self.ty));
                        for (&v, &p) in frame.vals.iter().zip(&frame.preds) {
                            s.m.add_incoming(phi, v, p);
                        }
                        s.m.place_phi(frame.b, phi);
                        phi
                    }
                };
                self.cache(frame.b, var, val);
                for c in frame.chain {
                    self.cache(c, var, val);
                }
            }
        }
    }

    fn add_operands(&mut self, s: &mut Shader, phi: Value, preds: &[Block], var: u32, init: &mut dyn FnMut(&mut Shader, u32) -> Value) {
        for &p in preds {
            let v = self.read(s, p, var, init);
            s.m.add_incoming(phi, v, p);
        }
    }

    pub fn seal(&mut self, s: &mut Shader, b: Block, init: &mut dyn FnMut(&mut Shader, u32) -> Value) {
        if self.at(b).sealed {
            return;
        }
        let preds = self.at(b).preds.clone();
        let inc = std::mem::take(&mut self.at(b).incomplete);
        for (var, phi) in inc {
            self.add_operands(s, phi, &preds, var, init);
        }
        self.at(b).sealed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::super::container::Stage;
    use super::super::module::BinOp;
    use super::*;

    #[test]
    fn deep_join_chain_does_not_overflow_stack() {
        let handle = std::thread::Builder::new()
            .name("ssa-deep-join-test".into())
            .stack_size(2 * 1024 * 1024)
            .spawn(|| {
                let mut s = Shader::new(Stage::Compute, (6, 6));
                let i32_ty = s.m.i32();
                let mut ssa = Ssa::new(i32_ty);
                let mut init = |sh: &mut Shader, _v: u32| sh.m.i32c(0);

                let entry = Block(0);
                ssa.expect_preds(&mut s, entry, 0, &mut init);
                let initial = s.m.i32c(42);
                ssa.write(entry, 0, initial);
                let cond = s.m.i1c(true);
                let one = s.m.i32c(1);

                // 4,000 sequential if/else diamonds: var 0 is untouched across the chain,
                // var 1 is updated on one arm of each diamond so non-trivial phis form.
                ssa.write(entry, 1, initial);
                let mut cur = entry;
                for _ in 0..4000 {
                    let then_b = s.m.new_block();
                    let else_b = s.m.new_block();
                    let join_b = s.m.new_block();
                    ssa.expect_preds(&mut s, then_b, 1, &mut init);
                    ssa.expect_preds(&mut s, else_b, 1, &mut init);
                    ssa.expect_preds(&mut s, join_b, 2, &mut init);

                    s.m.set_block(cur);
                    s.m.cond_br(cond, then_b, else_b);
                    ssa.add_pred(&mut s, then_b, cur, &mut init);
                    ssa.add_pred(&mut s, else_b, cur, &mut init);

                    s.m.set_block(then_b);
                    let prev = ssa.read(&mut s, then_b, 1, &mut init);
                    let next = s.m.bin(BinOp::Add, prev, one);
                    ssa.write(then_b, 1, next);
                    s.m.br(join_b);
                    ssa.add_pred(&mut s, join_b, then_b, &mut init);

                    s.m.set_block(else_b);
                    s.m.br(join_b);
                    ssa.add_pred(&mut s, join_b, else_b, &mut init);

                    cur = join_b;
                }

                s.m.set_block(cur);
                let got_unmodified = ssa.read(&mut s, cur, 0, &mut init);
                assert_eq!(got_unmodified, initial);
                let got_modified = ssa.read(&mut s, cur, 1, &mut init);
                assert_ne!(got_modified, initial);
                s.m.ret();
                let bc = s.try_finish([1, 1, 1], None).expect("valid DXIL");
                assert!(!bc.is_empty());
            })
            .expect("spawn bounded-stack thread");
        handle.join().expect("deep join chain overflowed or panicked");
    }

    #[test]
    fn deep_region_join_and_loop_phi_preserve_semantics() {
        let handle = std::thread::Builder::new()
            .name("ssa-region-loop-test".into())
            .stack_size(2 * 1024 * 1024)
            .spawn(|| {
                let mut s = Shader::new(Stage::Compute, (6, 6));
                let i32_ty = s.m.i32();
                let mut ssa = Ssa::new(i32_ty);
                let mut init = |sh: &mut Shader, _v: u32| sh.m.i32c(0);

                let entry = Block(0);
                ssa.expect_preds(&mut s, entry, 0, &mut init);
                let v0 = s.m.i32c(7);
                let v1 = s.m.i32c(9);
                ssa.write(entry, 0, v0);
                ssa.write(entry, 1, v1);
                let cond = s.m.i1c(true);
                let one = s.m.i32c(1);

                // Loop header (unsealed until back-edge) followed by 3,000 region joins.
                let hdr = s.m.new_block();
                s.m.set_block(entry);
                s.m.br(hdr);
                ssa.add_pred(&mut s, hdr, entry, &mut init);

                let mut cur = hdr;
                for _ in 0..3000 {
                    let then_b = s.m.new_block();
                    let join_b = s.m.new_block();
                    s.m.set_block(cur);
                    s.m.cond_br(cond, then_b, join_b);
                    ssa.expect_preds(&mut s, then_b, 1, &mut init);
                    ssa.add_pred(&mut s, then_b, cur, &mut init);
                    s.m.set_block(then_b);
                    ssa.begin_log();
                    let x = ssa.read(&mut s, then_b, 1, &mut init);
                    let x1 = s.m.bin(BinOp::Add, x, one);
                    ssa.write(then_b, 1, x1);
                    let written = ssa.end_log();
                    s.m.br(join_b);
                    ssa.seal_region_join(join_b, cur, then_b, written);
                    cur = join_b;
                }

                s.m.set_block(cur);
                let back_val = ssa.read(&mut s, cur, 0, &mut init);
                let next0 = s.m.bin(BinOp::Add, back_val, one);
                ssa.write(cur, 0, next0);
                let exit = s.m.new_block();
                ssa.expect_preds(&mut s, exit, 1, &mut init);
                s.m.cond_br(cond, hdr, exit);
                ssa.add_pred(&mut s, hdr, cur, &mut init);
                ssa.seal(&mut s, hdr, &mut init);
                ssa.add_pred(&mut s, exit, cur, &mut init);

                s.m.set_block(exit);
                let final0 = ssa.read(&mut s, exit, 0, &mut init);
                assert_ne!(final0, v0);
                let final1 = ssa.read(&mut s, exit, 1, &mut init);
                assert_ne!(final1, v1);
                s.m.ret();
                let bc = s.try_finish([1, 1, 1], None).expect("valid DXIL");
                assert!(!bc.is_empty());
            })
            .expect("spawn bounded-stack thread");
        handle.join().expect("region join + loop test failed");
    }
}
