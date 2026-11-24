//! Basic blocks of a decoded RDNA2 program, their edges and a reverse
//! post-order (the emission order: every block after its dominators).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::lower::{Dst, Lw, NOOPS, VAR_SCC};
use crate::gpu::rdna2::{self, Decoded, Variant};

/// Minimum reachable block count before collapsing forward `s_cbranch_execz` regions.
/// Small shaders keep hardware wave-skipping `s_cbranch_execz` branches; giant unrolled
/// shaders (hundreds/thousands of blocks) collapse them to avoid O(B^2) driver compile OOM.
const EXECZ_COLLAPSE_MIN_BLOCKS: usize = 128;

/// Maximum ops in a single collapsed `s_cbranch_execz` body (keeps huge whole-shader
/// early-out branches intact for GPU performance while collapsing unrolled inner regions).
const MAX_COLLAPSE_OPS: usize = 2048;

/// One operation inside a CFG block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CfgOp {
    /// Guest instruction index in `Cfg::insts`.
    Inst(usize),
    /// Snapshot `EXEC == 0` and pre-region values of `regs` (non-EXEC scalar regs + SCC)
    /// that are modified in a collapsed `s_cbranch_execz` region and live-out at its target.
    SaveExecZ { id: u32, regs: Vec<u32> },
    /// Restore `regs` via `select(was_exec0, saved, cur)` at the end of a collapsed region.
    RestoreIfExecZ { id: u32, regs: Vec<u32> },
}

/// How a block ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term {
    /// falls through / jumps to the block (index)
    Jump(usize),
    /// conditional branch: (instruction index of the branch, taken, not taken)
    Cond(usize, usize, usize),
    /// s_endpgm or the end of the code
    End,
}

pub struct CfgBlock {
    /// byte offset of the block's first instruction
    pub pc: u64,
    /// operations in emission order (the terminating branch excluded for `Cond`/`Jump` by branch)
    pub ops: Vec<CfgOp>,
    pub term: Term,
    pub preds: Vec<usize>,
}

pub struct Cfg {
    pub insts: Vec<Decoded>,
    pub blocks: Vec<CfgBlock>,
    /// reachable blocks in reverse post-order (blocks[rpo[0]] is the entry)
    pub rpo: Vec<usize>,
    /// targets of back edges (loop headers)
    pub headers: BTreeSet<usize>,
    /// original basic-block leader PCs before `s_cbranch_execz` collapsing (for `constant_samplers`)
    pub orig_leaders: BTreeSet<u64>,
}

fn is_end(name: &str) -> bool {
    matches!(name, "S_ENDPGM" | "S_ENDPGM_SAVED" | "S_ENDPGM_ORDERED_PS_DONE")
}

impl Cfg {
    pub fn build(code: &[u8]) -> Result<Cfg, String> {
        Self::build_for(code, true, true)
    }

    pub fn build_for(code: &[u8], wave64: bool, live_at_end: bool) -> Result<Cfg, String> {
        Self::build_with(code, None, wave64, live_at_end, EXECZ_COLLAPSE_MIN_BLOCKS)
    }

    /// Geometry mode: the ES's `s_setpc_b64` continues at `gs_start` (the GS code) when
    /// that is past the start, else ends the program (`recomp::Mode::Geometry`).
    pub fn build_geometry(code: &[u8], gs_start: Option<u64>) -> Result<Cfg, String> {
        Self::build_with(code, Some(gs_start.filter(|&s| s > 0)), true, false, EXECZ_COLLAPSE_MIN_BLOCKS)
    }

    pub(crate) fn build_with_threshold(code: &[u8], wave64: bool, live_at_end: bool, min_blocks: usize) -> Result<Cfg, String> {
        Self::build_with(code, None, wave64, live_at_end, min_blocks)
    }

    fn build_with(code: &[u8], setpc: Option<Option<u64>>, wave64: bool, live_at_end: bool, min_blocks: usize) -> Result<Cfg, String> {
        let mut insts: Vec<Decoded> = Vec::new();
        let mut off = 0usize;
        while off + 4 <= code.len() {
            let d = rdna2::decode(&code[off..], off as u64).map_err(|e| format!("+{off:#x}: cannot decode ({e:?})"))?;
            off += d.len;
            insts.push(d);
        }
        rdna2::resolve_far_jumps(&mut insts);
        let index: BTreeMap<u64, usize> = insts.iter().enumerate().map(|(i, d)| (d.pc, i)).collect();
        let mut leaders: BTreeSet<usize> = BTreeSet::from([0]);
        let is_setpc = |d: &Decoded| setpc.is_some() && d.inst.name == "S_SETPC_B64" && d.far_target.is_none();
        let setpc_target = match setpc {
            Some(Some(t)) => Some(*index.get(&t).ok_or_else(|| format!("GS start {t:#x} is not the start of an instruction"))?),
            _ => None,
        };
        if let Some(t) = setpc_target {
            leaders.insert(t);
        }
        for (i, d) in insts.iter().enumerate() {
            if is_setpc(d) {
                leaders.insert(i + 1);
                continue;
            }
            if let Some(t) = rdna2::branch_target(d) {
                let ti = *index.get(&t).ok_or_else(|| format!("+{:#x}: branch to {t:#x}, which is not the start of an instruction", d.pc))?;
                leaders.insert(ti);
                leaders.insert(i + 1);
            } else if d.inst.flags & rdna2::F_TERMINATOR != 0 || is_end(d.inst.name) {
                leaders.insert(i + 1);
            }
        }
        let starts: Vec<usize> = leaders.into_iter().filter(|&l| l < insts.len()).collect();
        let block_of: BTreeMap<usize, usize> = starts.iter().enumerate().map(|(b, &i)| (i, b)).collect();
        let mut blocks = Vec::new();
        for (b, &s) in starts.iter().enumerate() {
            let e = starts.get(b + 1).copied().unwrap_or(insts.len());
            let mut list = Vec::new();
            let mut term = if e < insts.len() { Term::Jump(b + 1) } else { Term::End };
            for i in s..e {
                let d = &insts[i];
                if is_end(d.inst.name) {
                    term = Term::End;
                    break;
                }
                if is_setpc(d) {
                    term = match setpc_target {
                        Some(t) => Term::Jump(block_of[&t]),
                        None => Term::End,
                    };
                    break;
                }
                if let Some(t) = rdna2::branch_target(d) {
                    let tb = block_of[&index[&t]];
                    let name = d.inst.name;
                    term = if name == "S_BRANCH" || d.far_target.is_some() {
                        Term::Jump(tb)
                    } else {
                        let next = if e < insts.len() { b + 1 } else { usize::MAX };
                        if next == usize::MAX {
                            return Err(format!("+{:#x}: conditional branch at the end of the code", d.pc));
                        }
                        Term::Cond(i, tb, next)
                    };
                    break;
                }
                list.push(CfgOp::Inst(i));
            }
            blocks.push(CfgBlock { pc: insts[s].pc, ops: list, term, preds: Vec::new() });
        }
        // a conditional branch whose both edges meet is a jump
        for b in &mut blocks {
            if let Term::Cond(_, t, f) = b.term {
                if t == f {
                    b.term = Term::Jump(t);
                }
            }
        }
        // DXIL requires reducible control flow (AMD's compiler faults on an irreducible
        // CFG; the console compiler emits loops entered in the middle): split nodes until
        // every retreating edge is a back edge to a dominator
        let (mut post, mut headers) = make_reducible(&mut blocks)?;
        let orig_leaders: BTreeSet<u64> = blocks
            .iter()
            .filter_map(|b| b.ops.first().and_then(|op| match *op { CfgOp::Inst(i) => Some(insts[i].pc), _ => None }))
            .collect();
        if post.len() >= min_blocks {
            if collapse_execz_regions(&insts, &mut blocks, &post, &headers, wave64, live_at_end) {
                (post, headers) = order(&blocks);
            }
        }
        for &b in &post {
            for t in succ_of(&blocks[b]) {
                if !blocks[t].preds.contains(&b) {
                    blocks[t].preds.push(b);
                }
            }
        }
        Ok(Cfg { insts, blocks, rpo: post, headers, orig_leaders })
    }
}

/// Collapses provably side-effect-free forward `S_CBRANCH_EXECZ` diamonds and merges
/// resulting single-predecessor `Jump` chains in O(N) reverse-RPO order.
fn collapse_execz_regions(
    insts: &[Decoded],
    blocks: &mut [CfgBlock],
    rpo: &[usize],
    headers: &BTreeSet<usize>,
    wave64: bool,
    live_at_end: bool,
) -> bool {
    let n = blocks.len();
    let mut pos = vec![usize::MAX; n];
    for (idx, &b) in rpo.iter().enumerate() {
        pos[b] = idx;
    }
    // Normalize any degenerate Cond(_, t, t) before counting preds.
    for &b in rpo {
        if let Term::Cond(_, t, f) = blocks[b].term {
            if t == f {
                blocks[b].term = Term::Jump(t);
            }
        }
    }
    let live_in = compute_scalar_liveness(insts, blocks, rpo, wave64, live_at_end);
    let mut pred_count = vec![0usize; n];
    if let Some(&entry) = rpo.first() {
        pred_count[entry] = 1; // keep entry from being absorbed
    }
    for &b in rpo {
        for t in succ_of(&blocks[b]) {
            pred_count[t] += 1;
        }
    }
    let exec_mask: u128 = if wave64 { (1u128 << 126) | (1u128 << 127) } else { 1u128 << 126 };
    let mut next_id = 0u32;
    let mut changed = false;
    for &b in rpo.iter().rev() {
        loop {
            match blocks[b].term {
                Term::Jump(nxt) if nxt != b && pred_count[nxt] == 1 && !headers.contains(&nxt) && pos[nxt] > pos[b] => {
                    let tail_ops = std::mem::take(&mut blocks[nxt].ops);
                    let tail_term = std::mem::replace(&mut blocks[nxt].term, Term::End);
                    blocks[b].ops.extend(tail_ops);
                    blocks[b].term = tail_term;
                    pred_count[nxt] = 0;
                    changed = true;
                }
                Term::Cond(br, target, fall)
                    if insts[br].inst.name == "S_CBRANCH_EXECZ"
                        && pos[target] > pos[b]
                        && pos[fall] > pos[b]
                        && pred_count[fall] == 1
                        && !headers.contains(&fall)
                        && blocks[fall].term == Term::Jump(target) =>
                {
                    let Some(written_scalar) = analyze_execz_region(&blocks[fall].ops, insts, wave64) else {
                        break;
                    };
                    let restore_mask = written_scalar & live_in[target] & !exec_mask;
                    let body_ops = std::mem::take(&mut blocks[fall].ops);
                    blocks[fall].term = Term::End;
                    if restore_mask != 0 {
                        let id = next_id;
                        next_id += 1;
                        let regs: Vec<u32> = (0..128u32).filter(|&r| (restore_mask >> r) & 1 != 0).collect();
                        blocks[b].ops.push(CfgOp::SaveExecZ { id, regs: regs.clone() });
                        blocks[b].ops.extend(body_ops);
                        blocks[b].ops.push(CfgOp::RestoreIfExecZ { id, regs });
                    } else {
                        blocks[b].ops.extend(body_ops);
                    }
                    blocks[b].term = Term::Jump(target);
                    pred_count[fall] = 0;
                    pred_count[target] = pred_count[target].saturating_sub(1);
                    changed = true;
                }
                _ => break,
            }
        }
    }
    changed
}

/// Bitmask helper for scalar register codes `0..128` (`125` = `VAR_SCC`).
#[inline]
fn reg_bit(r: u64) -> u128 {
    if r < 128 { 1u128 << r } else { 0 }
}

/// Bitmask for `n` consecutive scalar registers starting at `r` (skipping NULL = 125).
#[inline]
fn sreg_range(r: u64, n: u64) -> u128 {
    let mut m = 0u128;
    for k in 0..n {
        let c = r + k;
        if c < 128 && c != 125 {
            m |= 1u128 << c;
        }
    }
    m
}

/// Conservative scalar read mask and exact unconditional scalar must-write mask for liveness.
/// Bit `125` represents `VAR_SCC`. Over-approximating reads and under-approximating must-writes
/// is always sound.
fn inst_scalar_rw(d: &Decoded, wave64: bool) -> (u128, u128) {
    let name = d.inst.name.to_ascii_lowercase();
    let mut r = 0u128;
    let mut w = 0u128;
    let src_code_mask = |code: u64, n: u64| -> u128 {
        match code {
            0..=124 | 126 | 127 => sreg_range(code, n),
            251 => reg_bit(106) | reg_bit(107),
            252 => reg_bit(126) | reg_bit(127),
            253 => reg_bit(VAR_SCC as u64),
            _ => 0,
        }
    };
    for o in d.ienc.operands {
        if let Some(f) = o.field {
            if f == "LITERAL" {
                continue;
            }
            if f == "SRC0" && d.variant != Variant::Plain {
                continue;
            }
            let Some(raw) = rdna2::field_value(d.enc, &d.words, f) else { continue };
            let v = raw * rdna2::register_scale(f);
            let n = (o.size as u64).div_ceil(32).max(1);
            let is_scalar_op = ["SREG", "SSRC", "SDST", "OPR_SRC", "SMEM", "SADDR"].iter().any(|k| o.ty.contains(k))
                || matches!(f, "SBASE" | "SRSRC" | "SSAMP" | "SOFFSET" | "SADDR");
            if is_scalar_op && o.input {
                r |= src_code_mask(v, n);
            }
        } else {
            match o.ty {
                "OPR_VCC" if o.input => r |= reg_bit(106) | reg_bit(107),
                "OPR_EXEC" | "OPR_SDST_EXEC" if o.input => r |= reg_bit(126) | reg_bit(127),
                "OPR_SDST_M0" if o.input => r |= reg_bit(124),
                "OPR_SSRC_SPECIAL_SCC" if o.input => r |= reg_bit(VAR_SCC as u64),
                _ => {}
            }
        }
    }
    if d.variant == Variant::Sdwa {
        if let Some(e) = super::lower::ext_encoding(d) {
            if rdna2::field_value(e, &d.words, "S0") == Some(1) {
                if let Some(v) = rdna2::field_value(e, &d.words, "VSRC0") {
                    r |= src_code_mask(v, 1);
                }
            }
            if rdna2::field_value(e, &d.words, "S1") == Some(1) {
                if let Some(v) = rdna2::field_value(e, &d.words, "VSRC1") {
                    r |= src_code_mask(v, 1);
                }
            }
        }
    }
    // Special / implicit scalar reads
    match d.enc.name {
        "ENC_MIMG" => {
            let sr = Lw::field(d, "SRSRC");
            let ss = Lw::field(d, "SSAMP");
            r |= sreg_range(sr, 8) | sreg_range(ss, 4);
        }
        "ENC_MUBUF" | "ENC_MTBUF" => {
            let sr = Lw::field(d, "SRSRC");
            r |= sreg_range(sr, 4);
            if let Some(so) = rdna2::field_value(d.enc, &d.words, "SOFFSET") {
                r |= src_code_mask(so, 1);
            }
        }
        "ENC_SMEM" => {
            let sb = Lw::field(d, "SBASE");
            r |= sreg_range(sb, 4);
            if let Some(so) = rdna2::field_value(d.enc, &d.words, "SOFFSET") {
                r |= src_code_mask(so, 1);
            }
        }
        "ENC_FLAT_GLBL" => {
            let sa = Lw::fv(d, "SADDR");
            if sa != 0x7D && sa != 0x7F {
                r |= sreg_range(sa, 2);
            }
        }
        "ENC_DS" | "ENC_VINTRP" => {
            r |= reg_bit(124);
        }
        _ => {}
    }
    if name.starts_with("s_movrel") {
        // Indirect SGPR read/write via M0: conservatively reads all SGPRs + M0.
        r |= (1u128 << 125) - 1;
        return (r, 0);
    }
    if matches!(name.as_str(), "s_sendmsg" | "s_sendmsghalt") {
        r |= reg_bit(124);
    }
    if name.starts_with("s_cmov") || name.starts_with("s_cselect") || name.starts_with("s_addc") || name.starts_with("s_subb") {
        r |= reg_bit(VAR_SCC as u64);
    }
    if matches!(name.as_str(), "s_cbranch_scc0" | "s_cbranch_scc1") {
        r |= reg_bit(VAR_SCC as u64);
    }
    if matches!(name.as_str(), "s_cbranch_vccz" | "s_cbranch_vccnz") {
        r |= reg_bit(106) | reg_bit(107);
    }
    if matches!(name.as_str(), "s_cbranch_execz" | "s_cbranch_execnz") {
        r |= reg_bit(126) | if wave64 { reg_bit(127) } else { 0 };
    }
    if matches!(name.as_str(), "v_cndmask_b32" | "v_add_co_ci_u32" | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32") {
        r |= reg_bit(106) | reg_bit(107);
    }
    if name.contains("saveexec") || name.contains("wrexec") {
        r |= reg_bit(126) | if wave64 { reg_bit(127) } else { 0 };
    }
    if d.enc.name == "ENC_SOPK" && name != "s_movk_i32" && !name.starts_with("s_waitcnt") && name != "s_version" {
        let sd = Lw::field(d, "SDST");
        r |= src_code_mask(sd, 1);
    }
    if matches!(name.as_str(), "s_setreg_b32") {
        let sd = Lw::field(d, "SDST");
        r |= src_code_mask(sd, 1);
    }
    if name.starts_with("s_bitset") || name.starts_with("s_cmov") {
        let (outs, _) = Lw::ops(d);
        if let Some(Dst::S(n)) = outs.first().map(|o| Lw::dst_of(d, o)) {
            let sz = if name.ends_with("_b64") { 2 } else { 1 };
            r |= sreg_range(n, sz);
        }
    }
    // Exact unconditional scalar must-writes (conservative: only guaranteed overwrites)
    if NOOPS.contains(&name.as_str()) {
        return (r, 0);
    }
    match d.enc.name {
        "ENC_SOP1" | "ENC_SOP2" => {
            if matches!(name.as_str(), "s_setreg_imm32_b32" | "s_setreg_b32" | "s_getreg_b32" | "s_swappc_b64" | "s_setpc_b64") {
                return (r, 0);
            }
            let (outs, _) = Lw::ops(d);
            let dst = outs.first().map(|o| Lw::dst_of(d, o)).unwrap_or(Dst::Null);
            if name.contains("saveexec") || name.contains("wrexec") {
                let words = if !wave64 || name.ends_with("_b32") { 1 } else { 2 };
                w |= sreg_range(126, words) | reg_bit(VAR_SCC as u64);
                if let Dst::S(n) = dst {
                    w |= sreg_range(n, words);
                }
            } else {
                let wide = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
                let one_word = matches!(
                    name.as_str(),
                    "s_bcnt0_i32_b64" | "s_bcnt1_i32_b64" | "s_ff0_i32_b64" | "s_ff1_i32_b64" | "s_flbit_i32_b64" | "s_flbit_i32_i64"
                );
                let words = if wide && !one_word { 2 } else { 1 };
                // s_cmov* is conditional, already marked as reading dst above; unconditional write of dst still holds (reads old dst)
                if let Dst::S(n) = dst {
                    w |= sreg_range(n, words);
                }
                if salu_writes_scc(&name) {
                    w |= reg_bit(VAR_SCC as u64);
                }
            }
        }
        "ENC_SOPC" => {
            w |= reg_bit(VAR_SCC as u64);
        }
        "ENC_SOPK" => {
            if name.starts_with("s_cmpk_") {
                w |= reg_bit(VAR_SCC as u64);
            } else if matches!(name.as_str(), "s_movk_i32" | "s_cmovk_i32" | "s_addk_i32" | "s_mulk_i32") {
                let sd = Lw::field(d, "SDST");
                w |= sreg_range(sd, 1);
                if name == "s_addk_i32" {
                    w |= reg_bit(VAR_SCC as u64);
                }
            }
        }
        "ENC_SMEM" => {
            if let Some(cnt) = smem_dst_count(&name) {
                let sd = Lw::field(d, "SDATA");
                w |= sreg_range(sd, cnt);
            }
        }
        e if e.starts_with("ENC_VOP") || e.starts_with("VOP") => {
            let mw = if wave64 { 2 } else { 1 };
            if name.starts_with("v_cmpx_") {
                w |= sreg_range(126, mw);
            } else if name.starts_with("v_cmp_") && d.variant == Variant::Plain {
                let (outs, _) = Lw::ops(d);
                let dst = outs.first().map(|o| Lw::dst_of(d, o)).unwrap_or(Dst::S(106));
                if let Dst::S(n) = dst {
                    w |= sreg_range(n, mw);
                }
            } else if matches!(
                name.as_str(),
                "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_add_co_ci_u32" | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32"
            ) {
                let (outs, _) = Lw::ops(d);
                let sdst = outs.get(1).map(|o| Lw::dst_of(d, o)).unwrap_or(Dst::S(106));
                if let Dst::S(n) = sdst {
                    w |= sreg_range(n, mw);
                }
            } else if matches!(name.as_str(), "v_mad_u64_u32" | "v_mad_i64_i32") {
                let (outs, _) = Lw::ops(d);
                if let Some(Dst::S(n)) = outs.get(1).map(|o| Lw::dst_of(d, o)) {
                    w |= sreg_range(n, mw);
                }
            } else if matches!(name.as_str(), "v_readfirstlane_b32" | "v_readlane_b32") {
                let (outs, _) = Lw::ops(d);
                if let Some(Dst::S(n)) = outs.first().map(|o| Lw::dst_of(d, o)) {
                    w |= sreg_range(n, 1);
                }
            }
        }
        _ => {}
    }
    (r, w)
}

fn salu_writes_scc(name: &str) -> bool {
    !matches!(
        name,
        "s_mov_b32"
            | "s_mov_b64"
            | "s_cmov_b32"
            | "s_cmov_b64"
            | "s_cselect_b32"
            | "s_cselect_b64"
            | "s_mul_i32"
            | "s_mul_hi_u32"
            | "s_mul_hi_i32"
            | "s_brev_b32"
            | "s_brev_b64"
            | "s_bfm_b32"
            | "s_bfm_b64"
            | "s_bitset0_b32"
            | "s_bitset1_b32"
            | "s_bitset0_b64"
            | "s_bitset1_b64"
            | "s_sext_i32_i8"
            | "s_sext_i32_i16"
            | "s_ff0_i32_b32"
            | "s_ff1_i32_b32"
            | "s_flbit_i32_b32"
            | "s_flbit_i32"
            | "s_ff0_i32_b64"
            | "s_ff1_i32_b64"
            | "s_flbit_i32_b64"
            | "s_flbit_i32_i64"
            | "s_pack_ll_b32_b16"
            | "s_pack_lh_b32_b16"
            | "s_pack_hh_b32_b16"
            | "s_getpc_b64"
    )
}

fn smem_dst_count(name: &str) -> Option<u64> {
    if matches!(name, "s_memtime" | "s_memrealtime") {
        return Some(2);
    }
    Some(match name.trim_start_matches("s_buffer_load_").trim_start_matches("s_load_") {
        "dword" => 1,
        "dwordx2" => 2,
        "dwordx4" => 4,
        "dwordx8" => 8,
        "dwordx16" => 16,
        _ => return None,
    })
}

fn compute_scalar_liveness(
    insts: &[Decoded],
    blocks: &[CfgBlock],
    rpo: &[usize],
    wave64: bool,
    live_at_end: bool,
) -> Vec<u128> {
    let n = blocks.len();
    let mut b_use = vec![0u128; n];
    let mut b_def = vec![0u128; n];
    for &b in rpo {
        let mut u = 0u128;
        let mut def = 0u128;
        for op in &blocks[b].ops {
            if let CfgOp::Inst(i) = *op {
                let (ir, iw) = inst_scalar_rw(&insts[i], wave64);
                u |= ir & !def;
                def |= iw;
            }
        }
        if let Term::Cond(i, _, _) = blocks[b].term {
            let (ir, _) = inst_scalar_rw(&insts[i], wave64);
            u |= ir & !def;
        }
        b_use[b] = u;
        b_def[b] = def;
    }
    let end_live = if live_at_end { u128::MAX } else { 0 };
    let mut live_in = vec![0u128; n];
    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter().rev() {
            let out = match blocks[b].term {
                Term::Jump(t) => live_in[t],
                Term::Cond(_, t, f) => live_in[t] | live_in[f],
                Term::End => end_live,
            };
            let new_in = b_use[b] | (out & !b_def[b]);
            if new_in != live_in[b] {
                live_in[b] = new_in;
                changed = true;
            }
        }
    }
    live_in
}

/// Checks whether `ops` can be safely executed unconditionally when `EXEC == 0`:
/// 1. `ops.len() <= MAX_COLLAPSE_OPS`.
/// 2. Every instruction is in our strict side-effect-free allowlist (VALU, pure SALU,
///    read-only MUBUF/MTBUF/FLAT/GLOBAL loads, and NOOPS).
/// 3. Abstract interpretation proves `EXEC` remains `0` at every step when entered with
///    `EXEC == 0`, so every VGPR write and branchless read sees `act == false` (no-op).
///
/// Returns `Some(written_scalar_mask)` (all non-EXEC scalar regs + SCC that may be written)
/// if provably sound, or `None` to keep the branch intact.
fn analyze_execz_region(ops: &[CfgOp], insts: &[Decoded], wave64: bool) -> Option<u128> {
    if ops.len() > MAX_COLLAPSE_OPS {
        return None;
    }
    let exec_mask: u128 = if wave64 { (1u128 << 126) | (1u128 << 127) } else { 1u128 << 126 };
    let mut z: u128 = exec_mask;
    let mut written: u128 = 0;
    let mut saved_z: HashMap<u32, u128> = HashMap::new();

    for op in ops {
        match op {
            CfgOp::SaveExecZ { id, .. } => {
                saved_z.insert(*id, z);
            }
            CfgOp::RestoreIfExecZ { id, regs } => {
                let snap = saved_z.remove(id)?;
                for &r in regs {
                    if r < 128 {
                        written |= 1u128 << r;
                        if (snap >> r) & 1 != 0 {
                            z |= 1u128 << r;
                        } else {
                            z &= !(1u128 << r);
                        }
                    }
                }
            }
            CfgOp::Inst(i) => {
                let d = &insts[*i];
                let w = step_execz_inst(d, wave64, &mut z)?;
                written |= w;
            }
        }
        if (z & exec_mask) != exec_mask {
            return None;
        }
    }
    Some(written)
}

#[inline]
fn set_z(z: &mut u128, reg: u64, is_zero: bool) {
    if reg < 128 && reg != 125 {
        if is_zero {
            *z |= 1u128 << reg;
        } else {
            *z &= !(1u128 << reg);
        }
    }
}

/// Returns `(lo_is_zero, hi_is_zero)` for a scalar source code under `z`.
fn scalar_src_zero(d: &Decoded, code: u64, wide: bool, z: u128) -> (bool, bool) {
    match code {
        125 | 128 => (true, true),
        255 => {
            let lz = d.literal == Some(0);
            // Non-f64 64-bit scalar literals zero-extend into the high dword (`code64`).
            (lz, if wide { true } else { lz })
        }
        0..=124 | 126 => {
            let lz = (z >> code) & 1 != 0;
            let hz = if wide && code + 1 < 128 && code + 1 != 125 {
                (z >> (code + 1)) & 1 != 0
            } else {
                false
            };
            (lz, hz)
        }
        127 => ((z >> 127) & 1 != 0, false),
        _ => (false, false),
    }
}

/// Abstract step for one instruction inside a candidate `s_cbranch_execz` region.
/// Updates `z` (registers proven 0 when entered with `EXEC == 0`) and returns
/// `Some(written_scalar_mask)` if the instruction is in the side-effect-free allowlist.
fn step_execz_inst(d: &Decoded, wave64: bool, z: &mut u128) -> Option<u128> {
    let name = d.inst.name.to_ascii_lowercase();
    if matches!(name.as_str(), "s_sendmsg" | "s_sendmsghalt") {
        return None;
    }
    if NOOPS.contains(&name.as_str()) {
        return Some(0);
    }
    match d.enc.name {
        "ENC_SOP1" | "ENC_SOP2" | "ENC_SOPC" | "ENC_SOPK" => {
            if matches!(
                name.as_str(),
                "s_setreg_imm32_b32"
                    | "s_setreg_b32"
                    | "s_getreg_b32"
                    | "s_swappc_b64"
                    | "s_setpc_b64"
                    | "s_getpc_b64"
                    | "s_movrels_b32"
                    | "s_movrels_b64"
                    | "s_movreld_b32"
                    | "s_movreld_b64"
                    | "s_movrelsd_2_b32"
            ) || name.starts_with("s_call")
                || name.starts_with("s_rfe")
                || name == "s_trap"
            {
                return None;
            }
            if d.enc.name == "ENC_SOPC" {
                return Some(reg_bit(VAR_SCC as u64));
            }
            if d.enc.name == "ENC_SOPK" {
                if matches!(name.as_str(), "s_version" | "s_waitcnt_vscnt" | "s_waitcnt_vmcnt" | "s_waitcnt_expcnt" | "s_waitcnt_lgkmcnt") {
                    return Some(0);
                }
                if name.starts_with("s_cmpk_") {
                    return Some(reg_bit(VAR_SCC as u64));
                }
                let sd = Lw::field(d, "SDST");
                let simm = Lw::fv(d, "SIMM16") as u16;
                let mut w = sreg_range(sd, 1);
                match name.as_str() {
                    "s_movk_i32" => set_z(z, sd, simm == 0),
                    "s_cmovk_i32" => {
                        let cur_z = scalar_src_zero(d, sd, false, *z).0;
                        set_z(z, sd, cur_z && simm == 0);
                    }
                    "s_addk_i32" => {
                        set_z(z, sd, false);
                        w |= reg_bit(VAR_SCC as u64);
                    }
                    "s_mulk_i32" => {
                        let cur_z = scalar_src_zero(d, sd, false, *z).0;
                        set_z(z, sd, cur_z || simm == 0);
                    }
                    _ => return None,
                }
                return Some(w);
            }
            // ENC_SOP1 / ENC_SOP2
            let (outs, ins) = Lw::ops(d);
            let dst = outs.first().map(|o| Lw::dst_of(d, o)).unwrap_or(Dst::Null);
            if name.contains("saveexec") || name.contains("wrexec") {
                let b32 = name.ends_with("_b32") || !wave64;
                let words = if b32 { 1 } else { 2 };
                let keeps_exec_zero = matches!(
                    name.as_str(),
                    "s_and_saveexec_b64"
                        | "s_and_saveexec_b32"
                        | "s_andn1_saveexec_b64"
                        | "s_andn1_saveexec_b32"
                        | "s_and_wrexec_b64"
                        | "s_and_wrexec_b32"
                        | "s_andn1_wrexec_b64"
                        | "s_andn1_wrexec_b32"
                );
                if !keeps_exec_zero {
                    // Other saveexec ops (or/xor/andn2/nand/nor/xnor/orn*) can set EXEC != 0.
                    return None;
                }
                let (old_el, old_eh) = ((*z >> 126) & 1 != 0, (*z >> 127) & 1 != 0);
                // EXEC stays zero (0 & x == 0, 0 & ~x == 0); dst gets old EXEC (saveexec) or new EXEC (wrexec), both zero.
                let mut w = sreg_range(126, words) | reg_bit(VAR_SCC as u64);
                if let Dst::S(n) = dst {
                    w |= sreg_range(n, words);
                    set_z(z, n, old_el);
                    if words == 2 {
                        set_z(z, n + 1, old_eh);
                    }
                }
                set_z(z, 126, old_el);
                if words == 2 {
                    set_z(z, 127, old_eh);
                }
                return Some(w);
            }
            let wide = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
            let one_word = matches!(
                name.as_str(),
                "s_bcnt0_i32_b64" | "s_bcnt1_i32_b64" | "s_ff0_i32_b64" | "s_ff1_i32_b64" | "s_flbit_i32_b64" | "s_flbit_i32_i64"
            );
            let words = if wide && !one_word { 2 } else { 1 };
            let src_z = |idx: usize| -> (bool, bool) {
                let Some(o) = ins.get(idx) else { return (false, false) };
                let Some(f) = o.field else {
                    return match o.ty {
                        "OPR_VCC" => scalar_src_zero(d, 106, wide, *z),
                        "OPR_EXEC" | "OPR_SDST_EXEC" => scalar_src_zero(d, 126, wide, *z),
                        "OPR_SDST_M0" => scalar_src_zero(d, 124, false, *z),
                        _ => (false, false),
                    };
                };
                let code = Lw::field(d, f);
                scalar_src_zero(d, code, wide, *z)
            };
            let (a0, a1) = src_z(0);
            let (b0, b1) = src_z(1);
            let (r0, r1) = match name.as_str() {
                "s_mov_b32" | "s_mov_b64" => (a0, a1),
                "s_cmov_b32" | "s_cmov_b64" => {
                    let (c0, c1) = match dst {
                        Dst::S(n) => scalar_src_zero(d, n, wide, *z),
                        _ => (false, false),
                    };
                    (a0 && c0, a1 && c1)
                }
                "s_cselect_b32" | "s_cselect_b64" => (a0 && b0, a1 && b1),
                "s_and_b32" | "s_and_b64" => (a0 || b0, a1 || b1),
                "s_andn2_b32" | "s_andn2_b64" => (a0, a1),
                "s_or_b32" | "s_or_b64" | "s_xor_b32" | "s_xor_b64" => (a0 && b0, a1 && b1),
                "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" => (a0, false),
                "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" => (a0 && a1, a0 && a1),
                _ => (false, false),
            };
            let mut w = if salu_writes_scc(&name) { reg_bit(VAR_SCC as u64) } else { 0 };
            if let Dst::S(n) = dst {
                w |= sreg_range(n, words);
                set_z(z, n, r0);
                if words == 2 {
                    set_z(z, n + 1, r1);
                }
            }
            Some(w)
        }
        e if e.starts_with("ENC_VOP") || e.starts_with("VOP") => {
            if d.variant != Variant::Plain {
                return None;
            }
            if name.starts_with("v_writelane")
                || name.starts_with("v_readlane")
                || name.starts_with("v_readfirstlane")
                || name.starts_with("v_movrel")
                || name.starts_with("v_swaprel")
                || name.starts_with("v_interp")
                || name.starts_with("v_div_scale")
            {
                return None;
            }
            let mw = if wave64 { 2 } else { 1 };
            if name.starts_with("v_cmpx_") {
                // ballot(false && cond) == 0 -> EXEC stays 0
                return Some(sreg_range(126, mw));
            }
            if name.starts_with("v_cmp_") {
                let (outs, _) = Lw::ops(d);
                let dst = outs.first().map(|o| Lw::dst_of(d, o)).unwrap_or(Dst::S(106));
                if let Dst::S(n) = dst {
                    set_z(z, n, true);
                    if mw == 2 {
                        set_z(z, n + 1, true);
                    }
                    return Some(sreg_range(n, mw));
                }
                return Some(0);
            }
            if matches!(
                name.as_str(),
                "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_add_co_ci_u32" | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32"
            ) {
                let (outs, _) = Lw::ops(d);
                let sdst = outs.get(1).map(|o| Lw::dst_of(d, o)).unwrap_or(Dst::S(106));
                if let Dst::S(n) = sdst {
                    set_z(z, n, true);
                    if mw == 2 {
                        set_z(z, n + 1, true);
                    }
                    return Some(sreg_range(n, mw));
                }
                return Some(0);
            }
            if matches!(name.as_str(), "v_mad_u64_u32" | "v_mad_i64_i32") {
                let (outs, _) = Lw::ops(d);
                if let Some(Dst::S(n)) = outs.get(1).map(|o| Lw::dst_of(d, o)) {
                    set_z(z, n, true);
                    if mw == 2 {
                        set_z(z, n + 1, true);
                    }
                    return Some(sreg_range(n, mw));
                }
                return Some(0);
            }
            // Reject any other VALU op that unexpectedly writes a scalar register.
            let (outs, _) = Lw::ops(d);
            if outs.iter().any(|o| matches!(Lw::dst_of(d, o), Dst::S(_))) {
                return None;
            }
            Some(0)
        }
        "ENC_MUBUF" | "ENC_MTBUF" => {
            if (name.starts_with("buffer_load_") || name.starts_with("tbuffer_load_"))
                && Lw::fv(d, "LDS") == 0
                && Lw::fv(d, "TFE") == 0
            {
                Some(0)
            } else {
                None
            }
        }
        "ENC_FLAT" | "ENC_FLAT_GLBL" => {
            if name.starts_with("flat_load_") || name.starts_with("global_load_") {
                Some(0)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Makes the CFG reducible by splitting: for an irreducible edge's innermost multi-entry
/// cycle region, the region minus its first entry (in RPO) is copied for the edges entering
/// it elsewhere. The reverse post-order and loop headers of the result; gives up past four
/// times the original block count.
fn make_reducible(blocks: &mut Vec<CfgBlock>) -> Result<(Vec<usize>, BTreeSet<usize>), String> {
    let limit = 4 * blocks.len() + 256;
    loop {
        let (post, headers) = order(blocks);
        let Some((_, t)) = irreducible_edge(blocks, &post) else { return Ok((post, headers)) };
        let n = blocks.len();
        let mut pos = vec![usize::MAX; n];
        for (i, &b) in post.iter().enumerate() {
            pos[b] = i;
        }
        let mut preds = vec![Vec::new(); n];
        for &b in &post {
            for s in succ_of(&blocks[b]) {
                preds[s].push(b);
            }
        }
        // nodes reachable from `from` within `inside`, along successors or predecessors
        let reach = |from: usize, inside: &[bool], fwd: bool| {
            let mut seen = vec![false; n];
            let mut stack = vec![from];
            seen[from] = true;
            while let Some(x) = stack.pop() {
                let next = if fwd { succ_of(&blocks[x]) } else { preds[x].clone() };
                for y in next {
                    if inside[y] && !seen[y] {
                        seen[y] = true;
                        stack.push(y);
                    }
                }
            }
            seen
        };
        // shrink to the innermost cycle region around `t` with more than one entry
        let mut inside: Vec<bool> = (0..n).map(|b| pos[b] != usize::MAX).collect();
        let (region, entries) = loop {
            let (f, r) = (reach(t, &inside, true), reach(t, &inside, false));
            let region: Vec<bool> = (0..n).map(|b| f[b] && r[b]).collect();
            let entries: Vec<usize> = (0..n).filter(|&b| region[b] && (b == post[0] || preds[b].iter().any(|&p| !region[p]))).collect();
            if entries.len() != 1 {
                break (region, entries);
            }
            inside = region;
            inside[entries[0]] = false;
            if !inside[t] {
                return Err("irreducible control flow (no multi-entry region)".into());
            }
        };
        if blocks.len() >= limit || entries.is_empty() {
            return Err(format!("irreducible control flow (still after splitting to {} blocks)", blocks.len()));
        }
        let h = *entries.iter().min_by_key(|&&e| pos[e]).unwrap();
        let mut copy = vec![usize::MAX; n];
        for b in (0..n).filter(|&b| region[b] && b != h) {
            copy[b] = blocks.len();
            blocks.push(CfgBlock { pc: blocks[b].pc, ops: blocks[b].ops.clone(), term: blocks[b].term.clone(), preds: Vec::new() });
        }
        let redirect = |term: &mut Term, to: &dyn Fn(usize) -> usize| match term {
            Term::Jump(x) => *x = to(*x),
            Term::Cond(_, x, y) => {
                *x = to(*x);
                *y = to(*y)
            }
            Term::End => {}
        };
        // the copies stay among themselves except for `h` and the region's exits
        for b in (0..n).filter(|&b| copy[b] != usize::MAX) {
            redirect(&mut blocks[copy[b]].term, &|x| if copy[x] != usize::MAX { copy[x] } else { x });
        }
        // edges from outside the region into its other entries go to the copies
        for p in (0..n).filter(|&p| !region[p] && pos[p] != usize::MAX) {
            redirect(&mut blocks[p].term, &|x| if region[x] && x != h { copy[x] } else { x });
        }
    }
}

fn succ_of(b: &CfgBlock) -> Vec<usize> {
    match b.term {
        Term::Jump(t) => vec![t],
        Term::Cond(_, t, f) => vec![t, f],
        Term::End => vec![],
    }
}

/// Reachable blocks in reverse post-order from the entry (iterative DFS) and the targets of
/// retreating edges.
fn order(blocks: &[CfgBlock]) -> (Vec<usize>, BTreeSet<usize>) {
    let n = blocks.len();
    let succ = succ_of;
    let mut state = vec![0u8; n]; // 0 new, 1 on stack, 2 done
    let mut post = Vec::new();
    let mut headers = BTreeSet::new();
    if n > 0 {
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
        state[0] = 1;
        while let Some(&mut (b, ref mut k)) = stack.last_mut() {
            let s = succ(&blocks[b]);
            if *k < s.len() {
                let t = s[*k];
                *k += 1;
                match state[t] {
                    0 => {
                        state[t] = 1;
                        stack.push((t, 0));
                    }
                    1 => {
                        headers.insert(t);
                    }
                    _ => {}
                }
            } else {
                state[b] = 2;
                post.push(b);
                stack.pop();
            }
        }
    }
    post.reverse();
    (post, headers)
}

/// A retreating edge (a, t) of the reverse post-order `rpo` whose target does not dominate
/// its source (the CFG is irreducible), if any.
fn irreducible_edge(blocks: &[CfgBlock], rpo: &[usize]) -> Option<(usize, usize)> {
    let mut pos = vec![usize::MAX; blocks.len()];
    for (i, &b) in rpo.iter().enumerate() {
        pos[b] = i;
    }
    let mut preds = vec![Vec::new(); blocks.len()];
    for &b in rpo {
        for t in succ_of(&blocks[b]) {
            preds[t].push(b);
        }
    }
    // Cooper–Harvey–Kennedy dominators over the RPO
    let mut idom = vec![usize::MAX; blocks.len()];
    let entry = *rpo.first()?;
    idom[entry] = entry;
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &rpo[1..] {
            let mut nd = usize::MAX;
            for &p in preds[b].iter().filter(|&&p| idom[p] != usize::MAX) {
                nd = if nd == usize::MAX {
                    p
                } else {
                    let (mut x, mut y) = (p, nd);
                    while x != y {
                        while pos[x] > pos[y] {
                            x = idom[x];
                        }
                        while pos[y] > pos[x] {
                            y = idom[y];
                        }
                    }
                    x
                };
            }
            if idom[b] != nd {
                idom[b] = nd;
                changed = true;
            }
        }
    }
    let dominates = |a: usize, mut b: usize| loop {
        if a == b {
            return true;
        }
        if b == entry {
            return false;
        }
        b = idom[b];
    };
    rpo.iter().find_map(|&a| succ_of(&blocks[a]).into_iter().find(|&t| pos[t] <= pos[a] && !dominates(t, a)).map(|t| (a, t)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blk(term: Term) -> CfgBlock {
        CfgBlock { pc: 0, ops: Vec::new(), term, preds: Vec::new() }
    }

    /// The console's loop entered in the middle (`62b584e061c05fc5`'s shape): 0 → {1 | 3},
    /// 1 → 2, cycle 2 → 3 → {2 | 4}; split into a single-entry loop.
    #[test]
    fn loop_entered_in_the_middle_is_split() {
        let mut b = vec![blk(Term::Cond(0, 3, 1)), blk(Term::Jump(2)), blk(Term::Jump(3)), blk(Term::Cond(0, 2, 4)), blk(Term::End)];
        let (rpo, headers) = order(&b);
        assert!(irreducible_edge(&b, &rpo).is_some());
        let (rpo, headers2) = make_reducible(&mut b).unwrap();
        assert!(irreducible_edge(&b, &rpo).is_none());
        assert_eq!(b.len(), 6);
        assert_eq!(headers.len(), 1);
        assert_eq!(headers2.len(), 1);
    }

    fn enc(name: &str, encoding: &str, fields: &[(&str, u64)], lit: Option<u32>) -> Vec<u8> {
        rdna2::encode(name, encoding, fields, lit).unwrap()
    }

    /// Nested forward `s_cbranch_execz` over VALU + `s_and_saveexec_b64` + `s_mov_b64 exec`
    /// collapses completely to a single block when modified scratch SGPRs are dead at target.
    #[test]
    fn nested_execz_collapses_to_single_block() {
        let mut code = Vec::new();
        // s_and_saveexec_b64 s[4:5], s[2:3]
        code.extend(enc("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 2)], None));
        // s_cbranch_execz +5 dwords (skips 5 4-byte insts to s_mov_b64 exec, s[4:5])
        code.extend(enc("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 5)], None));
        // inner: s_and_saveexec_b64 s[6:7], s[8:9]
        code.extend(enc("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 8)], None));
        // inner: s_cbranch_execz +2 dwords (skips v_mov_b32 + v_add_f32 to s_mov_b64 exec, s[6:7])
        code.extend(enc("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 2)], None));
        code.extend(enc("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 256 + 2)], None));
        code.extend(enc("V_ADD_F32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 256 + 1), ("VSRC1", 3)], None));
        // inner join: s_mov_b64 exec, s[6:7]
        code.extend(enc("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 6)], None));
        // outer join: s_mov_b64 exec, s[4:5]
        code.extend(enc("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 4)], None));
        code.extend(enc("S_ENDPGM", "ENC_SOPP", &[], None));

        // Compute mode (`live_at_end = false`): s[6:7] and SCC are dead at target -> 0 Save/Restore ops!
        let cfg = Cfg::build_with_threshold(&code, true, false, 0).unwrap();
        assert_eq!(cfg.rpo.len(), 1, "expected single collapsed block, got {} blocks", cfg.rpo.len());
        let ops = &cfg.blocks[cfg.rpo[0]].ops;
        assert!(!ops.iter().any(|o| matches!(o, CfgOp::SaveExecZ { .. })), "dead scratch SGPRs should not emit SaveExecZ");

        // Kernel mode (`live_at_end = true`): s[6:7] and SCC are live at s_endpgm -> Save/Restore emitted!
        let cfg_k = Cfg::build_with_threshold(&code, true, true, 0).unwrap();
        assert_eq!(cfg_k.rpo.len(), 1);
        let ops_k = &cfg_k.blocks[cfg_k.rpo[0]].ops;
        assert!(ops_k.iter().any(|o| matches!(o, CfgOp::SaveExecZ { regs, .. } if regs.contains(&6) && regs.contains(&7) && regs.contains(&VAR_SCC))));
    }

    /// A region containing a side-effecting instruction (`s_barrier`) or an instruction that
    /// could restore non-zero EXEC (`s_mov_b64 exec, s[2:3]` where `s[2:3]` is not proven 0)
    /// MUST NOT be collapsed.
    #[test]
    fn unsafe_execz_regions_are_not_collapsed() {
        // Case 1: s_barrier inside execz region
        let mut c1 = Vec::new();
        c1.extend(enc("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)], None));
        c1.extend(enc("S_BARRIER", "ENC_SOPP", &[], None));
        c1.extend(enc("S_ENDPGM", "ENC_SOPP", &[], None));
        let cfg1 = Cfg::build_with_threshold(&c1, true, false, 0).unwrap();
        assert!(cfg1.rpo.len() > 1, "s_barrier region must not collapse");

        // Case 2: restoring EXEC from an unknown SGPR pair s[2:3] inside the region
        let mut c2 = Vec::new();
        c2.extend(enc("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 2)], None));
        c2.extend(enc("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 2)], None));
        c2.extend(enc("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 256 + 2)], None));
        c2.extend(enc("S_ENDPGM", "ENC_SOPP", &[], None));
        let cfg2 = Cfg::build_with_threshold(&c2, true, false, 0).unwrap();
        assert!(cfg2.rpo.len() > 1, "region restoring unknown EXEC must not collapse");

        // Case 3: v_writelane_b32 ignores EXEC, so must not collapse
        let mut c3 = Vec::new();
        c3.extend(enc("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 2)], None));
        c3.extend(enc("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", 0), ("SRC1", 128)], None));
        c3.extend(enc("S_ENDPGM", "ENC_SOPP", &[], None));
        let cfg3 = Cfg::build_with_threshold(&c3, true, false, 0).unwrap();
        assert!(cfg3.rpo.len() > 1, "v_writelane_b32 region must not collapse");
    }

    /// A region modifying a live-out scalar register (`s0`) MUST save and restore `s0`.
    #[test]
    fn live_out_scalar_reg_is_saved_and_restored() {
        let mut code = Vec::new();
        code.extend(enc("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)], None));
        code.extend(enc("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 129)], None)); // s0 = 1
        code.extend(enc("S_ADD_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 0), ("SSRC1", 129)], None)); // reads s0!
        code.extend(enc("S_ENDPGM", "ENC_SOPP", &[], None));
        let cfg = Cfg::build_with_threshold(&code, true, false, 0).unwrap();
        assert_eq!(cfg.rpo.len(), 1);
        let ops = &cfg.blocks[cfg.rpo[0]].ops;
        assert!(
            ops.iter().any(|o| matches!(o, CfgOp::SaveExecZ { regs, .. } if regs == &[0])),
            "expected SaveExecZ for live-out s0, got {ops:?}"
        );
        assert!(
            ops.iter().any(|o| matches!(o, CfgOp::RestoreIfExecZ { regs, .. } if regs == &[0])),
            "expected RestoreIfExecZ for live-out s0, got {ops:?}"
        );
    }
}
