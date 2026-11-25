//! `s_barrier` for compute and mesh thread groups (the recompiler's `xcl_s_barrier`,
//! `recomp/emit.rs` `PRELUDE_SAFE_BARRIER`).
//!
//! On RDNA2, `s_endpgm` removes the exiting wave from the workgroup's barrier count, so the
//! waves still running do not wait for it at `s_barrier`. A D3D12 group sync
//! (`GroupMemoryBarrierWithGroupSync`, `bar.sync 0` on NVIDIA) waits for every hardware
//! wave of the group: one that has ended (`s_endpgm`), run out of its loop budget, or
//! diverged after a split-wave64 desync hangs the others for good.
//!
//! So a thread group of several hardware waves whose program has `s_barrier` gets a
//! bounded arrival barrier in group-shared memory instead (on unless `XCL_SAFE_BARRIER=0`):
//! `alive` counts the live hardware waves (set at entry, decremented once when a wave
//! leaves the program — every exit of the translated program goes through one block);
//! `state` packs (generation: 20 bits, arrived: 12 bits) so advancing the generation
//! resets the arrivals. A single hardware wave needs no synchronization: a memory fence.

use super::lower::{Lw, VAR_BAR_DEAD};
use super::module::{Pred, RmwOp};
use super::ops::{barrier, GroupShared};
use crate::gpu::rdna2::Decoded;

/// How `s_barrier` is lowered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SBarrier {
    /// a plain group sync (the kernel ABI: one console wave, as the HLSL kernel)
    Sync,
    /// one hardware wave in the group: a group-shared memory fence
    Fence,
    /// several hardware waves: the bounded arrival barrier
    Safe,
}

/// Rounds a waiting wave spins before it gives up (the HLSL `1u << 20`).
pub const SPIN_LIMIT: u32 = 1 << 20;

/// The early-exit-safe barrier is on unless `XCL_SAFE_BARRIER=0` (the recompiler's switch).
pub fn safe_barrier_enabled() -> bool {
    std::env::var("XCL_SAFE_BARRIER").map_or(true, |v| v.trim() != "0")
}

/// The program has an `s_barrier`.
pub fn has_s_barrier(insts: &[Decoded]) -> bool {
    insts.iter().any(|d| d.inst.name.eq_ignore_ascii_case("S_BARRIER"))
}

/// The recompiler's policy (`Em::configure_round_modes` / `compute_or_mesh_hw_waves`):
/// `group_threads` is Some for compute and mesh groups, `hw_lanes` the hardware wave size.
pub fn policy(has_barrier: bool, group_threads: Option<u32>, hw_lanes: u32, enabled: bool) -> SBarrier {
    let Some(t) = group_threads else { return SBarrier::Sync };
    let waves = t.div_ceil(hw_lanes).max(1);
    if waves == 1 {
        SBarrier::Fence
    } else if has_barrier && enabled {
        SBarrier::Safe
    } else {
        SBarrier::Sync
    }
}

impl Lw {
    /// Words the safe barrier takes from the top of the LDS block (`alive`, `state` at
    /// 8189..=8190, below the split form's lane counter at 8191).
    pub fn bar_lds_words(&self) -> u32 {
        if self.sbar == SBarrier::Safe {
            2
        } else {
            0
        }
    }

    /// The safe barrier's two words (after [`Lw::setup_storage`]): in the LDS block when
    /// the program has one (D3D12 allows 32 KB, all of it already the LDS: the program's
    /// accesses stop below them), else two words of their own.
    pub fn setup_barrier_storage(&mut self) {
        if self.sbar != SBarrier::Safe {
            return;
        }
        if let Some(g) = self.lds {
            // split form: `setup_storage` put the exchange below them (`bar_lds_words`)
            self.lds_top = self.lds_top.min(8188);
            self.bar = Some((g, 8189));
        } else {
            let i32 = self.s.m.i32();
            let g = self.s.groupshared("xcl_bar", i32, 2);
            self.bar = Some((g, 0));
        }
    }

    fn bar_storage(&self) -> (GroupShared, u32) {
        self.bar.expect("safe barrier storage")
    }

    /// At entry (every thread of the group present, no branch yet — the entry block):
    /// `alive` = the group's hardware waves, `state` = 0 (`xcl_s_barrier_init`).
    pub fn barrier_init(&mut self) {
        if self.sbar != SBarrier::Safe {
            return;
        }
        let (g, at) = self.bar_storage();
        // every thread stores the same 0 (no branch in the entry block)
        let z = self.c(0);
        let ia = self.c(at);
        let pa = self.s.gs_ptr(g, ia);
        self.s.m.store(pa, z, 4);
        let is = self.c(at + 1);
        let ps = self.s.gs_ptr(g, is);
        self.s.m.store(ps, z, 4);
        self.s.barrier(barrier::GROUP_SYNC);
        let first = self.s.wave_is_first_lane();
        let k = self.selc(first, 1, z);
        let pa = self.s.gs_ptr(g, ia);
        self.s.m.atomic_rmw(RmwOp::Add, pa, k);
        self.s.barrier(barrier::GROUP_SYNC);
    }

    /// Where the program ends (`s_endpgm`, the end of the code, an exhausted loop budget):
    /// a live wave leaves the count once (`xcl_s_barrier_exit`).
    pub fn barrier_exit(&mut self) {
        if self.sbar != SBarrier::Safe {
            return;
        }
        let (g, at) = self.bar_storage();
        let dead = self.rd(VAR_BAR_DEAD);
        let alive = self.cmpc(Pred::Eq, dead, 0);
        let first = self.s.wave_is_first_lane();
        let go = self.land(alive, first);
        let _ = self.guarded(go, &mut |lw| {
            let ia = lw.c(at);
            let m1 = lw.c(u32::MAX);
            lw.s.gs_atomic(RmwOp::Add, g, ia, m1);
            Ok(())
        });
    }

    /// `s_barrier`.
    pub fn s_barrier(&mut self) {
        match self.sbar {
            SBarrier::Sync => self.s.barrier(barrier::GROUP_SYNC),
            SBarrier::Fence => self.s.barrier(barrier::TGSM_FENCE),
            SBarrier::Safe => self.safe_barrier(),
        }
    }

    /// The bounded arrival barrier (`xcl_s_barrier`): the wave's first lane counts the
    /// wave in, then waits (bounded) until the arrivals reach the live waves or another
    /// wave opened the next generation. A wave that gives up leaves the count and from
    /// then on only fences.
    fn safe_barrier(&mut self) {
        let (g, at) = self.bar_storage();
        let ia = at;
        let is = at + 1;
        self.s.barrier(barrier::TGSM_FENCE);
        let dead = self.rd(VAR_BAR_DEAD);
        let alive = self.cmpc(Pred::Eq, dead, 0);
        let first = self.s.wave_is_first_lane();
        let go = self.land(alive, first);
        let ok = self.tmp_var();
        let one = self.c(1);
        self.wr(ok, one);
        let _ = self.guarded(go, &mut |lw| {
            let i32 = lw.s.m.i32();
            let si = lw.c(is);
            let ai = lw.c(ia);
            let one = lw.c(1);
            let z = lw.c(0);
            let old = lw.s.gs_atomic(RmwOp::Add, g, si, one);
            let gen = lw.lshrc(old, 12);
            let g1 = lw.addc(gen, 1);
            let g1 = lw.andc(g1, 0xF_FFFF);
            let next = lw.shlc(g1, 12);
            let pre = lw.cur();
            // lp: the generation still open? chk: all live waves in (or none)? cas: open
            // the next one; tmo: count the round, give up after SPIN_LIMIT
            let lp = lw.block(Some(2));
            let chk = lw.block(Some(1));
            let cas = lw.block(Some(1));
            let tmo = lw.block(Some(2));
            let fail = lw.block(Some(1));
            let ex = lw.block(Some(3));
            lw.br(lp);
            lw.s.m.set_block(lp);
            let n = lw.s.m.phi(i32);
            lw.s.m.add_incoming(n, z, pre);
            let st = lw.s.gs_atomic(RmwOp::Add, g, si, z);
            let sg = lw.lshrc(st, 12);
            let same = lw.cmp(Pred::Eq, sg, gen);
            lw.cond_br(same, chk, ex);
            lw.s.m.set_block(chk);
            let live = lw.s.gs_atomic(RmwOp::Add, g, ai, z);
            let none = lw.cmp(Pred::SLe, live, z);
            let arrived = lw.andc(st, 0xFFF);
            let all = lw.cmp(Pred::UGe, arrived, live);
            let open = lw.lor(none, all);
            lw.cond_br(open, cas, tmo);
            lw.s.m.set_block(cas);
            let orig = lw.s.gs_cmpxchg(g, si, st, next);
            let won = lw.cmp(Pred::Eq, orig, st);
            lw.cond_br(won, ex, tmo);
            lw.s.m.set_block(tmo);
            let out = lw.cmpc(Pred::UGe, n, SPIN_LIMIT);
            let n1 = lw.addc(n, 1);
            lw.s.m.add_incoming(n, n1, tmo);
            lw.cond_br(out, fail, lp);
            lw.s.m.set_block(fail);
            let m1 = lw.c(u32::MAX);
            lw.s.gs_atomic(RmwOp::Add, g, ai, m1);
            lw.s.gs_cmpxchg(g, si, st, next);
            lw.br(ex);
            lw.s.m.set_block(ex);
            let okp = lw.s.m.phi(i32);
            let (one, z) = (lw.c(1), lw.c(0));
            lw.s.m.add_incoming(okp, one, lp);
            lw.s.m.add_incoming(okp, one, cas);
            lw.s.m.add_incoming(okp, z, fail);
            lw.wr(ok, okp);
            Ok(())
        });
        let okv = self.rd(ok);
        let okf = self.s.wave_read_lane_first(okv);
        let gave_up = self.cmpc(Pred::Eq, okf, 0);
        let dead = self.rd(VAR_BAR_DEAD);
        let dead = self.selc(gave_up, 1, dead);
        self.wr(VAR_BAR_DEAD, dead);
        self.s.barrier(barrier::TGSM_FENCE);
    }
}
