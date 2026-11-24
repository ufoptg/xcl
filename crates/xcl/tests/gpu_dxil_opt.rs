//! Tests for the Direct DXIL module optimizer (`gpu::dxil::module::opt`):
//! constant folding, algebraic simplification, dominator-tree vs. same-block CSE,
//! unreachable-block and trivial-phi cleanup, 64-bit-preserving DCE, and the
//! malformed-IR guard. Every emitted shader is checked against DXC's validator
//! and (where a wave32 GPU is present) executed and compared with CPU values.

use xcl::gpu::dxil::container::Stage;
use xcl::gpu::dxil::module::{BinOp, CastOp, Module, Pred, Value};
use xcl::gpu::dxil::shader::{RawUav, Shader};
use xcl::gpu::dxil::validate;
use xcl::gpu::recomp::{dxc, runner};

struct Kernel {
    s: Shader,
    lane: Value,
    res: RawUav,
    mem: RawUav,
}

impl Kernel {
    fn new() -> Self {
        let mut s = Shader::new(Stage::Compute, (6, 6));
        let res = s.raw_uav(1, 0);
        let mem = s.raw_uav(2, 0);
        let lane = s.flattened_thread_id_in_group();
        Self { s, lane, res, mem }
    }

    fn slot(&mut self, k: u32) -> Value {
        let base = self.s.m.i32c(64 * k);
        let idx = self.s.m.bin(BinOp::Add, base, self.lane);
        let two = self.s.m.i32c(2);
        self.s.m.bin(BinOp::Shl, idx, two)
    }

    fn input(&mut self, k: u32) -> Value {
        let off = self.slot(k);
        self.s.raw_load(self.mem, off, 1)[0]
    }

    fn out(&mut self, k: u32, v: Value) {
        let i32 = self.s.m.i32();
        let v = if self.s.m.type_of(v) == self.s.m.i1() {
            self.s.m.cast(CastOp::ZExt, v, i32)
        } else {
            v
        };
        let off = self.slot(k);
        self.s.raw_store(self.res, off, &[v]);
    }

    fn finish_and_disasm(self) -> Option<(Vec<u8>, String)> {
        let dxil = self.s.finish([32, 1, 1], Some(32));
        if dxc::location().is_err() {
            runner::skip_test("no DXC");
            return None;
        }
        if let Err(e) = validate::validate(&dxil) {
            let dis = validate::disassemble(&dxil).unwrap_or_else(|d| d);
            panic!("validator rejected optimized shader:\n{e}\n--- disassembly ---\n{dis}");
        }
        let dis = validate::disassemble(&dxil).expect("disassemble DXIL");
        Some((dxil, dis))
    }
}

fn run_wave32(dxil: &[u8], mem: &[u32]) -> Option<runner::KernelResult> {
    match runner::wave32_supported() {
        Ok(true) => {}
        Ok(false) => {
            runner::skip_test("no wave32");
            return None;
        }
        Err(e) => {
            runner::skip_test(&e);
            return None;
        }
    }
    Some(runner::run_with(dxil, &runner::KernelState::default(), mem, &[], &[]).expect("GPU run"))
}

fn lane_inputs() -> Vec<u32> {
    (0..32u32).map(|l| l.wrapping_mul(0x0101_0003) ^ 0xA5A5_0011).collect()
}

fn mem_of(inputs: &[u32]) -> Vec<u32> {
    let mut mem = vec![0u32; 64];
    mem[..inputs.len()].copy_from_slice(inputs);
    mem
}

fn out_slot(r: &runner::KernelResult, k: usize) -> &[u32] {
    &r.raw[64 * k..64 * k + 32]
}

#[test]
fn opt_constant_folding_and_identities_validate_and_run() {
    let mut k = Kernel::new();
    let a = k.input(0);
    let i32 = k.s.m.i32();
    let i8 = k.s.m.i8();
    let f32 = k.s.m.f32();

    // Constant-folded 32-bit arithmetic, signed/unsigned shifts, div/rem, casts, compares and select.
    let c_neg = k.s.m.i32c(0xFFFF_FF80); // -128
    let c_3 = k.s.m.i32c(3);
    let c_100 = k.s.m.i32c(100);
    let c_7 = k.s.m.i32c(7);
    let c_1 = k.s.m.i32c(1);
    let c_fe8 = k.s.m.const_int(i8, 0xFE);
    let c_ashr = k.s.m.bin(BinOp::AShr, c_neg, c_3); // -16 = 0xFFFF_FFF0
    let c_div = k.s.m.bin(BinOp::UDiv, c_100, c_7); // 14
    let c_rem = k.s.m.bin(BinOp::URem, c_100, c_7); // 2
    let c_sext = k.s.m.cast(CastOp::SExt, c_fe8, i32); // -2
    let c_cmp = k.s.m.cmp(Pred::SLt, c_neg, c_1); // true
    let c_sum = k.s.m.bin(BinOp::Add, c_ashr, c_div); // -2
    let c_mix = k.s.m.bin(BinOp::Xor, c_sum, c_rem); // -4 = 0xFFFF_FFFC
    let c_folded = k.s.m.select(c_cmp, c_mix, c_sext); // -4 = 0xFFFF_FFFC

    // Algebraic identities on runtime input `a`.
    let zero = k.s.m.i32c(0);
    let one = k.s.m.i32c(1);
    let all_ones = k.s.m.i32c(u32::MAX);
    let c_99 = k.s.m.i32c(99);
    let id1 = k.s.m.bin(BinOp::Add, a, zero);
    let id2 = k.s.m.bin(BinOp::And, id1, all_ones);
    let id3 = k.s.m.bin(BinOp::Mul, id2, one);
    let id3_f = k.s.m.bitcast(id3, f32);
    let id4 = k.s.m.bitcast(id3_f, i32);
    let cond_a = k.s.m.cmp(Pred::UGt, a, c_7);
    let id5 = k.s.m.select(cond_a, id4, a); // select(c, a, a) -> a

    let self_zero = k.s.m.bin(BinOp::Xor, a, a);
    let mul_zero = k.s.m.bin(BinOp::Mul, a, zero);
    let self_eq = k.s.m.cmp(Pred::Eq, a, a); // true
    let self_lt = k.s.m.cmp(Pred::SLt, a, a); // false
    let self_sel = k.s.m.select(self_eq, self_zero, c_99);
    let zero_out = k.s.m.bin(BinOp::Or, self_sel, mul_zero);

    k.out(0, c_folded);
    k.out(1, id5);
    k.out(2, zero_out);
    k.out(3, self_lt);

    let Some((dxil, dis)) = k.finish_and_disasm() else { return };
    for forbidden in [" ashr ", " udiv ", " urem ", " sext ", " select ", " bitcast ", " xor ", " mul "] {
        assert!(!dis.contains(forbidden), "expected `{forbidden}` to fold away:\n{dis}");
    }
    assert!(dis.contains("-4"), "expected folded constant -4 in disassembly:\n{dis}");

    let inputs = lane_inputs();
    let Some(res) = run_wave32(&dxil, &mem_of(&inputs)) else { return };
    assert_eq!(out_slot(&res, 0), &[0xFFFF_FFFCu32; 32]);
    assert_eq!(out_slot(&res, 1), &inputs[..]);
    assert_eq!(out_slot(&res, 2), &[0u32; 32]);
    assert_eq!(out_slot(&res, 3), &[0u32; 32]);
}

#[test]
fn opt_cse_merges_dominators_isolates_siblings_and_keeps_wave_is_first_lane() {
    let mut k = Kernel::new();
    let a = k.input(0);
    let i32 = k.s.m.i32();
    let c0 = k.s.m.i32c(0);
    let c1 = k.s.m.i32c(1);
    let c13 = k.s.m.i32c(13);
    let c3 = k.s.m.i32c(3);

    // Dominating expression in entry, repeated in both child branches.
    let dom_add = k.s.m.bin(BinOp::Add, a, c13);

    // Two identical wave_ballot calls in the same block must merge;
    // two wave_is_first_lane calls must NOT merge (excluded by opt.rs).
    let lsb = k.s.m.bin(BinOp::And, a, c1);
    let cond = k.s.m.cmp(Pred::Ne, lsb, c0);
    let i1_true = k.s.m.i1c(true);
    let b0 = k.s.wave_ballot(i1_true)[0];
    let b1 = k.s.wave_ballot(i1_true)[0];
    let ballot_xor = k.s.m.bin(BinOp::Xor, b0, b1); // folds to 0 once b0 and b1 CSE!
    let f0 = k.s.wave_is_first_lane();
    let f1 = k.s.wave_is_first_lane();
    let both_first = k.s.m.bin(BinOp::And, f0, f1);

    let then_b = k.s.m.new_block();
    let else_b = k.s.m.new_block();
    let join_b = k.s.m.new_block();
    k.s.m.cond_br(cond, then_b, else_b);

    // Sibling branches compute identical `a * 3`: undo stack must isolate them,
    // while `a + 13` must reuse `dom_add` from the dominating entry block.
    // A wave_ballot(true) inside `then_b` sees only the active lanes (`odd_mask`,
    // not `0xFFFF_FFFF`), so Cse::Block must NOT merge it with entry's ballot.
    k.s.m.set_block(then_b);
    let t_dup = k.s.m.bin(BinOp::Add, a, c13);
    let t_sib = k.s.m.bin(BinOp::Mul, a, c3);
    let t_ballot = k.s.wave_ballot(i1_true)[0];
    let t_sum = k.s.m.bin(BinOp::Add, t_dup, t_sib);
    let t_val = k.s.m.bin(BinOp::Xor, t_sum, t_ballot);
    k.s.m.br(join_b);

    k.s.m.set_block(else_b);
    let e_dup = k.s.m.bin(BinOp::Add, a, c13);
    let e_sib = k.s.m.bin(BinOp::Mul, a, c3);
    let e_val = k.s.m.bin(BinOp::Sub, e_sib, e_dup);
    k.s.m.br(join_b);

    k.s.m.set_block(join_b);
    let phi = k.s.m.phi(i32);
    k.s.m.add_incoming(phi, t_val, then_b);
    k.s.m.add_incoming(phi, e_val, else_b);
    let out0 = k.s.m.bin(BinOp::Add, phi, dom_add);
    k.out(0, out0);
    k.out(1, ballot_xor);
    k.out(2, both_first);

    let Some((dxil, dis)) = k.finish_and_disasm() else { return };
    assert_eq!(dis.matches(", 13\n").count(), 1, "dominator CSE should keep one `add ..., 13`:\n{dis}");
    assert_eq!(dis.matches("mul i32 ").count(), 2, "sibling branches must keep separate `mul`s:\n{dis}");
    assert_eq!(
        dis.matches("call %dx.types.fouri32 @dx.op.waveActiveBallot(").count(),
        2,
        "two entry ballots should CSE to 1 while the branch ballot stays separate:\n{dis}"
    );
    assert_eq!(dis.matches("call i1 @dx.op.waveIsFirstLane(").count(), 2, "waveIsFirstLane must not CSE:\n{dis}");

    let inputs = lane_inputs();
    let Some(res) = run_wave32(&dxil, &mem_of(&inputs)) else { return };
    let odd_mask: u32 = inputs.iter().enumerate().fold(0u32, |m, (l, &x)| if x & 1 != 0 { m | (1u32 << l) } else { m });
    let expected0: Vec<u32> = inputs
        .iter()
        .map(|&x| {
            let d = x.wrapping_add(13);
            let m = x.wrapping_mul(3);
            let branch = if x & 1 != 0 { d.wrapping_add(m) ^ odd_mask } else { m.wrapping_sub(d) };
            branch.wrapping_add(d)
        })
        .collect();
    assert_eq!(out_slot(&res, 0), &expected0[..]);
    assert_eq!(out_slot(&res, 1), &[0u32; 32]);
    let mut expected_first = [0u32; 32];
    expected_first[0] = 1;
    assert_eq!(out_slot(&res, 2), &expected_first);
}

#[test]
fn opt_prunes_unreachable_blocks_and_collapses_trivial_phis() {
    let mut k = Kernel::new();
    let a = k.input(0);
    let i32 = k.s.m.i32();
    let c0 = k.s.m.i32c(0);
    let c1 = k.s.m.i32c(1);
    let c7 = k.s.m.i32c(7);
    let c_dead = k.s.m.i32c(0xDEAD);
    let lsb = k.s.m.bin(BinOp::And, a, c1);
    let cond = k.s.m.cmp(Pred::Ne, lsb, c0);

    let dom_val = k.s.m.bin(BinOp::Add, a, c7);
    let then_b = k.s.m.new_block();
    let else_b = k.s.m.new_block();
    let dead_b = k.s.m.new_block();
    let join_b = k.s.m.new_block();
    k.s.m.cond_br(cond, then_b, else_b);

    // Both reachable arms compute an expression CSE collapses to `dom_val`;
    // `dead_b` is unreachable and feeds a distinct sentinel that must be pruned.
    k.s.m.set_block(then_b);
    let t_v = k.s.m.bin(BinOp::Add, a, c7);
    k.s.m.br(join_b);

    k.s.m.set_block(else_b);
    let e_v = k.s.m.bin(BinOp::Add, a, c7);
    k.s.m.br(join_b);

    k.s.m.set_block(dead_b);
    let dead_v = k.s.m.bin(BinOp::Add, a, c_dead);
    k.s.m.br(join_b);

    k.s.m.set_block(join_b);
    let phi = k.s.m.phi(i32);
    k.s.m.add_incoming(phi, t_v, then_b);
    k.s.m.add_incoming(phi, e_v, else_b);
    k.s.m.add_incoming(phi, dead_v, dead_b);
    let out = k.s.m.bin(BinOp::Or, phi, dom_val); // collapses to dom_val
    k.out(0, out);

    let Some((dxil, dis)) = k.finish_and_disasm() else { return };
    assert!(!dis.contains(" phi "), "trivial phi should collapse after unreachable pruning + CSE:\n{dis}");
    assert!(!dis.contains("57005"), "unreachable block constant 0xDEAD should be pruned:\n{dis}");

    let inputs = lane_inputs();
    let Some(res) = run_wave32(&dxil, &mem_of(&inputs)) else { return };
    let expected: Vec<u32> = inputs.iter().map(|&x| x.wrapping_add(7)).collect();
    assert_eq!(out_slot(&res, 0), &expected[..]);
}

#[test]
fn opt_dce_drops_dead_32bit_ops_but_preserves_64bit_flags() {
    let mut k = Kernel::new();
    let a = k.input(0);
    let i64 = k.s.m.i64();
    let f64 = k.s.m.f64();

    // Dead 32-bit chain with distinctive constant: DCE must remove it.
    let c_mul = k.s.m.i32c(0x1357_2468);
    let c_xor = k.s.m.i32c(0x2468_1357);
    let dead32 = k.s.m.bin(BinOp::Mul, a, c_mul);
    let _dead32_2 = k.s.m.bin(BinOp::Xor, dead32, c_xor);

    // Dead 64-bit i64 and f64 ops: DCE must keep them so Shader's inferred
    // Int64Ops / Doubles feature flags match the module under DXIL validation.
    let a64 = k.s.m.cast(CastOp::ZExt, a, i64);
    let c5_64 = k.s.m.const_int(i64, 5);
    let _dead_i64 = k.s.m.bin(BinOp::Add, a64, c5_64);
    let hi_one = k.s.m.i32c(0x3FF0_0000);
    let d = k.s.make_double(a, hi_one);
    let f64_one = k.s.m.const_float(f64, 1.0f64.to_bits());
    let _dead_f64 = k.s.m.fbin(BinOp::Add, d, f64_one, 0);

    k.out(0, a);

    let Some((_dxil, dis)) = k.finish_and_disasm() else { return };
    assert!(!dis.contains("324478056"), "dead 32-bit mul constant should be removed by DCE:\n{dis}");
    assert!(!dis.contains("610800471"), "dead 32-bit xor constant should be removed by DCE:\n{dis}");
    assert!(dis.contains("add i64 "), "dead i64 op must stay for Int64Ops shader flag:\n{dis}");
    assert!(dis.contains("fadd double "), "dead f64 op must stay for Doubles shader flag:\n{dis}");
}

#[test]
fn opt_leaves_malformed_use_before_def_for_writer_to_reject() {
    let mut m = Module::new();
    m.define_entry("main");
    let b0 = m.entry_block();
    let b1 = m.new_block();
    let c1 = m.i32c(1);
    let c2 = m.i32c(2);
    let c3 = m.i32c(3);

    // Define `v` in b1, then insert a use of `v` in b0 before its definition.
    m.set_block(b1);
    let v = m.bin(BinOp::Add, c1, c2);
    m.ret();

    m.set_block(b0);
    let _bad = m.bin(BinOp::Add, v, c3);
    m.br(b1);

    m.optimize();
    let err = m.try_to_bitcode("dxil-ms-dx", "").expect_err("malformed IR must still fail bitcode emission");
    assert!(err.contains("before") || err.contains("dominat") || !err.is_empty(), "unexpected error: {err}");
}
