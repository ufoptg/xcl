//! GPU parity and API tests for the per-lane Direct DXIL backend (`recomp::lower_dxil`):
//! exercises `S_WQM` + `S_CSELECT` + `V_CNDMASK`, scalar carry/borrow chains (`S_ADDC_U32` /
//! `S_SUBB_U32`), far jumps (`S_GETPC_B64` + `S_SETPC_B64`), `compute_code`, `compute_split`,
//! and `survey_compute_code`. Each kernel program is validated against DXC and run on the GPU
//! in wave32 (`lower_dxil::kernel`) and split wave64 (`lower_dxil::kernel_split`), compared
//! register-for-register against the HLSL+DXC path.

use xcl::gpu::dxil::validate;
use xcl::gpu::rdna2::{self, Decoded};
use xcl::gpu::recomp::{dxc, lower_dxil, runner, translate, CsInterface, Mode, Options};

type Inst<'a> = (&'a str, &'a str, &'a [(&'a str, u64)], Option<u32>);

const V: u64 = 256;
const ENDPGM: Inst<'static> = ("S_ENDPGM", "ENC_SOPP", &[], None);

fn assemble(lines: &[Inst]) -> (Vec<u8>, Vec<Decoded>) {
    let mut bytes = Vec::new();
    for &(name, enc, fields, lit) in lines {
        bytes.extend(rdna2::encode(name, enc, fields, lit).unwrap_or_else(|e| panic!("encode {name}: {e}")));
    }
    let mut insts = Vec::new();
    let mut off = 0usize;
    while off + 4 <= bytes.len() {
        let d = rdna2::decode(&bytes[off..], off as u64).unwrap_or_else(|e| panic!("decode @{off}: {e:?}"));
        off += d.len;
        insts.push(d);
    }
    rdna2::resolve_far_jumps(&mut insts);
    (bytes, insts)
}

fn gpu_ready() -> bool {
    match runner::wave32_supported() {
        Ok(true) => {}
        Ok(false) => {
            runner::skip_test("no wave32");
            return false;
        }
        Err(e) => {
            runner::skip_test(&e);
            return false;
        }
    }
    if dxc::location().is_err() {
        runner::skip_test("no DXC");
        return false;
    }
    true
}

fn init_state(lanes: usize) -> runner::KernelState {
    let mut st = runner::KernelState::default();
    st.sgpr[126] = u32::MAX;
    st.sgpr[127] = if lanes == 64 { u32::MAX } else { 0 };
    for l in 0..lanes {
        st.vgpr[l][0] = l as u32;
    }
    st
}

fn check_valid(label: &str, dxil: &[u8]) {
    if let Err(e) = validate::validate(dxil) {
        let dis = validate::disassemble(dxil).unwrap_or_else(|d| d);
        panic!("{label}: validator rejected DXIL:\n{e}\n--- disassembly ---\n{dis}");
    }
}

fn compare_kernel(label: &str, lines: &[Inst], wave64: bool) -> (runner::KernelResult, Vec<u8>) {
    let (bytes, insts) = assemble(lines);
    let (dxil, stale) = if wave64 {
        lower_dxil::kernel_split(&insts).unwrap_or_else(|e| panic!("{label}: kernel_split failed: {e}"))
    } else {
        lower_dxil::kernel(&insts, false).unwrap_or_else(|e| panic!("{label}: kernel failed: {e}"))
    };
    check_valid(label, &dxil);

    let lanes = if wave64 { 64 } else { 32 };
    let st = init_state(lanes);
    let direct = runner::run(&dxil, &st, &[]).unwrap_or_else(|e| panic!("{label}: direct GPU run: {e}"));

    let t = translate(&bytes, &Options { wave64, split_wave64: wave64, ..Default::default() })
        .unwrap_or_else(|e| panic!("{label}: HLSL translate: {e}"));
    assert!(t.complete(), "{label}: incomplete HLSL: {:?}", t.unsupported_why);
    let hlsl_dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &["-Gis"]).unwrap_or_else(|e| panic!("{label}: DXC: {e}"));
    let hlsl = runner::run(&hlsl_dxil, &st, &[]).unwrap_or_else(|e| panic!("{label}: HLSL GPU run: {e}"));

    let mut sregs = vec![106usize, 126];
    if wave64 {
        sregs.extend([107, 127]);
    }
    let mut vregs = Vec::new();
    for d in &insts {
        let (r, w) = rdna2::sgpr_uses(d);
        sregs.extend(r.iter().chain(&w).map(|&x| x as usize).filter(|&x| x < 125));
        let (r, w) = rdna2::vgpr_uses(d);
        vregs.extend(r.iter().chain(&w).map(|&x| x as usize));
    }
    let stale_at = |r: usize| stale.contains(&(r as u8)) || (wave64 && r % 2 == 1 && stale.contains(&(r as u8 - 1)));
    for &r in sregs.iter().filter(|&&r| !stale_at(r)) {
        assert_eq!(direct.sgpr[r], hlsl.sgpr[r], "{label}: s{r}");
    }
    for l in 0..lanes {
        for &r in &vregs {
            assert_eq!(direct.vgpr[l][r], hlsl.vgpr[l][r], "{label}: lane {l} v{r}");
        }
    }
    (direct, stale)
}

#[test]
fn per_lane_wqm_and_cselect_match_hlsl_on_gpu() {
    if !gpu_ready() {
        return;
    }
    // Wave32: lane 5 -> WQM quad 4..8, S_CSELECT_B32 on both masks and scalar numbers.
    let w32: &[Inst] = &[
        ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 133), ("VSRC1", 0)], None),
        ("S_WQM_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)], None),
        ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", 128), ("SRC1", 129), ("SRC2", 0)], None),
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 128), ("SSRC1", 128)], None),
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 0), ("SSRC1", 128)], None),
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 133), ("SSRC1", 135)], None),
        ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 2), ("SRC0", 128), ("SRC1", 129), ("SRC2", 1)], None),
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 128), ("SSRC1", 129)], None),
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 3), ("SSRC0", 133), ("SSRC1", 135)], None),
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 0), ("SSRC1", 128)], None),
        ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 3), ("SRC0", 128), ("SRC1", 129), ("SRC2", 4)], None),
        ENDPGM,
    ];
    let (r32, _) = compare_kernel("wqm+cselect wave32", w32, false);
    assert_eq!((r32.sgpr[2], r32.sgpr[3]), (5, 7));
    for l in 0..32 {
        let quad = (4..8).contains(&l) as u32;
        assert_eq!((r32.vgpr[l][1], r32.vgpr[l][2], r32.vgpr[l][3]), (quad, quad, 0), "wave32 lane {l}");
    }

    // Split wave64: lanes 5 and 37 -> quads 4..8 (lower half) and 36..40 (upper half),
    // plus S_CSELECT_B64 selecting a 64-bit mask across both wave32 halves.
    let w64: &[Inst] = &[
        ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 133), ("VSRC1", 0)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)], None),
        ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 165), ("VSRC1", 0)], None),
        ("S_OR_B64", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 0), ("SSRC1", 106)], None),
        ("S_WQM_B64", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 2)], None),
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 128), ("SSRC1", 128)], None),
        ("S_CSELECT_B64", "ENC_SOP2", &[("SDST", 6), ("SSRC0", 4), ("SSRC1", 128)], None),
        ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", 128), ("SRC1", 129), ("SRC2", 6)], None),
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 128), ("SSRC1", 129)], None),
        ("S_CSELECT_B64", "ENC_SOP2", &[("SDST", 8), ("SSRC0", 4), ("SSRC1", 128)], None),
        ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 2), ("SRC0", 128), ("SRC1", 129), ("SRC2", 8)], None),
        ENDPGM,
    ];
    let (r64, _) = compare_kernel("wqm+cselect split wave64", w64, true);
    for l in 0..64 {
        let quad = ((4..8).contains(&l) || (36..40).contains(&l)) as u32;
        assert_eq!((r64.vgpr[l][1], r64.vgpr[l][2]), (quad, 0), "split wave64 lane {l}");
    }
}

#[test]
fn per_lane_scalar_carry_borrow_and_far_jump_match_hlsl_on_gpu() {
    if !gpu_ready() {
        return;
    }
    let carry: &[Inst] = &[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 255)], Some(0xFFFF_FFFF)),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 0), ("SSRC1", 129)], None),
        ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 128), ("SSRC1", 128)], None),
        ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 5), ("SSRC0", 0), ("SSRC1", 128)], None),
        ("S_SUB_U32", "ENC_SOP2", &[("SDST", 3), ("SSRC0", 128), ("SSRC1", 129)], None),
        ("S_SUBB_U32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 133), ("SSRC1", 129)], None),
        ("S_SUBB_U32", "ENC_SOP2", &[("SDST", 6), ("SSRC0", 133), ("SSRC1", 129)], None),
        ENDPGM,
    ];
    for wave64 in [false, true] {
        let (r, _) = compare_kernel(if wave64 { "carry/borrow split wave64" } else { "carry/borrow wave32" }, carry, wave64);
        assert_eq!((r.sgpr[1], r.sgpr[2], r.sgpr[5]), (0, 1, 0xFFFF_FFFF));
        assert_eq!((r.sgpr[3], r.sgpr[4], r.sgpr[6]), (0xFFFF_FFFF, 3, 4));
    }

    // Far jump over four overwritten v1 assignments: s_getpc_b64 + 32 -> lands on v_add_nc_u32.
    let far: &[Inst] = &[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 135)], None),
        ("S_GETPC_B64", "ENC_SOP1", &[("SDST", 2)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 2), ("SSRC1", 255)], Some(32)),
        ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 3), ("SSRC0", 3), ("SSRC1", 128)], None),
        ("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 2)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 136)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 137)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 138)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 139)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 1), ("VSRC1", 0)], None),
        ENDPGM,
    ];
    for wave64 in [false, true] {
        let lanes = if wave64 { 64 } else { 32 };
        let (r, _) = compare_kernel(if wave64 { "far jump split wave64" } else { "far jump wave32" }, far, wave64);
        assert_eq!((r.sgpr[2], r.sgpr[3]), (40, 0));
        for l in 0..lanes {
            assert_eq!((r.vgpr[l][1], r.vgpr[l][2]), (7, 7 + l as u32), "lane {l}");
        }
    }
}

#[test]
fn per_lane_compute_code_split_and_survey() {
    // Compute program using a far jump + global store: mem[tid] = tid + 10.
    // v0 = thread ID in group; s[0:1] = base VA (0 from runner::dispatch args [0, 0]).
    let lines: &[Inst] = &[
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 130), ("VSRC1", 0)], None),
        ("S_GETPC_B64", "ENC_SOP1", &[("SDST", 4)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 4), ("SSRC1", 255)], Some(24)),
        ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 5), ("SSRC0", 5), ("SSRC1", 128)], None),
        ("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 138), ("VSRC1", 0)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 2), ("SADDR", 125), ("OFFSET", 0)], None),
        ENDPGM,
    ];
    let (bytes, insts) = assemble(lines);
    let cs32 = CsInterface { threads: [32, 1, 1], user_sgprs: 2, tgid: [false; 3], tg_size: false, tid_components: 1 };
    let cs64 = CsInterface { threads: [64, 1, 1], user_sgprs: 2, tgid: [false; 3], tg_size: false, tid_components: 1 };

    // survey_compute_code accepts the valid compute shader and reports refusals on invalid bytes.
    assert_eq!(lower_dxil::survey_compute_code(&bytes, &cs32, false), Vec::<String>::new());
    let bad_survey = lower_dxil::survey_compute_code(&[0xFF; 4], &cs32, false);
    assert!(!bad_survey.is_empty(), "undecodable bytes must be reported by survey_compute_code");

    let dxil32 = lower_dxil::compute_code(&bytes, &cs32, false).expect("compute_code wave32");
    let dxil64 = lower_dxil::compute_split(&insts, &cs64).expect("compute_split wave64");

    if !gpu_ready() {
        return;
    }
    check_valid("compute_code wave32", &dxil32);
    check_valid("compute_split wave64", &dxil64);

    let mem = vec![0u32; 128];
    let out32 = runner::dispatch(&dxil32, [1, 1, 1], &[0, 0], &mem).expect("dispatch wave32");
    let out64 = runner::dispatch(&dxil64, [1, 1, 1], &[0, 0], &mem).expect("dispatch split wave64");

    // Cross-check split wave64 compute output against HLSL+DXC as well.
    let t = translate(&bytes, &Options { mode: Mode::Compute(cs64), wave64: true, split_wave64: true, ..Default::default() })
        .expect("HLSL compute translate");
    assert!(t.complete(), "incomplete HLSL compute: {:?}", t.unsupported_why);
    let hlsl_dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).expect("DXC compute");
    let hlsl64 = runner::dispatch(&hlsl_dxil, [1, 1, 1], &[0, 0], &mem).expect("HLSL dispatch");

    for l in 0..32usize {
        assert_eq!(out32[l], l as u32 + 10, "wave32 lane {l}");
    }
    for l in 0..64usize {
        assert_eq!(out64[l], l as u32 + 10, "split wave64 lane {l}");
        assert_eq!(out64[l], hlsl64[l], "split wave64 vs HLSL lane {l}");
    }
}
