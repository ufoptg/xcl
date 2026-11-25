//! Direct DXIL vs HLSL+DXC GPU parity tests for packed math (`VOP3P` integer/dot
//! ops), relative VGPR moves (`movrel` / `swaprel`), and 64-bit integer / `f64` VALU
//! (`gpu::dxil::lower_wide`) across `split-w64`, `wave32`, and `native-w64`.

use xcl::gpu::dxil::kernel::{translate_full, KernelOptions};
use xcl::gpu::dxil::validate;
use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::{
    self, dxc,
    runner::{self, KernelState},
    Mode, Options,
};

const V: u64 = 256;

fn gpu_ready() -> bool {
    match (runner::adapter(), runner::wave64_programs_supported(), dxc::location()) {
        (Ok(_), Ok(true), Ok(_)) => true,
        (a, w, d) => {
            runner::skip_test(if a.is_err() || w.is_err() {
                "no GPU"
            } else if d.is_err() {
                "no DXC"
            } else {
                "no wave64"
            });
            false
        }
    }
}

fn configs() -> Vec<(&'static str, bool, bool)> {
    let mut out = Vec::new();
    if runner::wave64_supported() == Ok(true) {
        out.push(("native-w64", true, false));
    }
    if runner::wave32_supported() == Ok(true) {
        out.push(("split-w64", true, true));
        out.push(("wave32", false, false));
    }
    out
}

fn prog(list: &[(&str, &str, &[(&str, u64)], Option<u32>)]) -> Vec<u8> {
    let mut code = Vec::new();
    for (name, enc, fields, lit) in list {
        code.extend(encode(name, enc, fields, *lit).unwrap_or_else(|e| panic!("{name}: {e}")));
    }
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    code
}

fn assert_parity(
    label: &str,
    code: &[u8],
    st: &KernelState,
    check_sgpr: &[usize],
    check_vgpr: &[usize],
) {
    let cfgs = configs();
    assert!(!cfgs.is_empty(), "{label}: expected at least one wave config when gpu_ready() is true");
    for (cfg, wave64, split) in cfgs {
        let lanes = if wave64 { 64 } else { 32 };
        let ko = KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
        let direct = translate_full(code, ko).unwrap_or_else(|e| panic!("{label}/{cfg}: direct: {e}"));
        validate::validate(&direct.dxil).unwrap_or_else(|e| panic!("{label}/{cfg}: validate: {e}"));
        let opt = Options { mode: Mode::Kernel, wave64, split_wave64: split, ..Default::default() };
        let t = recomp::translate(code, &opt).unwrap_or_else(|e| panic!("{label}/{cfg}: hlsl: {e}"));
        assert!(t.complete(), "{label}/{cfg}: incomplete HLSL: {:?}", t.unsupported_why);
        let hlsl = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{label}/{cfg}: dxc: {e}\n{}", t.hlsl));
        let got = runner::run_with(&direct.dxil, st, &[], &[], &[]).unwrap_or_else(|e| panic!("{label}/{cfg}: run direct: {e}"));
        let want = runner::run_with(&hlsl, st, &[], &[], &[]).unwrap_or_else(|e| panic!("{label}/{cfg}: run hlsl: {e}"));
        for &s in check_sgpr {
            assert_eq!(got.sgpr[s], want.sgpr[s], "{label}/{cfg}: s{s}");
        }
        for l in 0..lanes {
            for &v in check_vgpr {
                assert_eq!(got.vgpr[l][v], want.vgpr[l][v], "{label}/{cfg}: lane {l} v{v}");
            }
        }
    }
}

/// Direct DXIL vs HLSL parity for VOP3P packed 16-bit integer ops (`add`/`sub`/`mul_lo`/
/// `mad`/`lshlrev`/`lshrrev`/`ashrrev`/`min`/`max`, wrapping & `CLAMP`, `OP_SEL`/`OP_SEL_HI`)
/// and dot products (`v_dot4_i32_i8`/`u8`, `v_dot4c_i32_i8`, `v_dot8_i32_i4`/`u4`,
/// `v_dot2_i32_i16`/`u16`, and exact-input `v_dot2_f32_f16`).
#[test]
fn direct_dxil_vop3p_packed_int_and_dot_parity() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        // v10..v21: packed u16 / i16 with and without CLAMP, shifts, min/max, OP_SEL swaps
        ("V_PK_ADD_U16", "ENC_VOP3P", &[("VDST", 10), ("SRC0", V + 0), ("SRC1", V + 1), ("OP_SEL_HI", 3), ("CLAMP", 1)], None),
        ("V_PK_ADD_U16", "ENC_VOP3P", &[("VDST", 11), ("SRC0", V + 0), ("SRC1", V + 1), ("OP_SEL_HI", 3)], None),
        ("V_PK_SUB_U16", "ENC_VOP3P", &[("VDST", 12), ("SRC0", V + 1), ("SRC1", V + 0), ("OP_SEL_HI", 3), ("CLAMP", 1)], None),
        ("V_PK_ADD_I16", "ENC_VOP3P", &[("VDST", 13), ("SRC0", V + 2), ("SRC1", V + 3), ("OP_SEL_HI", 3), ("CLAMP", 1)], None),
        ("V_PK_SUB_I16", "ENC_VOP3P", &[("VDST", 14), ("SRC0", V + 2), ("SRC1", V + 3), ("OP_SEL", 2), ("OP_SEL_HI", 1), ("CLAMP", 1)], None),
        ("V_PK_MAD_U16", "ENC_VOP3P", &[("VDST", 15), ("SRC0", V + 0), ("SRC1", V + 1), ("SRC2", V + 1), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1), ("CLAMP", 1)], None),
        ("V_PK_MAD_I16", "ENC_VOP3P", &[("VDST", 16), ("SRC0", V + 2), ("SRC1", V + 1), ("SRC2", V + 3), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1), ("CLAMP", 1)], None),
        ("V_PK_MUL_LO_U16", "ENC_VOP3P", &[("VDST", 17), ("SRC0", V + 0), ("SRC1", V + 1), ("OP_SEL_HI", 3)], None),
        ("V_PK_LSHLREV_B16", "ENC_VOP3P", &[("VDST", 18), ("SRC0", V + 4), ("SRC1", V + 0), ("OP_SEL_HI", 3)], None),
        ("V_PK_LSHRREV_B16", "ENC_VOP3P", &[("VDST", 19), ("SRC0", V + 4), ("SRC1", V + 2), ("OP_SEL_HI", 3)], None),
        ("V_PK_ASHRREV_I16", "ENC_VOP3P", &[("VDST", 20), ("SRC0", V + 4), ("SRC1", V + 2), ("OP_SEL_HI", 3)], None),
        ("V_PK_MIN_I16", "ENC_VOP3P", &[("VDST", 21), ("SRC0", V + 2), ("SRC1", V + 3), ("OP_SEL_HI", 3)], None),
        ("V_PK_MAX_U16", "ENC_VOP3P", &[("VDST", 22), ("SRC0", V + 0), ("SRC1", V + 1), ("OP_SEL_HI", 3)], None),
        // Dot products: i8, u8, i4, u4, i16, u16, and exact f16
        ("V_DOT4_I32_I8", "ENC_VOP3P", &[("VDST", 30), ("SRC0", V + 5), ("SRC1", V + 6), ("SRC2", V + 7), ("CLAMP", 1)], None),
        ("V_DOT4_I32_I8", "ENC_VOP3P", &[("VDST", 31), ("SRC0", V + 5), ("SRC1", V + 6), ("SRC2", V + 7)], None),
        ("V_DOT4_U32_U8", "ENC_VOP3P", &[("VDST", 32), ("SRC0", V + 5), ("SRC1", V + 6), ("SRC2", V + 8), ("CLAMP", 1)], None),
        ("V_DOT8_I32_I4", "ENC_VOP3P", &[("VDST", 33), ("SRC0", V + 5), ("SRC1", V + 6), ("SRC2", V + 7), ("CLAMP", 1)], None),
        ("V_DOT8_U32_U4", "ENC_VOP3P", &[("VDST", 34), ("SRC0", V + 5), ("SRC1", V + 6), ("SRC2", V + 8), ("CLAMP", 1)], None),
        ("V_DOT2_I32_I16", "ENC_VOP3P", &[("VDST", 35), ("SRC0", V + 2), ("SRC1", V + 3), ("SRC2", V + 7), ("OP_SEL_HI", 3), ("CLAMP", 1)], None),
        ("V_DOT2_U32_U16", "ENC_VOP3P", &[("VDST", 36), ("SRC0", V + 0), ("SRC1", V + 1), ("SRC2", V + 8), ("OP_SEL_HI", 3), ("CLAMP", 1)], None),
        ("V_DOT2_F32_F16", "ENC_VOP3P", &[("VDST", 37), ("SRC0", V + 9), ("SRC1", V + 9), ("SRC2", V + 41), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1)], None),
        ("V_DOT4C_I32_I8", "ENC_VOP2", &[("VDST", 38), ("SRC0", V + 5), ("VSRC1", 6)], None),
    ]);
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let lu = l as u32;
        r[0] = 0x0005_FFF0u32.wrapping_add(lu);
        r[1] = 0x0002_0020u32.wrapping_add(lu << 16);
        r[2] = 0x8010_7FF0u32 ^ (lu * 0x0003_0001);
        r[3] = 0xFFE0_0020u32.wrapping_add(lu);
        r[4] = ((lu & 7) << 16) | ((lu + 1) & 7);
        r[5] = 0x7F81_3A04u32.wrapping_add(lu * 0x0101_0101);
        r[6] = 0xFF06_F708u32 ^ (lu * 0x0010_0101);
        r[7] = if l % 4 == 0 { 0x7FFF_FFF0 } else if l % 4 == 1 { 0x8000_0010 } else { 100 + lu };
        r[8] = if l % 3 == 0 { 0xFFFF_FFF0 } else { 500 + lu };
        r[9] = 0xC000_3E00; // lo = 1.5, hi = -2.0 (exact in f16 and f32)
        r[38] = lu * 3;
        r[41] = (10.0f32 + lu as f32).to_bits();
    }
    assert_parity(
        "vop3p_packed_int_and_dot",
        &code,
        &st,
        &[],
        &[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 30, 31, 32, 33, 34, 35, 36, 37, 38],
    );
}

/// Direct DXIL vs HLSL parity for relative VGPR moves (`v_movrels_b32`, `v_movreld_b32`,
/// `v_movrelsd_b32`, `v_movrelsd_2_b32`, and `v_swaprel_b32`) including divergent `EXEC`.
#[test]
fn direct_dxil_movrel_and_swaprel_parity() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        // m0 = 3 -> v40 = v(20+3), v(30+3) = v1
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 131)], None),
        ("V_MOVRELS_B32", "ENC_VOP1", &[("VDST", 40), ("SRC0", V + 20)], None),
        ("V_MOVRELD_B32", "ENC_VOP1", &[("VDST", 30), ("SRC0", V + 1)], None),
        // m0 = 2 -> v(30+2) = v(20+2)
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 130)], None),
        ("V_MOVRELSD_B32", "ENC_VOP1", &[("VDST", 30), ("SRC0", V + 20)], None),
        // m0 = (4 << 16) | 1 -> write index 4 (v34), read index 1 (v21)
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 255)], Some((4 << 16) | 1)),
        ("V_MOVRELSD_2_B32", "ENC_VOP1", &[("VDST", 30), ("SRC0", V + 20)], None),
        // Divergent EXEC (0xAAAA_5555): swaprel only swaps active lanes (v41 <-> v(20+5) with m0=5)
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 133)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 255)], Some(0xAAAA_5555)),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 127), ("SSRC0", 255)], Some(0xAAAA_5555)),
        ("V_SWAPREL_B32", "ENC_VOP1", &[("VDST", 41), ("SRC0", V + 20)], None),
        ("V_MOVRELS_B32", "ENC_VOP1", &[("VDST", 42), ("SRC0", V + 20)], None),
        // Restore EXEC to inspect every register safely
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 255)], Some(0xFFFF_FFFF)),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 127), ("SSRC0", 255)], Some(0xFFFF_FFFF)),
    ]);
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let lu = l as u32;
        r[1] = 0xBEEF_0000 | lu;
        for k in 0..8 {
            r[20 + k] = 0x1000_0000 | ((k as u32) << 8) | lu;
            r[30 + k] = 0x2000_0000 | ((k as u32) << 8) | lu;
        }
        r[25] = 0x3333_0000 | lu;
        r[40] = 0xDEAD_0000 | lu;
        r[41] = 0x4444_0000 | lu;
        r[42] = 0x5555_0000 | lu;
    }
    assert_parity(
        "movrel_and_swaprel",
        &code,
        &st,
        &[],
        &[25, 32, 33, 34, 40, 41, 42],
    );
}

/// Direct DXIL vs HLSL parity for 64-bit integer shifts, `v_mad_u64_u32` / `v_mad_i64_i32`
/// (with carry masks), 64-bit integer comparisons, and `f64` arithmetic / rounding /
/// conversion / comparison ops.
#[test]
fn direct_dxil_valu64_int_and_f64_parity() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        // 64-bit integer shifts & MADs (carry out into s20..21 and s22..23)
        ("V_LSHLREV_B64", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 0), ("SRC1", V + 2)], None),
        ("V_LSHRREV_B64", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 0), ("SRC1", V + 2)], None),
        ("V_ASHRREV_I64", "ENC_VOP3", &[("VDST", 14), ("SRC0", V + 0), ("SRC1", V + 2)], None),
        ("V_MAD_U64_U32", "VOP3_SDST_ENC", &[("VDST", 16), ("SDST", 20), ("SRC0", V + 4), ("SRC1", V + 5), ("SRC2", V + 2)], None),
        ("V_MAD_I64_I32", "VOP3_SDST_ENC", &[("VDST", 18), ("SDST", 22), ("SRC0", V + 6), ("SRC1", V + 5), ("SRC2", V + 2)], None),
        ("V_CMP_LT_I64", "ENC_VOP3", &[("VDST", 24), ("SRC0", V + 14), ("SRC1", V + 12)], None),
        ("V_CMP_EQ_U64", "ENC_VOP3", &[("VDST", 26), ("SRC0", V + 10), ("SRC1", V + 2)], None),
        // f64 arithmetic, rounding, conversion, and compare (ABS/NEG modifiers)
        ("V_ADD_F64", "ENC_VOP3", &[("VDST", 30), ("SRC0", V + 40), ("SRC1", 242), ("NEG", 1)], None),
        ("V_MUL_F64", "ENC_VOP3", &[("VDST", 32), ("SRC0", V + 40), ("SRC1", V + 42), ("ABS", 1)], None),
        ("V_FMA_F64", "ENC_VOP3", &[("VDST", 34), ("SRC0", V + 40), ("SRC1", V + 42), ("SRC2", V + 44)], None),
        ("V_MIN_F64", "ENC_VOP3", &[("VDST", 36), ("SRC0", V + 30), ("SRC1", V + 42)], None),
        ("V_MAX_F64", "ENC_VOP3", &[("VDST", 38), ("SRC0", V + 30), ("SRC1", V + 42)], None),
        ("V_FLOOR_F64", "ENC_VOP1", &[("VDST", 50), ("SRC0", V + 40)], None),
        ("V_CEIL_F64", "ENC_VOP1", &[("VDST", 52), ("SRC0", V + 40)], None),
        ("V_FRACT_F64", "ENC_VOP1", &[("VDST", 54), ("SRC0", V + 40)], None),
        ("V_RNDNE_F64", "ENC_VOP1", &[("VDST", 56), ("SRC0", V + 40)], None),
        ("V_CVT_F32_F64", "ENC_VOP3", &[("VDST", 58), ("SRC0", V + 34)], None),
        ("V_CVT_F64_F32", "ENC_VOP1", &[("VDST", 60), ("SRC0", V + 58)], None),
        ("V_CMP_LT_F64", "ENC_VOP3", &[("VDST", 28), ("SRC0", V + 40), ("SRC1", V + 42)], None),
    ]);
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let lu = l as u32;
        r[0] = (lu * 3) & 63;
        let v64 = (0x8000_0001_0000_0000u64).wrapping_add((l as u64) * 0x0123_4567_89AB_CDEF);
        r[2] = v64 as u32;
        r[3] = (v64 >> 32) as u32;
        r[4] = 0xFFFF_FFF0u32.wrapping_sub(lu);
        r[5] = lu + 2;
        r[6] = (-100_000i32 + l as i32 * 5000) as u32;

        let da = (l as f64 - 12.5) * 0.75;
        let db = (l as f64 + 1.5) * 0.25;
        let dc = -3.125f64 + l as f64 * 0.125;
        for (idx, val) in [(40, da), (42, db), (44, dc)] {
            let bits = val.to_bits();
            r[idx] = bits as u32;
            r[idx + 1] = (bits >> 32) as u32;
        }
    }
    assert_parity(
        "valu64_int_and_f64",
        &code,
        &st,
        &[20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        &[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61],
    );
}
