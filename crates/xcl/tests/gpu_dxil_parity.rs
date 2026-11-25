//! GPU differential parity tests between the direct RDNA2 → DXIL translator
//! (`xcl::gpu::dxil::kernel`) and the HLSL + DXC path (`xcl::gpu::recomp`),
//! covering cross-lane/DPP wave ops, LDS atomics + barriers, and MIMG image
//! ops + 32-bit image atomics across native wave64, split wave64, and wave32.

use windows::Win32::Graphics::Direct3D12::{
    D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32_SINT, DXGI_FORMAT_R32_UINT,
};
use xcl::gpu::dxil::kernel::{translate_full, KernelOptions};
use xcl::gpu::dxil::validate;
use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::{
    self, dxc,
    runner::{self, KernelState, TexDim, TestSampler, TestTexture},
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

fn tsharp(idx: u32, dfmt: u32, rtype: u32) -> [u32; 8] {
    [0x2000_0000 | (idx << 4), 0, 0, (dfmt << 20) | (rtype << 28), 0, 0, 0, 0]
}

fn assert_parity(
    label: &str,
    code: &[u8],
    st: &KernelState,
    tex: &[TestTexture],
    samp: &[TestSampler],
    check_sgpr: &[usize],
    check_vgpr: &[usize],
    check_tex: &[usize],
) {
    for (cfg, wave64, split) in configs() {
        let lanes = if wave64 { 64 } else { 32 };
        let ko = KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
        let direct = translate_full(code, ko).unwrap_or_else(|e| panic!("{label}/{cfg}: direct: {e}"));
        validate::validate(&direct.dxil).unwrap_or_else(|e| panic!("{label}/{cfg}: validate: {e}"));
        let opt = Options { mode: Mode::Kernel, wave64, split_wave64: split, ..Default::default() };
        let t = recomp::translate(code, &opt).unwrap_or_else(|e| panic!("{label}/{cfg}: hlsl: {e}"));
        assert!(t.complete(), "{label}/{cfg}: incomplete HLSL: {:?}", t.unsupported_why);
        let hlsl = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{label}/{cfg}: dxc: {e}\n{}", t.hlsl));
        let got = runner::run_with(&direct.dxil, st, &[], tex, samp).unwrap_or_else(|e| panic!("{label}/{cfg}: run direct: {e}"));
        let want = runner::run_with(&hlsl, st, &[], tex, samp).unwrap_or_else(|e| panic!("{label}/{cfg}: run hlsl: {e}"));
        for &s in check_sgpr {
            assert_eq!(got.sgpr[s], want.sgpr[s], "{label}/{cfg}: s{s}");
        }
        for l in 0..lanes {
            for &v in check_vgpr {
                assert_eq!(got.vgpr[l][v], want.vgpr[l][v], "{label}/{cfg}: lane {l} v{v}");
            }
        }
        for &ti in check_tex {
            assert_eq!(got.textures[ti], want.textures[ti], "{label}/{cfg}: texture {ti}");
        }
    }
}

/// Direct DXIL vs HLSL parity for cross-lane / wave ops: readlane, readfirstlane (under
/// divergent EXEC), writelane, ds_swizzle (quad + XOR), ds_bpermute,
/// v_permlane16 / v_permlanex16 (including FI/BC flags), and DPP16 / DPP8.
#[test]
fn direct_dxil_cross_lane_and_dpp_parity() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        // v_readlane from low half (lane 3) and high half (lane 37 -> lane 5 in wave32)
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 0), ("SRC1", 128 + 3)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 0), ("SRC1", 128 + 37)], None),
        // v_writelane into lane 7
        ("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 2), ("SRC0", 0), ("SRC1", 128 + 7)], None),
        // ds_swizzle: quad reverse [3,2,1,0] (0x801B) and XOR-1 (0x041F)
        ("DS_SWIZZLE_B32", "ENC_DS", &[("VDST", 10), ("ADDR", 0), ("OFFSET0", 0x1B), ("OFFSET1", 0x80)], None),
        ("DS_SWIZZLE_B32", "ENC_DS", &[("VDST", 11), ("ADDR", 0), ("OFFSET0", 0x1F), ("OFFSET1", 0x04)], None),
        // ds_bpermute using lane byte-index in v1
        ("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 12), ("ADDR", 1), ("DATA0", 0)], None),
        // v_permlane16 / v_permlanex16 (s1/s2 = lane select masks; v13 tests FI=1, BC=1)
        ("V_PERMLANE16_B32", "ENC_VOP3", &[("VDST", 13), ("SRC0", V + 0), ("SRC1", 1), ("SRC2", 2), ("OP_SEL", 0b11)], None),
        ("V_PERMLANE16_B32", "ENC_VOP3", &[("VDST", 14), ("SRC0", V + 0), ("SRC1", 1), ("SRC2", 2)], None),
        ("V_PERMLANEX16_B32", "ENC_VOP3", &[("VDST", 15), ("SRC0", V + 0), ("SRC1", 1), ("SRC2", 2)], None),
        // DPP16 quad reverse (0x1B) and DPP8 ([7,6,5,4,3,2,1,0])
        ("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 16), ("SRC0", 0xFA), ("VSRC0", 0), ("DPP_CTRL", 0x1B), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)], None),
        ("V_MOV_B32", "VOP1_VOP_DPP8", &[("VDST", 17), ("SRC0", 0xE9), ("VSRC0", 0), ("LANE_SEL_0", 7), ("LANE_SEL_1", 6), ("LANE_SEL_2", 5), ("LANE_SEL_3", 4), ("LANE_SEL_4", 3), ("LANE_SEL_5", 2), ("LANE_SEL_6", 1), ("LANE_SEL_7", 0)], None),
        // Divergent EXEC (drop lanes 0..3): v_readfirstlane must pick lane 4
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 255)], Some(0xFFFF_FFF0)),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 127), ("SSRC0", 255)], Some(0x5555_5555)),
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 12), ("SRC0", V + 0)], None),
        ("DS_SWIZZLE_B32", "ENC_DS", &[("VDST", 18), ("ADDR", 0), ("OFFSET0", 0x1B), ("OFFSET1", 0x80)], None),
    ]);
    let mut st = KernelState::default();
    st.sgpr[0] = 0xDEAD_BEEF;
    st.sgpr[1] = 0x7654_3210;
    st.sgpr[2] = 0xFEDC_BA98;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let lu = l as u32;
        r[0] = 0x1000_0000 | (lu * 0x0101);
        r[1] = ((lu ^ 5) & 31) * 4;
        r[2] = 0xCAFE_0000 | lu;
        r[18] = 0x5555_0000 | lu;
    }
    assert_parity(
        "cross_lane_and_dpp",
        &code,
        &st,
        &[],
        &[],
        &[10, 11, 12],
        &[2, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        &[],
    );
}

/// Direct DXIL vs HLSL parity for LDS atomics (`add`, `sub`, `and`, `or`, `xor`,
/// `min`/`max` signed & unsigned, `wrxchg`, `cmpst`, `inc`, `dec`), multi-lane
/// contention, `s_barrier`, and divergent EXEC masking.
#[test]
fn direct_dxil_lds_atomics_and_barrier_parity() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        // Initialize per-lane LDS slots [l*4] and shared contention slots [256..272]
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 0), ("DATA0", 1)], None),
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 0), ("DATA0", 2), ("OFFSET0", 0), ("OFFSET1", 1)], None), // +256
        ("S_BARRIER", "ENC_SOPP", &[], None),
        // Per-lane sequential atomics at slot l*4
        ("DS_ADD_RTN_U32", "ENC_DS", &[("VDST", 10), ("ADDR", 0), ("DATA0", 3)], None),
        ("DS_SUB_RTN_U32", "ENC_DS", &[("VDST", 11), ("ADDR", 0), ("DATA0", 4)], None),
        ("DS_AND_RTN_B32", "ENC_DS", &[("VDST", 12), ("ADDR", 0), ("DATA0", 5)], None),
        ("DS_OR_RTN_B32", "ENC_DS", &[("VDST", 13), ("ADDR", 0), ("DATA0", 6)], None),
        ("DS_XOR_RTN_B32", "ENC_DS", &[("VDST", 14), ("ADDR", 0), ("DATA0", 7)], None),
        ("DS_WRXCHG_RTN_B32", "ENC_DS", &[("VDST", 15), ("ADDR", 0), ("DATA0", 8)], None),
        ("DS_CMPST_RTN_B32", "ENC_DS", &[("VDST", 16), ("ADDR", 0), ("DATA0", 8), ("DATA1", 9)], None),
        ("DS_INC_RTN_U32", "ENC_DS", &[("VDST", 17), ("ADDR", 0), ("DATA0", 20)], None),
        ("DS_DEC_RTN_U32", "ENC_DS", &[("VDST", 18), ("ADDR", 0), ("DATA0", 20)], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 19), ("ADDR", 0)], None),
        // Multi-lane contention on shared slots 512 (add), 516 (umin), 520 (umax), 524 (smin), 528 (smax)
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 21), ("DATA0", 22)], None), // lane 0..3 init 512..528
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("DS_ADD_U32", "ENC_DS", &[("ADDR", 23), ("DATA0", 24), ("OFFSET0", 0)], None),
        ("DS_MIN_U32", "ENC_DS", &[("ADDR", 23), ("DATA0", 25), ("OFFSET0", 4)], None),
        ("DS_MAX_U32", "ENC_DS", &[("ADDR", 23), ("DATA0", 26), ("OFFSET0", 8)], None),
        ("DS_MIN_I32", "ENC_DS", &[("ADDR", 23), ("DATA0", 27), ("OFFSET0", 12)], None),
        ("DS_MAX_I32", "ENC_DS", &[("ADDR", 23), ("DATA0", 28), ("OFFSET0", 16)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 30), ("ADDR", 23), ("OFFSET0", 0)], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 31), ("ADDR", 23), ("OFFSET0", 4)], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 32), ("ADDR", 23), ("OFFSET0", 8)], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 33), ("ADDR", 23), ("OFFSET0", 12)], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 34), ("ADDR", 23), ("OFFSET0", 16)], None),
        // Divergent EXEC: even lanes only increment slot l*4 + 256
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 255)], Some(0x5555_5555)),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 127), ("SSRC0", 255)], Some(0x5555_5555)),
        ("DS_ADD_RTN_U32", "ENC_DS", &[("VDST", 35), ("ADDR", 0), ("DATA0", 3), ("OFFSET1", 1)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 255)], Some(0xFFFF_FFFF)),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 127), ("SSRC0", 255)], Some(0xFFFF_FFFF)),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 36), ("ADDR", 0), ("OFFSET1", 1)], None),
    ]);
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let lu = l as u32;
        r[0] = lu * 4;
        r[1] = 100 + lu * 7;
        r[2] = 500 + lu;
        r[3] = 13;
        r[4] = 5;
        r[5] = 0x0F0F_FFF3;
        r[6] = 0x00A0_0004;
        r[7] = 0x00FF_00AA;
        r[8] = if l % 2 == 0 { 10 } else { 3 };
        r[9] = if l % 2 == 0 { 10 } else { 7 }; // cmpswap matches even lanes, misses odd
        r[20] = 10; // inc/dec wrap limit
        r[21] = 512 + (lu % 5) * 4;
        r[22] = match l % 5 {
            0 => 0,
            1 => u32::MAX,
            2 => 0,
            3 => 100,
            _ => (-1000i32) as u32,
        };
        r[23] = 512;
        r[24] = lu + 1;
        r[25] = if l == 5 { 9 } else { 0x8000_0000 | lu };
        r[26] = if l == 9 { 0xE000_0000 } else { lu + 20 };
        r[27] = if l == 7 { i32::MIN as u32 } else { (lu as i32 - 12) as u32 };
        r[28] = if l == 11 { i32::MAX as u32 } else { (-400 + lu as i32) as u32 };
        r[35] = 0xAAAA_0000 | lu;
    }
    assert_parity(
        "lds_atomics_and_barrier",
        &code,
        &st,
        &[],
        &[],
        &[],
        &[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 30, 31, 32, 33, 34, 35, 36],
        &[],
    );
}

/// Direct DXIL vs HLSL parity for MIMG texture ops (`image_load`, `image_store`,
/// `image_gather4_lz`) and 32-bit `image_atomic_*` (`add`, `sub`, `and`, `or`, `xor`,
/// `swap` GLC=0/1, `umin`, `umax`, `smin`, `smax`, `inc`, `dec`, `cmpswap`, contention, EXEC).
#[test]
fn direct_dxil_mimg_image_ops_and_atomics_parity() {
    if !gpu_ready() {
        return;
    }
    // tex0: 16x4 RGBA32F SRV (SRSRC 4->0)
    // tex1: 16x4 RGBA32F UAV (SRSRC 6->2)
    // tex2: 16x4 R32_UINT UAV (SRSRC 8->4) — per-lane int atomics + swap GLC=1/0 + inc/dec/cmpswap
    // tex3: 16x4 R32_SINT UAV (SRSRC 10->6) — per-lane smin/smax
    // tex4: 1x1  R32_UINT UAV (SRSRC 12->8) — contention add
    let code = prog(&[
        ("IMAGE_LOAD", "ENC_MIMG", &[("VDATA", 4), ("VADDR", 0), ("SRSRC", 4), ("DMASK", 0xF), ("DIM", 1), ("UNORM", 1)], None),
        ("IMAGE_STORE", "ENC_MIMG", &[("VDATA", 4), ("VADDR", 0), ("SRSRC", 6), ("DMASK", 0xF), ("DIM", 1), ("UNORM", 1)], None),
        ("IMAGE_GATHER4_LZ", "ENC_MIMG", &[("VDATA", 8), ("VADDR", 2), ("SRSRC", 4), ("SSAMP", 14), ("DMASK", 0x1), ("DIM", 1)], None),
        ("IMAGE_ATOMIC_ADD", "ENC_MIMG", &[("VDATA", 12), ("VADDR", 0), ("SRSRC", 8), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1), ("GLC", 1)], None),
        ("IMAGE_ATOMIC_SUB", "ENC_MIMG", &[("VDATA", 13), ("VADDR", 0), ("SRSRC", 8), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1), ("GLC", 1)], None),
        ("IMAGE_ATOMIC_AND", "ENC_MIMG", &[("VDATA", 14), ("VADDR", 0), ("SRSRC", 8), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1), ("GLC", 1)], None),
        ("IMAGE_ATOMIC_OR", "ENC_MIMG", &[("VDATA", 15), ("VADDR", 0), ("SRSRC", 8), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1), ("GLC", 1)], None),
        ("IMAGE_ATOMIC_XOR", "ENC_MIMG", &[("VDATA", 16), ("VADDR", 0), ("SRSRC", 8), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1), ("GLC", 1)], None),
        ("IMAGE_ATOMIC_SWAP", "ENC_MIMG", &[("VDATA", 17), ("VADDR", 0), ("SRSRC", 8), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1), ("GLC", 1)], None),
        ("IMAGE_ATOMIC_SWAP", "ENC_MIMG", &[("VDATA", 18), ("VADDR", 0), ("SRSRC", 8), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1), ("GLC", 0)], None),
        ("IMAGE_ATOMIC_INC", "ENC_MIMG", &[("VDATA", 19), ("VADDR", 0), ("SRSRC", 8), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1), ("GLC", 1)], None),
        ("IMAGE_ATOMIC_DEC", "ENC_MIMG", &[("VDATA", 20), ("VADDR", 0), ("SRSRC", 8), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1), ("GLC", 1)], None),
        ("IMAGE_ATOMIC_CMPSWAP", "ENC_MIMG", &[("VDATA", 22), ("VADDR", 0), ("SRSRC", 8), ("DMASK", 0x3), ("DIM", 1), ("UNORM", 1), ("GLC", 1)], None),
        ("IMAGE_ATOMIC_SMIN", "ENC_MIMG", &[("VDATA", 24), ("VADDR", 0), ("SRSRC", 10), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1), ("GLC", 1)], None),
        ("IMAGE_ATOMIC_SMAX", "ENC_MIMG", &[("VDATA", 25), ("VADDR", 0), ("SRSRC", 10), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1), ("GLC", 1)], None),
        ("IMAGE_ATOMIC_ADD", "ENC_MIMG", &[("VDATA", 26), ("VADDR", 28), ("SRSRC", 12), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1)], None),
    ]);
    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 14, 9)); // RGBA32F
    st.sgpr[24..32].copy_from_slice(&tsharp(2, 14, 9));
    st.sgpr[32..40].copy_from_slice(&tsharp(4, 20, 9)); // R32_UINT
    st.sgpr[40..48].copy_from_slice(&tsharp(6, 21, 9)); // R32_SINT
    st.sgpr[48..56].copy_from_slice(&tsharp(8, 20, 9));
    st.sgpr[56..60].copy_from_slice(&[0x3000_0000, 0, 0, 0]); // sampler 0
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let lu = l as u32;
        r[0] = lu % 16;
        r[1] = lu / 16;
        r[2] = (((lu % 16) as f32 + 0.5) / 16.0).to_bits();
        r[3] = (((lu / 16) as f32 + 0.5) / 4.0).to_bits();
        r[12] = 7;
        r[13] = 3;
        r[14] = 0x0F0F_FFF0;
        r[15] = 0x00A0_0005;
        r[16] = 0xFFFF_00FF;
        r[17] = 0xABCD_0000 | lu;
        r[18] = if l % 2 == 0 { 10 } else { 4 };
        r[19] = 10; // inc limit
        r[20] = 10; // dec limit
        r[22] = 0x7777_0000 | lu; // cmpswap new
        r[23] = if l % 2 == 0 { 10 } else { 99 }; // cmpswap cmp
        r[24] = (-200i32) as u32;
        r[25] = if l % 2 == 0 { (-50i32) as u32 } else { i32::MAX as u32 };
        r[26] = lu + 1;
        r[28] = 0;
        r[29] = 0;
    }
    let f32_data: Vec<u8> = (0..64u32)
        .flat_map(|i| [i as f32, (i * 2) as f32, (i as f32) * 0.25, 1.0f32])
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let u32_init: Vec<u8> = (0..64u32).flat_map(|l| (5 + l * 3).to_le_bytes()).collect();
    let i32_init: Vec<u8> = (0..64u32)
        .flat_map(|l| (if l % 2 == 0 { (-100i32 + l as i32) as u32 } else { 50 + l }).to_le_bytes())
        .collect();
    let tex = vec![
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 16, height: 4, depth: 1, texel_bytes: 16, data: f32_data, uav: false },
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 16, height: 4, depth: 1, texel_bytes: 16, data: vec![0; 64 * 16], uav: true },
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_UINT, width: 16, height: 4, depth: 1, texel_bytes: 4, data: u32_init, uav: true },
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_SINT, width: 16, height: 4, depth: 1, texel_bytes: 4, data: i32_init, uav: true },
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_UINT, width: 1, height: 1, depth: 1, texel_bytes: 4, data: 0u32.to_le_bytes().to_vec(), uav: true },
    ];
    let samp = vec![TestSampler {
        filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        comparison: None,
    }];
    assert_parity(
        "mimg_ops_and_atomics",
        &code,
        &st,
        &tex,
        &samp,
        &[],
        &[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 25],
        &[0, 1, 2, 3],
    );
}
