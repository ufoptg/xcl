//! Checks of the direct compute translator (`kernel`, `lower`): translated
//! programs pass the validator. The GPU differential tests are the
//! recompiler's instruction tests run with `XCL_DIRECT_DXIL=1`.

use super::kernel::{translate, KernelOptions};
use crate::gpu::rdna2::encode;

fn prog(list: &[(&str, &str, &[(&str, u64)], Option<u32>)]) -> Vec<u8> {
    let mut v = Vec::new();
    for (n, e, f, l) in list {
        v.extend(encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}")));
    }
    v.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    v
}

#[test]
fn straight_line_and_loop_kernels_validate() {
    let code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 138)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 1)], None),
        ("S_SUB_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 1), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFC)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 0)], None),
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 160), ("VSRC1", 0)], None),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", 257), ("VSRC1", 2)], None),
    ]);
    for (wave64, split) in [(false, false), (true, true), (true, false)] {
        translate(&code, KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 }).unwrap_or_else(|e| panic!("wave64 {wave64} split {split}: {e}"));
    }
}

/// Development aid: `XCL_DUMP_HLSL=<file>` compiles a compute shader with DXC
/// and prints the disassembly (DXC's lowering of an HLSL intrinsic is the
/// reference for the direct path's).
#[test]
#[ignore]
fn dump_dxc_lowering() {
    let Ok(path) = std::env::var("XCL_DUMP_HLSL") else { return };
    let src = std::fs::read_to_string(path).unwrap();
    let c = crate::gpu::recomp::dxc::compile(&src, "main", "cs_6_6", &[]).unwrap();
    println!("{}", super::validate::disassemble(&c).unwrap());
}

#[test]
fn f64_conversions_validate() {
    for (name, enc) in [("V_CVT_F32_F64", "ENC_VOP1"), ("V_CVT_F64_F32", "ENC_VOP1"), ("V_CVT_I32_F64", "ENC_VOP1"), ("V_CVT_F64_I32", "ENC_VOP1")] {
        let code = prog(&[(name, enc, &[("VDST", 2), ("SRC0", 256)], None)]);
        if let Err(e) = translate(&code, KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 }) {
            panic!("{name}: {}", &e[..e.len().min(600)]);
        }
    }
    let code = prog(&[("V_FMA_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", 256), ("SRC1", 258), ("SRC2", 260)], None)]);
    translate(&code, KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 }).map_err(|e| e[..e.len().min(600)].to_string()).unwrap();
    let code = prog(&[("V_SQRT_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", 256)], None)]);
    translate(&code, KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 }).map_err(|e| e[..e.len().min(600)].to_string()).unwrap();
}

#[test]
fn f64_bisect() {
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("add", prog(&[("V_ADD_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", 256), ("SRC1", 258)], None)])),
        ("fma_k", prog(&[("V_FMA_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", 256), ("SRC1", 258), ("SRC2", 242)], None)])),
        ("fma_v", prog(&[("V_FMA_F64", "ENC_VOP3", &[("VDST", 2), ("SRC0", 256), ("SRC1", 258), ("SRC2", 260)], None)])),
    ];
    let mut bad = Vec::new();
    for (n, c) in cases {
        if let Err(e) = translate(&c, KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 }) {
            bad.push(format!("{n}: {}", &e[..e.len().min(200)]));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

/// A long program (thousands of guarded memory accesses between a register's write and
/// its final store) translates, validates and stays reasonably fast.
#[test]
fn long_programs_translate() {
    let mut list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = Vec::new();
    list.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 7), ("SRC0", 129)], None));
    for i in 0..2000u64 {
        list.push(("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", vec![("VDST", 2 + (i % 4)), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 4 * (i % 64))], None));
        list.push(("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 1), ("SRC0", 256 + 2 + (i % 4)), ("VSRC1", 1)], None));
    }
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let code = prog(&refs);
    let t = std::time::Instant::now();
    translate(&code, KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 }).unwrap_or_else(|e| panic!("{}", &e[..e.len().min(500)]));
    println!("4001 instructions: {:?}", t.elapsed());
}

/// A direct-translated VS links with an HLSL-path PS and vice versa (both declare their
/// signatures as the HLSL entry structs do), drawing the same pixels as two direct stages.
#[test]
fn direct_and_hlsl_graphics_stages_link() {
    use crate::gpu::dxbc::SigElement as S;
    use crate::gpu::recomp::{runner, translate, Mode, Options, PsInterface, VsInterface};
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32_FLOAT};
    if !super::graphics::tests::draw_ready() {
        return;
    }
    if crate::gpu::recomp::dxc::location().is_err() {
        return runner::skip_test("no DXC");
    }
    let el = |name: &str, index: u32, sv: u32, ct: u32, reg: u32, mask: u8| S { name: name.into(), index, system_value: sv, component_type: ct, register: reg, mask, rw_mask: mask, stream: 0 };
    const V: u64 = 256;
    let vs_code = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
        ("EXP", "ENC_EXP", &[("TGT", 33), ("EN", 0x3), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 0), ("VSRC3", 0)], None),
    ]);
    let vsi = VsInterface {
        inputs: vec![el("POSITION", 0, 0, 3, 0, 3), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF), el("TEXCOORD", 3, 0, 3, 2, 0x3)],
        attr_base: 4,
        params: None,
    };
    let ps_code = prog(&[
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 5), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 1)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 6), ("VSRC", 1), ("ATTR", 1), ("ATTRCHAN", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 242)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None),
    ]);
    let psi = PsInterface {
        inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF), el("TEXCOORD", 3, 0, 3, 2, 0x3)],
        outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x302,
        attrs: None,
    };
    let (vo, po) = (Options { mode: Mode::Vertex(vsi.clone()), wave64: true, ..Default::default() }, Options { mode: Mode::Pixel(psi.clone()), wave64: true, ..Default::default() });
    let hvs = crate::gpu::recomp::dxc::compile(&translate(&vs_code, &vo).unwrap().hlsl, "main", "vs_6_6", &[]).unwrap();
    let hps = crate::gpu::recomp::dxc::compile(&translate(&ps_code, &po).unwrap().hlsl, "main", "ps_6_6", &[]).unwrap();
    let ko = KernelOptions { wave64: true, split: false, mem_windows: 1, sampler_base: 0 };
    let dvs = super::kernel::translate_vertex(&vs_code, ko, &vsi).unwrap().dxil;
    let dps = super::kernel::translate_pixel(&ps_code, ko, &psi).unwrap().dxil;
    let c = [0.8f32, 0.6, 0.4, 1.0];
    let verts: Vec<f32> = [[-1.0f32, -1.0], [-1.0, 3.0], [3.0, -1.0]].iter().flat_map(|p| [p[0], p[1], c[0], c[1], c[2], c[3]]).collect();
    let layout = [
        runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 },
        runner::VertexElement { semantic: "COLOR", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 8 },
    ];
    let draw = |vs: &[u8], ps: &[u8]| runner::draw(vs, ps, &layout, &verts, 24, &[], &[], 8, [0.0; 4]).unwrap();
    let want = draw(&hvs, &hps);
    assert_eq!(draw(&dvs, &dps), want, "direct VS + direct PS");
    assert_eq!(draw(&dvs, &hps), want, "direct VS + HLSL PS");
    assert_eq!(draw(&hvs, &dps), want, "HLSL VS + direct PS");
}

fn sig(name: &str, index: u32, sv: u32, ct: u32, reg: u32, mask: u8) -> crate::gpu::dxbc::SigElement {
    crate::gpu::dxbc::SigElement { name: name.into(), index, system_value: sv, component_type: ct, register: reg, mask, rw_mask: mask, stream: 0 }
}

/// VS / PS with a branch: the blocks after it are created after the exit block, whose
/// output stores read registers they define (once "forward reference to value N",
/// `module.rs` writing blocks in creation order). Also a PS reading SV_IsFrontFace
/// (declared uint, as the validator demands).
#[test]
fn branchy_graphics_stages_validate() {
    use crate::gpu::recomp::{PsInterface, VsInterface};
    let body = |exp: (&'static str, &'static str, &'static [(&'static str, u64)], Option<u32>)| {
        prog(&[
            ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)], None),
            ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 242)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 256 + 4)], None),
            exp,
        ])
    };
    let vs_code = body(("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 4), ("VSRC3", 5), ("DONE", 1)], None));
    let vsi = VsInterface { inputs: vec![], outputs: vec![sig("SV_Position", 0, 1, 3, 0, 0xF)], attr_base: 4, params: None };
    let ko = KernelOptions { wave64: true, split: false, mem_windows: 1, sampler_base: 0 };
    for wave64 in [true, false] {
        super::kernel::translate_vertex(&vs_code, KernelOptions { wave64, ..ko }, &vsi).unwrap_or_else(|e| panic!("VS wave64 {wave64}: {}", &e[..e.len().min(800)]));
    }
    // v2 = front face (SPI_PS_INPUT_ENA bit 12 after the two I/J pairs of 0x3)
    let ps_code = body(("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 2), ("VSRC3", 5), ("DONE", 1), ("VM", 1)], None));
    let psi = PsInterface { inputs: vec![sig("SV_Position", 0, 1, 3, 0, 0xF)], outputs: vec![sig("SV_Target", 0, 64, 3, 0, 0xF)], input_ena: 0x1002, attrs: None };
    for wave64 in [true, false] {
        super::kernel::translate_pixel(&ps_code, KernelOptions { wave64, ..ko }, &psi).unwrap_or_else(|e| panic!("PS wave64 {wave64}: {}", &e[..e.len().min(800)]));
    }
}

/// A long pixel shader (guarded memory accesses make thousands of joins between a
/// register's write and the output stores) translates quickly and stays small: the
/// SSA builder drops trivial phis, `Module::optimize` cleans up the rest.
#[test]
fn long_pixel_shader_translates_fast() {
    use crate::gpu::recomp::PsInterface;
    let mut list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = Vec::new();
    for r in 4..16u64 {
        list.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", r), ("SRC0", 129 + r)], None));
    }
    for i in 0..1500u64 {
        list.push(("BUFFER_LOAD_DWORD", "ENC_MUBUF", vec![("VDATA", 20 + (i % 4)), ("VADDR", 0), ("SRSRC", 1), ("SOFFSET", 128), ("OFFSET", 4 * (i % 64)), ("OFFEN", 1)], None));
        list.push(("V_ADD_F32", "ENC_VOP2", vec![("VDST", 4 + (i % 4)), ("SRC0", 256 + 20 + (i % 4)), ("VSRC1", 4 + (i % 4))], None));
    }
    list.push(("EXP", "ENC_EXP", vec![("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None));
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let code = prog(&refs);
    let psi = PsInterface { inputs: vec![sig("SV_Position", 0, 1, 3, 0, 0xF)], outputs: vec![sig("SV_Target", 0, 64, 3, 0, 0xF)], input_ena: 0x2, attrs: None };
    let t = std::time::Instant::now();
    let k = super::kernel::translate_pixel(&code, KernelOptions { wave64: true, split: false, mem_windows: 1, sampler_base: 0 }, &psi).unwrap_or_else(|e| panic!("{}", &e[..e.len().min(800)]));
    println!("3000 instructions: {:?}, {} B", t.elapsed(), k.dxil.len());
}

/// A vertex and a pixel shader that branch before their exports translate (their epilogue then
/// uses values of blocks created after the exit block), and draw what the HLSL path's stages
/// draw, whichever way the branch goes.
#[test]
fn graphics_stages_with_a_branch_before_their_exports() {
    use crate::gpu::dxbc::SigElement as S;
    use crate::gpu::recomp::{runner, translate, Mode, Options, PsInterface, VsInterface};
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32_FLOAT};
    let el = |name: &str, index: u32, sv: u32, ct: u32, reg: u32, mask: u8| S { name: name.into(), index, system_value: sv, component_type: ct, register: reg, mask, rw_mask: mask, stream: 0 };
    const V: u64 = 256;
    // s0 != 0 skips the instruction that clears a colour channel
    let vs_code = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 128)], None),
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
        ("EXP", "ENC_EXP", &[("TGT", 33), ("EN", 0x3), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 0), ("VSRC3", 0)], None),
    ]);
    let vsi = VsInterface {
        inputs: vec![el("POSITION", 0, 0, 3, 0, 3), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF), el("TEXCOORD", 3, 0, 3, 2, 0x3)],
        attr_base: 4,
        params: None,
    };
    let ps_code = prog(&[
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 5), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 1)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 6), ("VSRC", 1), ("ATTR", 1), ("ATTRCHAN", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 242)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 128)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None),
    ]);
    let psi = PsInterface {
        inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF), el("TEXCOORD", 3, 0, 3, 2, 0x3)],
        outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x302,
        attrs: None,
    };
    let ko = KernelOptions { wave64: true, split: false, mem_windows: 1, sampler_base: 0 };
    let dvs = super::kernel::translate_vertex(&vs_code, ko, &vsi).unwrap_or_else(|e| panic!("VS: {}", &e[..e.len().min(600)])).dxil;
    let dps = super::kernel::translate_pixel(&ps_code, ko, &psi).unwrap_or_else(|e| panic!("PS: {}", &e[..e.len().min(600)])).dxil;
    if !super::graphics::tests::draw_ready() {
        return;
    }
    if crate::gpu::recomp::dxc::location().is_err() {
        return runner::skip_test("no DXC");
    }
    let (vo, po) = (Options { mode: Mode::Vertex(vsi.clone()), wave64: true, ..Default::default() }, Options { mode: Mode::Pixel(psi.clone()), wave64: true, ..Default::default() });
    let hvs = crate::gpu::recomp::dxc::compile(&translate(&vs_code, &vo).unwrap().hlsl, "main", "vs_6_6", &[]).unwrap();
    let hps = crate::gpu::recomp::dxc::compile(&translate(&ps_code, &po).unwrap().hlsl, "main", "ps_6_6", &[]).unwrap();
    let c = [0.8f32, 0.6, 0.4, 1.0];
    let verts: Vec<f32> = [[-1.0f32, -1.0], [-1.0, 3.0], [3.0, -1.0]].iter().flat_map(|p| [p[0], p[1], c[0], c[1], c[2], c[3]]).collect();
    let layout = [
        runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 },
        runner::VertexElement { semantic: "COLOR", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 8 },
    ];
    let draw = |vs: &[u8], ps: &[u8], s0: u32| runner::draw(vs, ps, &layout, &verts, 24, &[s0], &[], 8, [0.0; 4]).unwrap();
    let (cleared, kept) = (draw(&hvs, &hps, 0), draw(&hvs, &hps, 1));
    assert_ne!(cleared, kept, "the branch changes the picture");
    assert_eq!(draw(&dvs, &dps, 0), cleared, "branch not taken");
    assert_eq!(draw(&dvs, &dps, 1), kept, "branch taken");
}

/// Geometry mode (ES+GS, the compute pass of `recomp::Mode::Geometry`): an ES that writes
/// its vertex to the LDS ring, the GS that reads its triangle's vertices, writes them to
/// its output area and emits a strip; split and native forms validate (also the ES or GS
/// alone, as the coverage sweep translates them).
#[test]
fn geometry_mode_validates() {
    use crate::gpu::recomp::GsInterface;
    let e = |n: &str, enc: &str, f: &[(&str, u64)]| encode(n, enc, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));
    let mut es = Vec::new();
    // v20 = lane * ES item (16 B), ring[v20] = v9 (a fetched input)
    es.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", 128)]));
    es.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", 256 + 20)]));
    es.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 20), ("SRC0", 132), ("VSRC1", 20)]));
    es.extend(e("DS_WRITE_B32", "ENC_DS", &[("ADDR", 20), ("DATA0", 9)]));
    es.extend(e("S_BARRIER", "ENC_SOPP", &[]));
    es.extend(e("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 6)]));
    let mut gs = Vec::new();
    // v0 = vertex 0's dword offset: read it, write 3 output vertices at s2 + lane * 48
    gs.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 130), ("VSRC1", 0)]));
    gs.extend(e("DS_READ_B32", "ENC_DS", &[("ADDR", 30), ("VDST", 31)]));
    gs.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 32), ("SRC0", 193), ("SRC1", 128)]));
    gs.extend(e("V_MUL_U32_U24", "ENC_VOP2", &[("VDST", 32), ("SRC0", 176), ("VSRC1", 32)]));
    gs.extend(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 32), ("SRC0", 2), ("VSRC1", 32)]));
    for k in 0..3u64 {
        gs.extend(e("DS_WRITE_B32", "ENC_DS", &[("ADDR", 32), ("DATA0", 31), ("OFFSET0", 16 * k)]));
        gs.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x22)]));
    }
    gs.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x12)]));
    gs.extend(e("S_ENDPGM", "ENC_SOPP", &[]));
    let sig = |name: &str, sv: u32, reg: u32| crate::gpu::dxbc::SigElement { name: name.into(), index: 0, system_value: sv, component_type: 3, register: reg, mask: 0xF, rw_mask: 0xF, stream: 0 };
    let g = GsInterface {
        es_inputs: vec![sig("POSITION", 0, 0)],
        es_outputs: vec![sig("SV_Position", 1, 0)],
        outputs: vec![sig("SV_Position", 1, 0)],
        max_vertex_count: 3,
        gs_start: Some(es.len() as u64),
    };
    let mut code = es.clone();
    code.extend_from_slice(&gs);
    let mut es_end = es;
    es_end.extend(e("S_ENDPGM", "ENC_SOPP", &[]));
    let cases = [
        ("ES+GS", code, g.clone()),
        ("ES alone", es_end, GsInterface { gs_start: None, ..g.clone() }),
        ("GS alone", gs, GsInterface { gs_start: Some(0), ..g }),
    ];
    for (name, code, g) in cases {
        for split in [true, false] {
            let o = KernelOptions { wave64: true, split, mem_windows: 1, sampler_base: 0 };
            super::kernel::translate_geometry(&code, o, &g).unwrap_or_else(|e| panic!("{name} split {split}: {}", &e[..e.len().min(3000)]));
        }
    }
}

/// Exhaustive bit-exact check of the IEEE-754 bit-manipulation formula used by
/// `lower_mem::uf_small_bits` for all 10-bit (mb=5) and 11-bit (mb=6) float inputs.
#[test]
fn uf_small_ieee_bit_math_matches_exact_formula() {
    for mb in [5u32, 6u32] {
        let p = 1u32 << mb;
        let mmask = p - 1;
        let mshift = 23 - mb;
        let den_scale = f32::from_bits(if mb == 6 { 0x3580_0000 } else { 0x3600_0000 });
        for v in 0..(1u32 << (mb + 5)) {
            let e = v >> mb;
            let m = v & mmask;
            // Reference formula (matching HLSL `uf_small`)
            let fm = (m as f32) / (p as f32);
            let ref_f = if e == 31 {
                f32::from_bits(if m != 0 { 0x7FC0_0000 } else { 0x7F80_0000 })
            } else if e == 0 {
                fm * (-14.0f32).exp2()
            } else {
                ((e as f32) - 15.0).exp2() * (1.0 + fm)
            };
            // Exact IEEE bit math
            let got_bits = if e == 31 {
                if m != 0 { 0x7FC0_0000 } else { 0x7F80_0000 }
            } else if e == 0 {
                ((m as f32) * den_scale).to_bits()
            } else {
                ((e + 112) << 23) | (m << mshift)
            };
            assert_eq!(got_bits, ref_f.to_bits(), "mb={mb} v={v:#x} (e={e} m={m})");
        }
    }
}

/// Multi-window switch dispatch, vector rawBufferLoad/Store, hoisted FormatLoad
/// descriptors, single/multi-component FormatLoad, and hoisted BVH loads all pass
/// DXIL validation across 1..=4 memory windows and wave configurations.
#[test]
fn multi_window_mem_format_and_bvh_validate() {
    let code = prog(&[
        // Load descriptor s[12:15] once, then reuse across repeated BUFFER_LOAD_FORMAT_X
        ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 12), ("SBASE", 0), ("OFFSET", 0x40), ("SOFFSET", 125)], None),
        ("S_LOAD_DWORDX8", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 0), ("OFFSET", 0x80), ("SOFFSET", 125)], None),
        ("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 24), ("SBASE", 12), ("OFFSET", 0x10), ("SOFFSET", 125)], None),
        // Divergent EXEC so lane guards are active (non-constant `act`)
        ("V_CMPX_LT_U32", "ENC_VOPC", &[("SRC0", 160), ("VSRC1", 0)], None),
        // Repeated single-component format loads sharing s[12:15] (exercises hoisted FmtComp1 + Load2)
        ("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[("VDATA", 4), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[("VDATA", 5), ("VADDR", 1), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        // Sub-4 and full-4 component format loads + D16 + static MTBUF + small byte load
        ("BUFFER_LOAD_FORMAT_XY", "ENC_MUBUF", &[("VDATA", 6), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 8), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("BUFFER_LOAD_FORMAT_D16_X", "ENC_MUBUF", &[("VDATA", 12), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("BUFFER_LOAD_FORMAT_D16_HI_X", "ENC_MUBUF", &[("VDATA", 12), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("TBUFFER_LOAD_FORMAT_X", "ENC_MTBUF", &[("VDATA", 13), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("FORMAT", 28), ("IDXEN", 1)], None),
        ("BUFFER_LOAD_UBYTE", "ENC_MUBUF", &[("VDATA", 14), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        // Vector MUBUF & FLAT loads/stores
        ("BUFFER_LOAD_DWORDX4", "ENC_MUBUF", &[("VDATA", 16), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("BUFFER_STORE_DWORDX4", "ENC_MUBUF", &[("VDATA", 16), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("GLOBAL_LOAD_DWORDX4", "ENC_FLAT_GLBL", &[("VDST", 20), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 0)], None),
        ("GLOBAL_STORE_DWORDX4", "ENC_FLAT_GLBL", &[("DATA", 20), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 16)], None),
        ("GLOBAL_LOAD_USHORT", "ENC_FLAT_GLBL", &[("VDST", 24), ("ADDR", 0), ("SADDR", 0x7D), ("OFFSET", 4)], None),
        // BVH node intersection (box32 / box16 / tri hoisted vector loads)
        ("IMAGE_BVH_INTERSECT_RAY", "ENC_MIMG", &[("VDATA", 28), ("VADDR", 0), ("SRSRC", 3), ("DMASK", 0xF)], None),
    ]);
    for windows in [1u32, 2, 3, 4] {
        for (wave64, split) in [(true, false), (false, false)] {
            translate(&code, KernelOptions { wave64, split, mem_windows: windows, sampler_base: 0 })
                .unwrap_or_else(|e| panic!("w={windows} wave64={wave64}: {}", &e[..e.len().min(800)]));
            if windows > 1 {
                translate(&code, KernelOptions { wave64, split, mem_windows: windows | crate::gpu::recomp::HEAP_WINDOWS, sampler_base: 0 })
                    .unwrap_or_else(|e| panic!("heap w={windows} wave64={wave64}: {}", &e[..e.len().min(800)]));
            }
        }
    }
}

/// `image_get_lod` with a static sampler (an S# built from constants in the code) translates
/// to `CalculateLevelOfDetail` on the sampler bound at `s<i>, space1`, shared with a sample of
/// the same S#, and validates.
#[test]
fn ps_get_lod_with_a_static_sampler() {
    use crate::gpu::recomp::{Mode, PsInterface};
    let clamp = 2 | 2 << 3 | 2 << 6;
    let linear = [clamp, 0x00FF_F000u32, 1 << 20 | 1 << 22, 0];
    let mut list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = Vec::new();
    for (k, &w) in linear.iter().enumerate() {
        list.push(if w == 0 { ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 24 + k as u64), ("SSRC0", 128)], None) } else { ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 24 + k as u64), ("SSRC0", 255)], Some(w)) });
    }
    // uv = (I, J) of the first barycentric pair (v0, v1); T# s[16:23], S# s[24:27]
    list.push(("IMAGE_GET_LOD", "ENC_MIMG", vec![("VDATA", 4), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x3), ("DIM", 1)], None));
    list.push(("IMAGE_SAMPLE", "ENC_MIMG", vec![("VDATA", 6), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x3), ("DIM", 1)], None));
    list.push(("EXP", "ENC_EXP", vec![("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None));
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let code = prog(&refs);
    let psi = PsInterface { inputs: vec![sig("SV_Position", 0, 1, 3, 0, 0xF)], outputs: vec![sig("SV_Target", 0, 64, 3, 0, 0xF)], input_ena: 0x2, attrs: None };
    let ko = KernelOptions { wave64: true, split: false, mem_windows: 1, sampler_base: 0 };
    let k = super::kernel::translate_pixel(&code, ko, &psi).unwrap_or_else(|e| panic!("{}", &e[..e.len().min(800)]));
    assert_eq!(k.static_samplers, vec![(linear, false)], "one static sampler, shared by get_lod and sample");
    if crate::gpu::recomp::dxc::location().is_ok() {
        super::coverage::translate_mode(&code, &Mode::Pixel(psi), ko, true).unwrap_or_else(|e| panic!("{}", &e[..e.len().min(800)]));
    }
}

/// `image_bvh(64)_intersect_ray` shares one rolled 4-child box loop between box32 and
/// box16: validates across wave / window configs (including heap windows, a guest loop,
/// and statically-known box types) and keeps per-instruction loads / f16 unpacks rolled.
#[test]
fn bvh_intersect_rolls_box_loop_and_validates() {
    use crate::gpu::recomp::{CsInterface, HEAP_WINDOWS};
    const F: &[(&str, u64)] = &[("VDATA", 100), ("VADDR", 0), ("SRSRC", 2), ("DMASK", 15)];
    let one = prog(&[("IMAGE_BVH_INTERSECT_RAY", "ENC_MIMG", F, None)]);
    let two = prog(&[("IMAGE_BVH_INTERSECT_RAY", "ENC_MIMG", F, None), ("IMAGE_BVH_INTERSECT_RAY", "ENC_MIMG", F, None)]);
    let wide = prog(&[("IMAGE_BVH64_INTERSECT_RAY", "ENC_MIMG", F, None)]);
    // Guest traversal loop around image_bvh_intersect_ray (loop-within-loop SSA sealing).
    let looped = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 132)], None),
        ("IMAGE_BVH_INTERSECT_RAY", "ENC_MIMG", F, None),
        ("S_SUB_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 1), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFB)], None),
    ]);
    for (wave64, split, mem_windows) in [(false, false, 1), (true, true, 1), (true, false, 1), (true, false, 2), (true, false, 2 | HEAP_WINDOWS)] {
        let o = KernelOptions { wave64, split, mem_windows, sampler_base: 0 };
        translate(&one, o).unwrap_or_else(|e| panic!("bvh w64={wave64} split={split} win={mem_windows:#x}: {}", &e[..e.len().min(600)]));
    }
    let heap = KernelOptions { wave64: true, split: false, mem_windows: 2 | HEAP_WINDOWS, sampler_base: 0 };
    translate(&wide, heap).unwrap_or_else(|e| panic!("bvh64 heap: {}", &e[..e.len().min(600)]));
    for mem_windows in [1, 2 | HEAP_WINDOWS] {
        let o = KernelOptions { wave64: true, split: false, mem_windows, sampler_base: 0 };
        translate(&looped, o).unwrap_or_else(|e| panic!("looped bvh win={mem_windows:#x}: {}", &e[..e.len().min(600)]));
    }
    // Statically-known box32 (5) and box16 (4) under constant EXEC fold is_box32.
    let cs = CsInterface { threads: [32, 1, 1], user_sgprs: 12, tgid: [false; 3], tg_size: false, tid_components: 1 };
    for src in [133u64, 132] {
        let mov = [("VDST", 0), ("SRC0", src)];
        let const_node = prog(&[("V_MOV_B32", "ENC_VOP1", &mov[..], None), ("IMAGE_BVH_INTERSECT_RAY", "ENC_MIMG", F, None)]);
        for mem_windows in [1, 2 | HEAP_WINDOWS] {
            let o = KernelOptions { wave64: false, split: false, mem_windows, sampler_base: 0 };
            super::kernel::translate_compute(&const_node, o, &cs).unwrap_or_else(|e| panic!("const box {src} win={mem_windows:#x}: {}", &e[..e.len().min(600)]));
        }
    }
    let o = KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 };
    let (c1, c2) = (translate(&one, o).unwrap(), translate(&two, o).unwrap());
    super::validate::validate(&c1).unwrap();
    super::validate::validate(&c2).unwrap();
    let (dis1, dis2) = (super::validate::disassemble(&c1).expect("disassembly"), super::validate::disassemble(&c2).expect("disassembly"));
    let per_bvh = dis2.matches("@dx.op.rawBufferLoad.i32(i32 139").count() - dis1.matches("@dx.op.rawBufferLoad.i32(i32 139").count();
    assert_eq!(per_bvh, 8, "rolled bvh_intersect should emit 8 rawBufferLoad calls (4 box + 4 tri; was 15), got {per_bvh}");
    let f16s = dis1.matches("@dx.op.legacyF16ToF32(i32 131").count();
    assert_eq!(f16s, 6, "rolled box16 should emit 6 legacyF16ToF32 calls (was 24), got {f16s}");
    assert!(dis1.contains("phi float"), "rolled box loop should emit float phis");
}

/// Direct DXIL `image_bvh_intersect_ray` on the GPU: box32 and box16 with 4 scrambled-distance
/// hits (sorted and unsorted), tmax culling, grazing ray (`inv = +inf`), invalid / NaN / inverted
/// children, triangle hit & miss, instance node, and bit-exact agreement with HLSL + DXC.
#[test]
fn bvh_intersect_gpu_matches_box_sort_and_triangle() {
    use crate::gpu::recomp::{dxc, runner, Mode};
    match runner::wave64_programs_supported() {
        Ok(true) => {}
        Ok(false) => return runner::skip_test("no wave64"),
        Err(e) => return runner::skip_test(&e),
    }
    let half_bits = |x: f32| -> u32 {
        if x == 0.0 {
            return 0;
        }
        let b = x.to_bits();
        ((b >> 16) & 0x8000) | (((((b >> 23) & 0xFF) as i32 - 127 + 15) as u32) << 10) | ((b >> 13) & 0x3FF)
    };
    let pack16 = |mn: [f32; 3], mx: [f32; 3]| [half_bits(mn[0]) | (half_bits(mn[1]) << 16), half_bits(mn[2]) | (half_bits(mx[0]) << 16), half_bits(mx[1]) | (half_bits(mx[2]) << 16)];
    let base = 0x1000usize;
    let mut mem = vec![0u32; 0x2000 / 4];
    let w = |mem: &mut Vec<u32>, at: usize, vals: &[u32]| mem[(base + at) / 4..(base + at) / 4 + vals.len()].copy_from_slice(vals);
    let ff = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    const F: u32 = u32::MAX;
    const C0: u32 = 0x10;
    const C1: u32 = 0x20;
    const C2: u32 = 0x30;
    const C3: u32 = 0x40;
    let (box32_id, box16_id, edge_id, tri_id, inst_id) = (5u32, (0x80u32 >> 3) | 4, (0x100u32 >> 3) | 5, 0x180u32 >> 3, (0x1C0u32 >> 3) | 6);
    // box32 @ 0x000: 4 hits at t = 8, 2, 6, 4 → sorted [C1, C3, C2, C0]
    w(&mut mem, 0x000, &[C0, C1, C2, C3]);
    for (k, (z0, z1)) in [(8.0f32, 9.0), (2.0, 3.0), (6.0, 7.0), (4.0, 5.0)].into_iter().enumerate() {
        w(&mut mem, 16 + 24 * k, &ff(&[-1.0, -1.0, z0, 1.0, 1.0, z1]));
    }
    // box16 @ 0x080: 4 hits at t = 7, 3, 9, 1 → sorted [C3, C1, C0, C2]
    w(&mut mem, 0x080, &[C0, C1, C2, C3]);
    for (k, (z0, z1)) in [(7.0f32, 8.0), (3.0, 4.0), (9.0, 10.0), (1.0, 2.0)].into_iter().enumerate() {
        w(&mut mem, 0x080 + 16 + 12 * k, &pack16([-1.0, -1.0, z0], [1.0, 1.0, z1]));
    }
    // box32 @ 0x100: invalid child, NaN box, inverted box (mn > mx), and one valid hit at t = 4
    w(&mut mem, 0x100, &[F, C1, C2, C3]);
    w(&mut mem, 0x100 + 16, &ff(&[-1.0, -1.0, 1.0, 1.0, 1.0, 2.0]));
    w(&mut mem, 0x100 + 40, &ff(&[f32::NAN, -1.0, 2.0, 1.0, 1.0, 3.0]));
    w(&mut mem, 0x100 + 64, &ff(&[2.0, -1.0, 3.0, -2.0, 1.0, 4.0]));
    w(&mut mem, 0x100 + 88, &ff(&[-1.0, -1.0, 4.0, 1.0, 1.0, 5.0]));
    // triangle @ 0x180 in z = 5, triangle id 9 (I = v1, J = v2)
    w(&mut mem, 0x180, &ff(&[-2.0, -1.0, 5.0, 2.0, -1.0, 5.0, -2.0, 3.0, 5.0]));
    w(&mut mem, 0x180 + 60, &[9]);

    let mimg = |vdata: u64, vaddr: u64, srsrc: u64| encode("IMAGE_BVH_INTERSECT_RAY", "ENC_MIMG", &[("VDATA", vdata), ("VADDR", vaddr), ("SRSRC", srsrc), ("DMASK", 15)], None).unwrap();
    let mut code = Vec::new();
    for (vd, va, sr) in [(100, 0, 2), (104, 0, 3), (108, 20, 2), (112, 20, 3), (116, 40, 2), (120, 60, 2), (124, 80, 2)] {
        code.extend(mimg(vd, va, sr));
    }
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());

    let mut st = runner::KernelState::default();
    st.sgpr[8..12].copy_from_slice(&[(base >> 8) as u32, 1 << 31, F, 0]);
    st.sgpr[12..16].copy_from_slice(&[(base >> 8) as u32, 0, F, 0]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (o, tmax) = match l % 4 {
            0 => ([0.0f32, 0.0, 0.0], 100.0f32),
            1 => ([0.0, 0.0, 0.0], 5.0),
            2 => ([-1.0, 0.0, 0.0], 100.0), // grazing ray on mn.x = -1.0 with inv.x = +inf
            _ => ([5.0, 5.0, 0.0], 100.0),
        };
        let ray = [tmax, o[0], o[1], o[2], 0.0, 0.0, 1.0, f32::INFINITY, f32::INFINITY, 1.0];
        for (at, node) in [(0, box32_id), (20, box16_id), (40, edge_id), (60, tri_id), (80, inst_id)] {
            r[at] = node;
            for (k, x) in ray.iter().enumerate() {
                r[at + 1 + k] = x.to_bits();
            }
        }
    }

    let dxil = runner::direct_kernel(&code, true).unwrap_or_else(|e| panic!("direct: {}", &e[..e.len().min(600)]));
    let dres = runner::run(&dxil, &st, &mem).expect("direct GPU run");
    for l in 0..64usize {
        let v = &dres.vgpr[l];
        let q = |at: usize| [v[at], v[at + 1], v[at + 2], v[at + 3]];
        assert_eq!(q(124), [F, F, F, F], "lane {l}: instance node");
        match l % 4 {
            0 | 2 => {
                assert_eq!(q(100), [C1, C3, C2, C0], "lane {l}: box32 sorted");
                assert_eq!(q(104), [C0, C1, C2, C3], "lane {l}: box32 unsorted");
                assert_eq!(q(108), [C3, C1, C0, C2], "lane {l}: box16 sorted");
                assert_eq!(q(112), [C0, C1, C2, C3], "lane {l}: box16 unsorted");
                assert_eq!(q(116), [C3, F, F, F], "lane {l}: box32 invalid/NaN/inverted");
                let tri = q(120).map(f32::from_bits);
                let want_i = if l % 4 == 0 { 0.5 } else { 0.25 };
                assert_eq!((tri[0] / tri[1], tri[2] / tri[1], tri[3] / tri[1]), (5.0, want_i, 0.25), "lane {l}: triangle hit");
            }
            1 => {
                assert_eq!(q(100), [C1, C3, F, F], "lane {l}: box32 sorted tmax=5");
                assert_eq!(q(104), [F, C1, F, C3], "lane {l}: box32 unsorted tmax=5");
                assert_eq!(q(108), [C3, C1, F, F], "lane {l}: box16 sorted tmax=5");
                assert_eq!(q(112), [F, C1, F, C3], "lane {l}: box16 unsorted tmax=5");
                assert_eq!(q(116), [C3, F, F, F], "lane {l}: box32 edge tmax=5");
                let tri = q(120).map(f32::from_bits);
                assert_eq!((tri[0] / tri[1], tri[2] / tri[1], tri[3] / tri[1]), (5.0, 0.5, 0.25), "lane {l}: triangle hit");
            }
            _ => {
                for at in [100, 104, 108, 112, 116] {
                    assert_eq!(q(at), [F, F, F, F], "lane {l} v{at}: miss");
                }
                assert_eq!(q(120), [f32::INFINITY.to_bits(), 1.0f32.to_bits(), 0, 0], "lane {l}: triangle miss");
            }
        }
    }
    if dxc::location().is_ok() {
        let hlsl = crate::gpu::recomp::translate(&code, &runner::options(Mode::Kernel)).unwrap().hlsl;
        let hdxil = dxc::compile(&hlsl, "main", "cs_6_6", &[]).unwrap();
        let hres = runner::run(&hdxil, &st, &mem).expect("HLSL GPU run");
        for l in 0..64usize {
            assert_eq!(dres.vgpr[l][100..128], hres.vgpr[l][100..128], "lane {l}: direct vs HLSL");
        }
    }
}
