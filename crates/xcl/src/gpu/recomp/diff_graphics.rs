//! `XCL_DXIL_DIFF=1`: differential testing of the graphics stages. A test that draws console
//! VS / PS code (or a mesh-mode first pass) through the HLSL path draws it once more with the
//! direct DXIL translations (`gpu::dxil::kernel`) of the same code, and the two renders (and
//! the mesh records) must agree bit for bit. A stage the direct translator refuses keeps the
//! test's container (logged once per stage and reason); with nothing translated directly
//! there is nothing to compare. A failure of the translator itself (validation, builder
//! error, panic) is an error, not a refusal. Unset, `check` / `check_mesh` do nothing.

use std::collections::BTreeSet;
use std::sync::Mutex;

use super::{runner, Mode, Options};

/// The differing values a report lists.
const SHOWN: usize = 8;

/// `XCL_DXIL_DIFF=1`
pub fn enabled() -> bool {
    std::env::var("XCL_DXIL_DIFF").is_ok_and(|v| v == "1")
}

/// The direct DXIL container of a console program the tests translate with `opt` (VS, PS or
/// mesh-mode first pass), with the options `tests::direct_or` gives the direct translator;
/// Err when it refuses the program.
pub fn direct_stage(code: &[u8], opt: &Options) -> Result<Vec<u8>, String> {
    use crate::gpu::dxil::kernel::{translate_mesh, translate_pixel, translate_vertex, KernelOptions};
    if opt.mem_windows > 4 || !(opt.hw_wave == 0 || (opt.hw_wave == 32 && !opt.wave64)) {
        return Err("memory windows / hardware wave size the direct translator does not take".into());
    }
    // the split form is the mesh pass's alone (a VS / PS is wave-size independent)
    let split = matches!(opt.mode, Mode::Mesh(_)) && opt.wave64 && (opt.split_wave64 || runner::split_wave64());
    let mem_windows = if opt.heap_windows && opt.mem_windows > 1 { opt.mem_windows | super::HEAP_WINDOWS } else { opt.mem_windows };
    let o = KernelOptions { wave64: opt.wave64, split, mem_windows, sampler_base: opt.static_sampler_base };
    match &opt.mode {
        Mode::Vertex(vs) => translate_vertex(code, o, vs).map(|k| k.dxil),
        Mode::Pixel(ps) => translate_pixel(code, o, ps).map(|k| k.dxil),
        Mode::Mesh(ms) => translate_mesh(code, o, ms).map(|k| k.dxil),
        _ => Err("not a graphics stage".into()),
    }
}

fn stage_name(mode: &Mode) -> &'static str {
    match mode {
        Mode::Vertex(_) => "VS",
        Mode::Pixel(_) => "PS",
        Mode::Mesh(_) => "mesh pass",
        _ => "kernel",
    }
}

/// A failure of the direct translator itself (DXIL it built but could not finish or
/// validate, or a panic), as opposed to a refusal of an unsupported construct.
fn translator_failed(why: &str) -> bool {
    why.starts_with("direct DXIL failed validation") || why.starts_with("direct DXIL builder error") || why.starts_with("the direct DXIL translator panicked")
}

/// `direct_stage`, or None (logged once per stage and reason) where it is refused; Err where
/// the translator failed.
fn direct_or_note(code: &[u8], opt: &Options) -> Result<Option<Vec<u8>>, String> {
    use std::io::Write;
    static NOTED: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());
    match direct_stage(code, opt) {
        Ok(dxil) => Ok(Some(dxil)),
        Err(e) if translator_failed(&e) => Err(format!("XCL_DXIL_DIFF: the direct translator failed on the {}: {e}", stage_name(&opt.mode))),
        Err(e) => {
            let why = e.lines().next().unwrap_or("").chars().take(200).collect::<String>();
            let stage = stage_name(&opt.mode);
            if NOTED.lock().unwrap().insert(format!("{stage}: {why}")) {
                // to stderr itself, as `runner::skip_test`: libtest captures `eprint!`
                let _ = writeln!(std::io::stderr(), "XCL_DXIL_DIFF: the direct translator refuses the {stage} ({why}); it stays HLSL");
            }
            Ok(None)
        }
    }
}

/// One pixel: its components, and the bits of a NaN (whose payload `{:?}` hides).
fn show(p: &[f32; 4]) -> String {
    let c: Vec<String> = p.iter().map(|x| if x.is_nan() { format!("NaN({:#010x})", x.to_bits()) } else { format!("{x:?}") }).collect();
    format!("[{}]", c.join(", "))
}

/// Two renders of a `width`-pixel-wide target (row by row) compared bit for bit: Err names
/// how many pixels differ and the first `SHOWN` as (x, y): HLSL / direct.
pub fn compare_pixels(what: &str, width: usize, hlsl: &[[f32; 4]], direct: &[[f32; 4]]) -> Result<(), String> {
    if hlsl.len() != direct.len() {
        return Err(format!("{what}: HLSL drew {} pixels, direct DXIL {}", hlsl.len(), direct.len()));
    }
    let bits = |p: &[f32; 4]| p.map(f32::to_bits);
    let differ: Vec<usize> = (0..hlsl.len()).filter(|&i| bits(&hlsl[i]) != bits(&direct[i])).collect();
    if differ.is_empty() {
        return Ok(());
    }
    let w = width.max(1);
    let first: Vec<String> = differ.iter().take(SHOWN).map(|&i| format!("({}, {}): {} / {}", i % w, i / w, show(&hlsl[i]), show(&direct[i]))).collect();
    Err(format!("{what}: {} of {} pixels differ (x, y): HLSL / direct DXIL: {}", differ.len(), hlsl.len(), first.join("; ")))
}

/// Dwords (mesh records, exports) compared: Err names how many differ and the first `SHOWN`
/// as [index]: HLSL / direct.
pub fn compare_words(what: &str, hlsl: &[u32], direct: &[u32]) -> Result<(), String> {
    if hlsl.len() != direct.len() {
        return Err(format!("{what}: HLSL gave {} dwords, direct DXIL {}", hlsl.len(), direct.len()));
    }
    let differ: Vec<usize> = (0..hlsl.len()).filter(|&i| hlsl[i] != direct[i]).collect();
    if differ.is_empty() {
        return Ok(());
    }
    let first: Vec<String> = differ.iter().take(SHOWN).map(|&i| format!("[{i}]: {:#010x} / {:#010x}", hlsl[i], direct[i])).collect();
    Err(format!("{what}: {} of {} dwords differ, [index]: HLSL / direct DXIL: {}", differ.len(), hlsl.len(), first.join("; ")))
}

/// `XCL_DXIL_DIFF=1`: `check_now`; unset, nothing (no translation, no draw).
pub fn check<F>(vs: &[u8], ps: &[u8], vs_code: (&[u8], &Options), ps_code: (&[u8], &Options), width: usize, draw: F) -> Result<(), String>
where
    F: FnMut(&[u8], &[u8]) -> Result<Vec<[f32; 4]>, String>,
{
    if !enabled() {
        return Ok(());
    }
    check_now(vs, ps, vs_code, ps_code, width, draw)
}

/// `draw` (one render of a VS + PS pair, a `width`-pixel-wide target) with the test's
/// containers `vs` / `ps`, and again with the direct translations of their console code
/// (`vs_code` / `ps_code` with the `Options` they were translated with), compared bit for bit.
/// A stage the direct translator refuses keeps the test's container; both refused: Ok.
/// A failure of the translator itself is an Err.
pub fn check_now<F>(vs: &[u8], ps: &[u8], vs_code: (&[u8], &Options), ps_code: (&[u8], &Options), width: usize, mut draw: F) -> Result<(), String>
where
    F: FnMut(&[u8], &[u8]) -> Result<Vec<[f32; 4]>, String>,
{
    let (dvs, dps) = (direct_or_note(vs_code.0, vs_code.1)?, direct_or_note(ps_code.0, ps_code.1)?);
    if dvs.is_none() && dps.is_none() {
        return Ok(());
    }
    let what = format!("{} VS + {} PS", if dvs.is_some() { "direct" } else { "HLSL" }, if dps.is_some() { "direct" } else { "HLSL" });
    let want = draw(vs, ps)?;
    let got = draw(dvs.as_deref().unwrap_or(vs), dps.as_deref().unwrap_or(ps)).map_err(|e| format!("{what}: {e}"))?;
    compare_pixels(&what, width, &want, &got)
}

/// `XCL_DXIL_DIFF=1`: a mesh-mode draw (`draw` of a first pass, as `runner::mesh_draw`:
/// pixels of a `width`-pixel-wide target and records) with the test's first pass `cs`, and
/// again with the direct translation of its console code, pixels and the live part of the
/// records (`live_records`) compared bit for bit. Unset, or the direct translator refuses the
/// program: nothing; a failure of the translator itself is an Err.
pub fn check_mesh<F>(cs: &[u8], code: (&[u8], &Options), width: usize, mut draw: F) -> Result<(), String>
where
    F: FnMut(&[u8]) -> Result<(Vec<[f32; 4]>, Vec<u32>), String>,
{
    if !enabled() {
        return Ok(());
    }
    let Some(dcs) = direct_or_note(code.0, code.1)? else {
        return Ok(());
    };
    let (want_px, want_rec) = draw(cs)?;
    let (px, rec) = draw(&dcs).map_err(|e| format!("direct mesh pass: {e}"))?;
    if let Mode::Mesh(ms) = &code.1.mode {
        compare_words("direct mesh pass records", &live_records(ms, &want_rec), &live_records(ms, &rec))?;
    }
    compare_pixels("direct mesh pass", width, &want_px, &px)
}

/// Mesh records (`ms.record_dwords()` each) with what the pass-through mesh shader does not
/// read set to 0: the spare header words, vertices past min(count, max_vertices) and
/// primitives past min(count, max_primitives). The counts stay, so differing counts differ.
pub fn live_records(ms: &super::MsInterface, rec: &[u32]) -> Vec<u32> {
    let (rd, vd) = (ms.record_dwords(), ms.vert_dwords());
    let (mv, mp) = (ms.max_vertices as usize, ms.max_primitives as usize);
    let mut out = rec.to_vec();
    for r in out.chunks_exact_mut(rd) {
        let (nv, np) = ((r[0] as usize).min(mv), (r[1] as usize).min(mp));
        r[2..super::MESH_HEADER].fill(0);
        r[super::MESH_HEADER + nv * vd..super::MESH_HEADER + mv * vd].fill(0);
        r[super::MESH_HEADER + mv * vd + np..].fill(0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_renders_compare_equal() {
        let px = vec![[0.5, -0.0, f32::NAN, 1.0]; 16];
        assert_eq!(compare_pixels("t", 4, &px, &px.clone()), Ok(()));
        assert_eq!(compare_words("t", &[1, 2, 3], &[1, 2, 3]), Ok(()));
    }

    #[test]
    fn report_counts_and_lists_the_first_eight_with_coordinates() {
        let want = vec![[0.0f32; 4]; 16 * 16];
        let mut got = want.clone();
        // 10 pixels differ: (3, 0), then x = 0..9 of row 2
        got[3][1] = 1.0;
        for x in 0..9 {
            got[2 * 16 + x][0] = 0.25;
        }
        let e = compare_pixels("direct VS + direct PS", 16, &want, &got).unwrap_err();
        assert!(e.starts_with("direct VS + direct PS: 10 of 256 pixels differ"), "{e}");
        assert!(e.contains("(3, 0): [0.0, 0.0, 0.0, 0.0] / [0.0, 1.0, 0.0, 0.0]"), "{e}");
        assert!(e.contains("(6, 2): "), "{e}");
        assert!(!e.contains("(7, 2): "), "only the first 8: {e}");
        assert_eq!(e.matches("; ").count(), 7, "8 listed: {e}");
    }

    #[test]
    fn comparison_is_bitwise() {
        // -0.0 == 0.0 as floats, but not as bits; a NaN equals itself as bits
        let e = compare_pixels("t", 1, &[[0.0; 4]], &[[-0.0, 0.0, 0.0, 0.0]]).unwrap_err();
        assert!(e.contains("1 of 1 pixels differ") && e.contains("[-0.0, 0.0, 0.0, 0.0]"), "{e}");
        let (a, b) = (f32::from_bits(0x7fc0_0000), f32::from_bits(0x7fc0_0001));
        assert_eq!(compare_pixels("t", 1, &[[a; 4]], &[[a; 4]]), Ok(()));
        let e = compare_pixels("t", 1, &[[a; 4]], &[[b; 4]]).unwrap_err();
        assert!(e.contains("NaN(0x7fc00000)") && e.contains("NaN(0x7fc00001)"), "{e}");
    }

    #[test]
    fn size_mismatch_and_words_report() {
        let e = compare_pixels("t", 4, &[[0.0; 4]; 4], &[[0.0; 4]; 3]).unwrap_err();
        assert_eq!(e, "t: HLSL drew 4 pixels, direct DXIL 3");
        let e = compare_words("records", &[0; 20], &[1; 20]).unwrap_err();
        assert!(e.starts_with("records: 20 of 20 dwords differ"), "{e}");
        assert!(e.contains("[0]: 0x00000000 / 0x00000001") && e.contains("[7]: ") && !e.contains("[8]: "), "{e}");
        assert_eq!(compare_words("r", &[1], &[]).unwrap_err(), "r: HLSL gave 1 dwords, direct DXIL 0");
    }

    #[test]
    fn unset_check_does_not_draw() {
        if enabled() {
            return;
        }
        let o = Options::default();
        let r = check(&[], &[], (&[], &o), (&[], &o), 1, |_, _| panic!("drawn with XCL_DXIL_DIFF unset"));
        assert_eq!(r, Ok(()));
        let r = check_mesh(&[], (&[], &o), 1, |_| panic!("drawn with XCL_DXIL_DIFF unset"));
        assert_eq!(r, Ok(()));
    }

    #[test]
    fn refusals_are_not_translator_failures() {
        assert!(!translator_failed("+0x10: v_foo: unsupported"));
        assert!(translator_failed("direct DXIL failed validation: bad"));
        assert!(translator_failed("direct DXIL builder error: bad"));
        assert!(translator_failed("the direct DXIL translator panicked: bad"));
    }

    #[test]
    fn live_records_keep_counts_live_vertices_and_primitives() {
        let ms = super::super::MsInterface { threads: 64, max_vertices: 3, max_primitives: 2, ..Default::default() };
        let (rd, vd) = (ms.record_dwords(), ms.vert_dwords());
        assert_eq!((rd, vd), (4 + 3 * 4 + 2, 4));
        // two records: 2 vertices / 1 primitive, then counts above the maxima (clamped)
        let rec: Vec<u32> = (0..2 * rd as u32).map(|i| i + 100).collect();
        let mut a = rec.clone();
        (a[0], a[1], a[rd], a[rd + 1]) = (2, 1, 9, 9);
        let live = live_records(&ms, &a);
        assert_eq!(&live[..4], &[2, 1, 0, 0]);
        assert_eq!(&live[4..4 + 2 * vd], &a[4..4 + 2 * vd]);
        assert!(live[4 + 2 * vd..4 + 3 * vd].iter().all(|&x| x == 0));
        assert_eq!((live[4 + 3 * vd], live[4 + 3 * vd + 1]), (a[4 + 3 * vd], 0));
        assert_eq!(&live[rd + 4..], &a[rd + 4..]);
        // a dead slot differs: no difference; a count differs: a difference
        let mut b = a.clone();
        b[4 + 2 * vd] ^= 1;
        assert_eq!(compare_words("r", &live, &live_records(&ms, &b)), Ok(()));
        b[1] = 0;
        assert!(compare_words("r", &live, &live_records(&ms, &b)).is_err());
    }

    #[test]
    fn kernel_mode_is_not_a_graphics_stage() {
        assert_eq!(direct_stage(&[], &Options::default()), Err("not a graphics stage".into()));
    }

    fn el(name: &str, sv: u32, reg: u32, mask: u8) -> crate::gpu::dxbc::SigElement {
        crate::gpu::dxbc::SigElement { name: name.into(), index: 0, system_value: sv, component_type: 3, register: reg, mask, rw_mask: mask, stream: 0 }
    }

    fn ins(name: &str, enc: &str, fields: &[(&str, u64)], lit: Option<u32>) -> Vec<u8> {
        crate::gpu::rdna2::encode(name, enc, fields, lit).unwrap_or_else(|x| panic!("{name}: {x}"))
    }

    /// A synthetic wave64 VS + PS pair: the VS passes POSITION (v4 / v5) and COLOR (v8..v11)
    /// through; the PS keeps pixels with centre x > 8, interpolates COLOR and writes
    /// (2·r + 0.5, g·b, -b, 1).
    fn pair() -> ((Vec<u8>, Mode), (Vec<u8>, Mode)) {
        const V: u64 = 256;
        let mut vs: Vec<u8> = [
            ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None),
            ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None),
            ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
            ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
            ins("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1)], None),
            ins("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
        ]
        .concat();
        vs.extend(ins("S_ENDPGM", "ENC_SOPP", &[], None));
        let mut ps: Vec<u8> = [
            ins("V_CMPX_LT_F32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 2)], Some(8.0f32.to_bits())),
            ins("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", 0)], None),
            ins("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 0)], None),
            ins("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 5), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", 1)], None),
            ins("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 5), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 1)], None),
            ins("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 6), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", 2)], None),
            ins("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 6), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 2)], None),
            ins("V_FMA_F32", "ENC_VOP3", &[("VDST", 4), ("SRC0", 244), ("SRC1", V + 4), ("SRC2", 240)], None),
            ins("V_MUL_F32", "ENC_VOP2", &[("VDST", 5), ("SRC0", V + 6), ("VSRC1", 5)], None),
            ins("V_SUB_F32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 128), ("VSRC1", 6)], None),
            ins("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 242)], None),
            ins("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None),
        ]
        .concat();
        ps.extend(ins("S_ENDPGM", "ENC_SOPP", &[], None));
        let vsi = super::super::VsInterface {
            inputs: vec![el("POSITION", 0, 0, 3), el("COLOR", 0, 1, 0xF)],
            outputs: vec![el("SV_Position", 1, 0, 0xF), el("COLOR", 0, 1, 0xF)],
            attr_base: 4,
            params: None,
        };
        let psi = super::super::PsInterface {
            inputs: vec![el("SV_Position", 1, 0, 0xF), el("COLOR", 0, 1, 0xF)],
            outputs: vec![el("SV_Target", 64, 0, 0xF)],
            input_ena: 0x302,
            attrs: None,
        };
        ((vs, Mode::Vertex(vsi)), (ps, Mode::Pixel(psi)))
    }

    fn wave64(mode: Mode) -> Options {
        Options { mode, wave64: true, ..Default::default() }
    }

    /// The synthetic pair goes through the direct translator (no GPU; its validator needs
    /// DXC): `check` would otherwise compare nothing.
    #[test]
    fn synthetic_pair_translates_directly() {
        if super::super::dxc::location().is_err() {
            return runner::skip_test("no DXC");
        }
        let ((vs, vm), (ps, pm)) = pair();
        let (vo, po) = (wave64(vm), wave64(pm));
        direct_stage(&vs, &vo).unwrap_or_else(|e| panic!("VS: {}", &e[..e.len().min(800)]));
        direct_stage(&ps, &po).unwrap_or_else(|e| panic!("PS: {}", &e[..e.len().min(800)]));
    }

    /// The synthetic pair drawn through both backends (whatever `XCL_DXIL_DIFF` says): HLSL
    /// + DXC and direct DXIL render the same bits, and the PS really ran.
    #[test]
    fn gpu_synthetic_pair_draws_the_same_through_both_backends() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32_FLOAT};
        match (runner::adapter(), runner::wave64_programs_supported(), super::super::dxc::location()) {
            (Ok(_), Ok(true), Ok(_)) => {}
            (a, w, d) => return runner::skip_test(if a.is_err() || w.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" }),
        }
        let ((vs_code, vm), (ps_code, pm)) = pair();
        let (vo, po) = (runner::options(vm), runner::options(pm));
        let compile = |code: &[u8], opt: &Options, profile: &str| {
            let t = super::super::translate(code, opt).unwrap();
            assert!(t.complete(), "{:?}", t.unsupported_why);
            super::super::dxc::compile(&t.hlsl, "main", profile, &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl))
        };
        let (vs, ps) = (compile(&vs_code, &vo, "vs_6_6"), compile(&ps_code, &po, "ps_6_6"));
        let c = [0.75f32, 0.5, 0.25, 1.0];
        let verts: Vec<f32> = [[-1.0f32, -1.0], [-1.0, 3.0], [3.0, -1.0]].iter().flat_map(|p| [p[0], p[1], c[0], c[1], c[2], c[3]]).collect();
        let layout = [
            runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 },
            runner::VertexElement { semantic: "COLOR", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 8 },
        ];
        let clear = [0.0, 0.0, 1.0, 1.0];
        let draw = |vs: &[u8], ps: &[u8]| runner::draw(vs, ps, &layout, &verts, 24, &[], &[], 16, clear);
        let px = draw(&vs, &ps).unwrap();
        assert!(px[8].iter().zip([2.0, 0.125, -0.25, 1.0]).all(|(a, b)| (a - b).abs() < 1e-5), "pixel (8, 0) = {:?}", px[8]);
        assert_eq!(px[7], clear, "pixel (7, 0) is discarded");
        check_now(&vs, &ps, (&vs_code, &vo), (&ps_code, &po), 16, draw).unwrap();
    }
}
