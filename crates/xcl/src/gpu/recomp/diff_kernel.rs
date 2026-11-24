//! `XCL_DXIL_DIFF=1`: differential kernel runs (the testing strategy of
//! docs/specs/2026-11-24-direct-dxil.md). `runner::run_code`, `run_code_with` and
//! `run_program` translate the program through both backends (HLSL + DXC and the direct DXIL
//! translator), run both kernels with the same inputs and compare everything a kernel test can
//! observe: SGPRs, SCC, the VGPRs of every lane, exports, flat memory and the UAV test textures
//! (raw). A difference is an error that names the field, the first `SHOWN` differing slots
//! (HLSL value / direct value) and the count; the run otherwise returns the HLSL result.
//! A program the direct translator refuses runs with the HLSL result alone (`refusals` counts
//! them, each distinct reason is logged once), so the whole suite runs in this mode.
//! Tests that compile the HLSL kernel themselves (`kernel_wave`, `direct_or` in
//! `recomp/tests.rs`) `pair` its container with the direct translation; every later
//! `runner::run*` (or, for a compute-mode program, `runner::dispatch`: flat memory only) of
//! that container on the same thread then runs both (a test that runs one kernel from many
//! states is compared for each).
//!
//! Each backend stores only the registers it uses and `res` starts zeroed, so a register that
//! one backend leaves unstored (0) while the other stores its unchanged input value is not a
//! difference: an SGPR on its own, a VGPR only when that holds in every lane that ran (one
//! side 0 in all of them, the other the input in all of them).

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::io::Write;
use std::sync::Mutex;

use super::diff_table::{self, Outcome};
use super::pso::StaticSampler;
use super::runner::{self, KernelResult, KernelState, TestSampler, TestTexture};
use crate::gpu::dxil::kernel::Kernel;
use super::{KERNEL_EXPORTS, KERNEL_LANES, KERNEL_SCC_SLOT, KERNEL_SPLIT_SGPRS, KERNEL_VGPRS};

/// Differing slots listed per field.
pub const SHOWN: usize = 8;

/// Programs the direct translator refused, and the distinct reasons already logged.
static REFUSED: Mutex<(usize, BTreeSet<String>)> = Mutex::new((0, BTreeSet::new()));

thread_local! {
    /// An HLSL kernel container `pair`ed with the lanes it runs, the program (for the
    /// `diff_table` line) and the direct translation of the same program.
    static PAIRED: RefCell<Option<(Vec<u8>, usize, Vec<u8>, Result<Kernel, String>)>> = const { RefCell::new(None) };
}

/// `XCL_DXIL_DIFF=1`: kernel runs go through both backends and must agree.
pub fn enabled() -> bool {
    std::env::var("XCL_DXIL_DIFF").is_ok_and(|v| v == "1")
}

/// Programs the direct translator refused so far (their runs used the HLSL result alone).
pub fn refusals() -> usize {
    REFUSED.lock().unwrap_or_else(|p| p.into_inner()).0
}

fn refused(why: &str) {
    let mut r = REFUSED.lock().unwrap_or_else(|p| p.into_inner());
    r.0 += 1;
    let first = why.lines().next().unwrap_or_default().to_string();
    if r.1.insert(first.clone()) {
        let _ = writeln!(std::io::stderr(), "XCL_DXIL_DIFF: direct DXIL refused a program, HLSL result only: {first}");
    }
}

/// A failure of the direct translator itself (a panic, DXIL it built but could not finish or
/// validate), as opposed to a refusal of an unsupported construct.
fn translator_failed(why: &str) -> bool {
    use crate::gpu::dxil::coverage::{classify, Outcome};
    !matches!(classify(why), Outcome::Unsupported) || why.starts_with("direct DXIL builder error")
}

/// `runner::run_code_with` through both backends; the HLSL result when they agree.
pub fn run_code_with(code: &[u8], state: &KernelState, mem: &[u32], textures: &[TestTexture], samplers: &[TestSampler]) -> Result<KernelResult, String> {
    let t = super::translate(code, &runner::options(super::Mode::Kernel))?;
    if !t.complete() {
        return Err(format!("unsupported instructions: {:?}", t.unsupported));
    }
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|x| format!("{x}\n--- HLSL ---\n{}", t.hlsl))?;
    run_paired(&dxil, KERNEL_LANES, code, runner::direct_kernel_full(code, true), state, mem, textures, samplers, &t.static_samplers)
}

/// Under `XCL_DXIL_DIFF=1`: later runs of the HLSL kernel `hlsl` (a wave64 or wave32
/// translation of the console program `code`) on this thread also run the direct translation
/// of the same program (`direct`, called only in this mode) and compare.
pub fn pair(hlsl: &[u8], code: &[u8], wave64: bool, direct: impl FnOnce() -> Result<Kernel, String>) {
    if enabled() {
        let lanes = if wave64 { KERNEL_LANES } else { KERNEL_LANES / 2 };
        PAIRED.with(|p| *p.borrow_mut() = Some((hlsl.to_vec(), lanes, code.to_vec(), direct())));
    }
}

/// The lanes, program and direct translation `pair`ed with `dxil`, if any (taken out while
/// it runs).
fn take_paired(dxil: &[u8]) -> Option<(usize, Vec<u8>, Result<Kernel, String>)> {
    PAIRED.with(|p| {
        let mut p = p.borrow_mut();
        if p.as_ref().is_some_and(|(h, _, _, _)| h == dxil) {
            p.take().map(|(_, lanes, code, k)| (lanes, code, k))
        } else {
            None
        }
    })
}

/// `runner::run_with_statics` of a `pair`ed container: both backends, compared (the pair
/// stays for the next run). None when `dxil` is not paired.
pub fn run_if_paired(
    dxil: &[u8],
    state: &KernelState,
    mem: &[u32],
    textures: &[TestTexture],
    samplers: &[TestSampler],
    statics: &[StaticSampler],
) -> Option<Result<KernelResult, String>> {
    // taken out first, so the runs below do not come back here
    let (lanes, code, direct) = take_paired(dxil)?;
    let again = copy(&direct);
    let r = run_paired(dxil, lanes, &code, direct, state, mem, textures, samplers, statics);
    PAIRED.with(|p| *p.borrow_mut() = Some((dxil.to_vec(), lanes, code, again)));
    Some(r)
}

/// `Kernel` has no `Clone`.
fn copy(direct: &Result<Kernel, String>) -> Result<Kernel, String> {
    match direct {
        Ok(k) => Ok(Kernel { dxil: k.dxil.clone(), static_samplers: k.static_samplers.clone() }),
        Err(why) => Err(why.clone()),
    }
}

/// `runner::dispatch` of a `pair`ed compute-mode container: both backends, the flat memory
/// they leave compared (the pair stays for the next run). None when `dxil` is not paired.
pub fn dispatch_if_paired(dxil: &[u8], groups: [u32; 3], sgprs: &[u32], mem: &[u32]) -> Option<Result<Vec<u32>, String>> {
    let (lanes, code, direct) = take_paired(dxil)?;
    let again = copy(&direct);
    let r = runner::dispatch(dxil, groups, sgprs, mem).and_then(|h| {
        let k = match direct {
            Ok(k) if k.static_samplers.is_empty() => k,
            Ok(_) => {
                refused("needs static samplers (runner::dispatch binds none)");
                diff_table::record(&code, Outcome::Refused);
                return Ok(h);
            }
            Err(why) if !translator_failed(&why) => {
                refused(&why);
                diff_table::record(&code, Outcome::Refused);
                return Ok(h);
            }
            Err(why) => {
                diff_table::record(&code, Outcome::Failed);
                return Err(format!("XCL_DXIL_DIFF: {why}"));
            }
        };
        let d = runner::dispatch(&k.dxil, groups, sgprs, mem).map_err(|x| {
            diff_table::record(&code, Outcome::Failed);
            format!("XCL_DXIL_DIFF: the direct DXIL dispatch failed (the HLSL one ran): {x}")
        })?;
        let c = compare_mem(&h, &d);
        diff_table::record(&code, if c.is_ok() { Outcome::Matched } else { Outcome::Mismatched });
        c?;
        Ok(h)
    });
    PAIRED.with(|p| *p.borrow_mut() = Some((dxil.to_vec(), lanes, code, again)));
    Some(r)
}

/// The flat memory two dispatches left, compared; the error is the report.
pub fn compare_mem(hlsl: &[u32], direct: &[u32]) -> Result<(), String> {
    let at = |v: &[u32], i: usize| v.get(i).copied().unwrap_or(0);
    let mut fields = vec![field("mem", hlsl.len().max(direct.len()), |i| Some((at(hlsl, i), at(direct, i))), |i| format!("mem[{i}] (byte {:#x})", i * 4))];
    if hlsl.len() != direct.len() {
        fields.push(Field { name: "mem length", count: 1, shown: vec![format!("hlsl {} direct {} dwords", hlsl.len(), direct.len())] });
    }
    report(&fields)
}

/// Runs the HLSL kernel `hlsl` (`lanes` lanes) and, unless the direct translator refused,
/// `direct` with the same inputs; the HLSL result when they agree. `code` (the program) goes
/// into the `diff_table` line.
#[allow(clippy::too_many_arguments)]
fn run_paired(
    hlsl: &[u8],
    lanes: usize,
    code: &[u8],
    direct: Result<Kernel, String>,
    state: &KernelState,
    mem: &[u32],
    textures: &[TestTexture],
    samplers: &[TestSampler],
    statics: &[StaticSampler],
) -> Result<KernelResult, String> {
    let h = runner::run_with_statics(hlsl, state, mem, textures, samplers, statics)?;
    let k = match direct {
        Ok(k) => k,
        Err(why) if !translator_failed(&why) => {
            refused(&why);
            diff_table::record(code, Outcome::Refused);
            return Ok(h);
        }
        Err(why) => {
            diff_table::record(code, Outcome::Failed);
            return Err(format!("XCL_DXIL_DIFF: {why}"));
        }
    };
    let d = runner::run_with_statics(&k.dxil, state, mem, textures, samplers, &k.static_samplers).map_err(|x| {
        diff_table::record(code, Outcome::Failed);
        format!("XCL_DXIL_DIFF: the direct DXIL kernel failed (the HLSL one ran): {x}")
    })?;
    let c = compare(state, lanes, &h, &d);
    diff_table::record(code, if c.is_ok() { Outcome::Matched } else { Outcome::Mismatched });
    c?;
    Ok(h)
}

fn sgpr_name(i: usize) -> String {
    match i {
        106 => "vcc_lo".into(),
        107 => "vcc_hi".into(),
        124 => "m0".into(),
        126 => "exec_lo".into(),
        127 => "exec_hi".into(),
        _ => format!("s{i}"),
    }
}

/// One compared field: the number of differing slots and the first `SHOWN` of them.
struct Field {
    name: &'static str,
    count: usize,
    shown: Vec<String>,
}

/// Slots `0..n` where `pair` gives different values (`None`: the slot is not compared).
fn field(name: &'static str, n: usize, pair: impl Fn(usize) -> Option<(u32, u32)>, label: impl Fn(usize) -> String) -> Field {
    let mut f = Field { name, count: 0, shown: Vec::new() };
    for i in 0..n {
        if let Some((h, d)) = pair(i).filter(|(h, d)| h != d) {
            f.count += 1;
            if f.shown.len() < SHOWN {
                f.shown.push(format!("{}: hlsl {h:#010x} direct {d:#010x}", label(i)));
            }
        }
    }
    f
}

/// A register both backends agree on: equal, or one left it unstored (0) and the other
/// stored its unchanged input value.
fn reg(h: u32, d: u32, input: u32) -> Option<(u32, u32)> {
    if h == d || (h == 0 && d == input) || (d == 0 && h == input) {
        None
    } else {
        Some((h, d))
    }
}

fn dword(b: &[u8], i: usize) -> u32 {
    let mut w = [0u8; 4];
    for (k, x) in w.iter_mut().enumerate() {
        *x = b.get(i * 4 + k).copied().unwrap_or(0);
    }
    u32::from_le_bytes(w)
}

/// Compares the observable state of an HLSL and a direct DXIL run of `lanes` lanes from
/// `state`; the error is the report.
pub fn compare(state: &KernelState, lanes: usize, hlsl: &KernelResult, direct: &KernelResult) -> Result<(), String> {
    let at = |v: &[u32], i: usize| v.get(i).copied().unwrap_or(0);
    let vin = |lane: usize, r: usize| state.vgpr.get(lane).and_then(|v| v.get(r)).copied().unwrap_or(0);
    let vg = |x: &KernelResult, l: usize, r: usize| x.vgpr.get(l).and_then(|v| v.get(r)).copied().unwrap_or(0);
    // VGPRs one side left unstored in every lane that ran while the other kept the input
    let unstored: Vec<bool> = (0..KERNEL_VGPRS)
        .map(|r| {
            let all = |a: &KernelResult, b: &KernelResult| (0..lanes).all(|l| vg(a, l, r) == 0 && vg(b, l, r) == vin(l, r));
            all(hlsl, direct) || all(direct, hlsl)
        })
        .collect();
    let mut fields = vec![
        field(
            "sgpr",
            hlsl.sgpr.len().max(direct.sgpr.len()),
            |i| if i == KERNEL_SCC_SLOT { None } else { reg(at(&hlsl.sgpr, i), at(&direct.sgpr, i), at(&state.sgpr, i)) },
            sgpr_name,
        ),
        field("scc", 1, |_| reg(hlsl.scc, direct.scc, state.scc), |_| "scc".into()),
        field(
            "vgpr",
            KERNEL_LANES * KERNEL_VGPRS,
            |i| {
                let (l, r) = (i / KERNEL_VGPRS, i % KERNEL_VGPRS);
                if unstored[r] { None } else { Some((vg(hlsl, l, r), vg(direct, l, r))) }
            },
            |i| format!("lane {} v{}", i / KERNEL_VGPRS, i % KERNEL_VGPRS),
        ),
        field(
            "exports",
            KERNEL_SPLIT_SGPRS - KERNEL_EXPORTS,
            |i| Some((at(&hlsl.raw, KERNEL_EXPORTS + i), at(&direct.raw, KERNEL_EXPORTS + i))),
            |i| format!("exp{} lane {}.{}", i / (KERNEL_LANES * 4), i / 4 % KERNEL_LANES, ['x', 'y', 'z', 'w'][i % 4]),
        ),
        field(
            "mem",
            hlsl.mem.len().max(direct.mem.len()),
            |i| Some((at(&hlsl.mem, i), at(&direct.mem, i))),
            |i| format!("mem[{i}] (byte {:#x})", i * 4),
        ),
    ];
    if hlsl.mem.len() != direct.mem.len() {
        fields.push(Field { name: "mem length", count: 1, shown: vec![format!("hlsl {} direct {} dwords", hlsl.mem.len(), direct.mem.len())] });
    }
    for t in 0..hlsl.textures.len().max(direct.textures.len()) {
        let (h, d) = (hlsl.textures.get(t).map_or(&[][..], |v| v), direct.textures.get(t).map_or(&[][..], |v| v));
        let mut f = field("texture", h.len().max(d.len()).div_ceil(4), |i| Some((dword(h, i), dword(d, i))), |i| format!("texture {t} dword {i} (byte {:#x})", i * 4));
        if h.len() != d.len() {
            f.count += 1;
            f.shown.push(format!("texture {t} length: hlsl {} direct {} bytes", h.len(), d.len()));
        }
        fields.push(f);
    }
    report(&fields)
}

/// The report of the compared `fields`: Ok when none differs.
fn report(fields: &[Field]) -> Result<(), String> {
    let total: usize = fields.iter().map(|f| f.count).sum();
    if total == 0 {
        return Ok(());
    }
    let mut s = format!("XCL_DXIL_DIFF: HLSL and direct DXIL runs differ in {total} values");
    for f in fields.iter().filter(|f| f.count > 0) {
        s.push_str(&format!("\n  {} ({} differ):", f.name, f.count));
        for line in &f.shown {
            s.push_str(&format!("\n    {line}"));
        }
        if f.count > f.shown.len() {
            s.push_str(&format!("\n    ... {} more", f.count - f.shown.len()));
        }
    }
    Err(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::rdna2::encode;
    use crate::gpu::recomp::{KERNEL_RES_DWORDS, KERNEL_SGPRS};

    fn blank() -> KernelResult {
        KernelResult {
            sgpr: vec![0; KERNEL_SGPRS],
            vgpr: vec![vec![0; KERNEL_VGPRS]; KERNEL_LANES],
            scc: 0,
            raw: vec![0; KERNEL_RES_DWORDS],
            mem: vec![0; 64],
            textures: Vec::new(),
        }
    }

    #[test]
    fn equal_results_compare_equal() {
        let st = KernelState::default();
        let mut a = blank();
        a.sgpr[3] = 7;
        a.vgpr[9][4] = 0xDEAD;
        a.mem[2] = 1;
        a.textures = vec![vec![1, 2, 3, 4, 5]];
        let mut b = blank();
        b.sgpr[3] = 7;
        b.vgpr[9][4] = 0xDEAD;
        b.mem[2] = 1;
        b.textures = vec![vec![1, 2, 3, 4, 5]];
        assert_eq!(compare(&st, KERNEL_LANES, &a, &b), Ok(()));
    }

    #[test]
    fn report_names_field_slot_values_and_count() {
        let st = KernelState::default();
        let (mut a, mut b) = (blank(), blank());
        a.vgpr[5][3] = 1;
        b.vgpr[5][3] = 2;
        a.sgpr[126] = 0xFFFF_0000;
        b.scc = 1;
        let e = compare(&st, KERNEL_LANES, &a, &b).unwrap_err();
        assert!(e.contains("differ in 3 values"), "{e}");
        assert!(e.contains("vgpr (1 differ):\n    lane 5 v3: hlsl 0x00000001 direct 0x00000002"), "{e}");
        assert!(e.contains("exec_lo: hlsl 0xffff0000 direct 0x00000000"), "{e}");
        assert!(e.contains("scc (1 differ):\n    scc: hlsl 0x00000000 direct 0x00000001"), "{e}");
        assert!(!e.contains("mem"), "{e}");
    }

    #[test]
    fn report_lists_the_first_shown_differences_only() {
        let st = KernelState::default();
        let (a, mut b) = (blank(), blank());
        for i in 0..20 {
            b.mem[i] = i as u32 + 1;
        }
        let e = compare(&st, KERNEL_LANES, &a, &b).unwrap_err();
        assert!(e.contains("mem (20 differ):"), "{e}");
        assert!(e.contains("mem[7] (byte 0x1c): hlsl 0x00000000 direct 0x00000008"), "{e}");
        assert!(!e.contains("mem[8]"), "{e}");
        assert!(e.contains("... 12 more"), "{e}");
    }

    #[test]
    fn unstored_unchanged_registers_are_not_differences() {
        let mut st = KernelState::default();
        st.sgpr[10] = 100;
        st.vgpr[2][7] = 0x55;
        let (mut a, mut b) = (blank(), blank());
        // HLSL stored the input values, direct DXIL left them unstored
        a.sgpr[10] = 100;
        a.vgpr[2][7] = 0x55;
        a.sgpr[126] = u32::MAX;
        a.sgpr[127] = u32::MAX;
        assert_eq!(compare(&st, KERNEL_LANES, &a, &b), Ok(()));
        // a value that is neither the input nor unstored is a difference
        b.sgpr[10] = 5;
        let e = compare(&st, KERNEL_LANES, &a, &b).unwrap_err();
        assert!(e.contains("s10: hlsl 0x00000064 direct 0x00000005"), "{e}");
    }

    #[test]
    fn exports_and_textures_are_decoded() {
        let st = KernelState::default();
        let (mut a, mut b) = (blank(), blank());
        a.raw[KERNEL_EXPORTS + (2 * KERNEL_LANES + 7) * 4 + 1] = 0x3F80_0000;
        a.textures = vec![vec![0; 8], vec![0; 8]];
        b.textures = vec![vec![0; 8], vec![0, 0, 0, 0, 0x78, 0x56, 0x34, 0x12]];
        let e = compare(&st, KERNEL_LANES, &a, &b).unwrap_err();
        assert!(e.contains("exp2 lane 7.y: hlsl 0x3f800000 direct 0x00000000"), "{e}");
        assert!(e.contains("texture 1 dword 1 (byte 0x4): hlsl 0x00000000 direct 0x12345678"), "{e}");
        assert!(!e.contains("texture 0"), "{e}");
        // a texture size mismatch is reported too
        b.textures[0].pop();
        assert!(compare(&st, KERNEL_LANES, &a, &b).unwrap_err().contains("texture 0 length: hlsl 8 direct 7 bytes"));
    }

    #[test]
    fn a_pair_is_found_by_its_own_container_only() {
        let k = || Ok(Kernel { dxil: vec![9], static_samplers: Vec::new() });
        PAIRED.with(|p| *p.borrow_mut() = Some((vec![1, 2, 3], 32, vec![7], k())));
        assert!(take_paired(&[1, 2]).is_none());
        let (lanes, code, k) = take_paired(&[1, 2, 3]).unwrap();
        assert_eq!((lanes, code, k.unwrap().dxil), (32, vec![7], vec![9]));
        // taken out while it runs (`run_if_paired` puts it back afterwards)
        assert!(take_paired(&[1, 2, 3]).is_none());
        assert!(run_if_paired(&[1, 2], &KernelState::default(), &[], &[], &[], &[]).is_none());
    }

    #[test]
    fn a_vgpr_zeroed_in_one_lane_is_a_difference() {
        let mut st = KernelState::default();
        for v in st.vgpr.iter_mut() {
            v[4] = 0x55;
        }
        let (mut a, mut b) = (blank(), blank());
        for l in 0..KERNEL_LANES {
            a.vgpr[l][4] = 0x55;
            b.vgpr[l][4] = 0x55;
        }
        assert_eq!(compare(&st, KERNEL_LANES, &a, &b), Ok(()));
        // direct stored the register but zeroed an EXEC-masked lane
        b.vgpr[3][4] = 0;
        let e = compare(&st, KERNEL_LANES, &a, &b).unwrap_err();
        assert!(e.contains("vgpr (1 differ):\n    lane 3 v4: hlsl 0x00000055 direct 0x00000000"), "{e}");
        // a wave32 run: lanes 32.. ran in neither, so one side unstored still matches
        let mut c = blank();
        for l in 0..32 {
            c.vgpr[l][4] = 0x55;
        }
        assert_eq!(compare(&st, 32, &c, &blank()), Ok(()));
        assert!(compare(&st, KERNEL_LANES, &c, &blank()).is_err());
    }

    #[test]
    fn dispatch_memory_is_compared() {
        assert_eq!(compare_mem(&[1, 2, 3], &[1, 2, 3]), Ok(()));
        let e = compare_mem(&[1, 2, 3], &[1, 7, 3]).unwrap_err();
        assert!(e.contains("mem (1 differ):\n    mem[1] (byte 0x4): hlsl 0x00000002 direct 0x00000007"), "{e}");
        assert!(compare_mem(&[1, 2], &[1, 2, 0]).unwrap_err().contains("hlsl 2 direct 3 dwords"));
    }

    #[test]
    fn refusals_are_not_translator_failures() {
        assert!(!translator_failed("+0x10: no direct lowering for some instruction"));
        assert!(translator_failed("direct DXIL failed validation: bad"));
        assert!(translator_failed("direct DXIL builder error: bad"));
        assert!(translator_failed("the direct DXIL translator panicked: overflow"));
    }

    /// A small program through both backends via the diff entry point (independent of
    /// `XCL_DXIL_DIFF`): the runs agree and give the expected registers.
    #[test]
    fn gpu_diff_run_of_a_small_program() {
        match (runner::adapter(), runner::wave64_programs_supported(), crate::gpu::recomp::dxc::location()) {
            (Ok(_), Ok(true), Ok(_)) => {}
            (a, w, d) => {
                runner::skip_test(if a.is_err() || w.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" });
                return;
            }
        }
        let mut code = Vec::new();
        code.extend(encode("S_ADD_U32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 0), ("SSRC1", 1)], None).unwrap());
        code.extend(encode("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 2), ("VSRC1", 0)], None).unwrap());
        code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        // the program must reach the direct backend, not just its refusal path
        runner::direct_kernel_full(&code, true).unwrap_or_else(|e| panic!("direct DXIL refused the test program: {e}"));
        let mut st = KernelState::default();
        st.sgpr[0] = 5;
        st.sgpr[1] = 7;
        for (lane, v) in st.vgpr.iter_mut().enumerate() {
            v[0] = lane as u32;
        }
        let r = run_code_with(&code, &st, &[1, 2, 3], &[], &[]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((r.sgpr[2], r.scc), (12, 0));
        for lane in 0..KERNEL_LANES {
            assert_eq!(r.vgpr[lane][1], 12 + lane as u32, "lane {lane}");
        }
        assert_eq!(&r.mem[..3], &[1, 2, 3]);
    }
}
