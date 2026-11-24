//! `xcl shaders <file|folder|pack> --explain`: where the compile cost of console programs comes
//! from, stage by stage: the guest program, the direct DXIL translator's output
//! and the HLSL path (HLSL → DXC → DXIL, [`hlsl`])
//! ([`direct`]).
//!
//! Each number is *portable* or *local*. Portable numbers are counts that depend only on the
//! program, XCL's commit, the DXC release and the options in the fingerprint; another machine
//! with the same fingerprint gets the same values (the report's `portable digest` says so in
//! one line). Local numbers are this machine's times and memory, comparable only with a run on
//! the same machine. `--explain-diff` ([`diff`]) compares two `--json` reports: portable numbers
//! always, local ones only when both reports come from the same machine.
//!
//! Programs are named by container hash (FNV-64, as `[SHADER]` log lines and `--hash` use it),
//! FSSM index and code offset. Record names and code are not printed, so a report can be
//! posted without game data in it.

mod direct;
mod diff;
mod hlsl;
mod limits;

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use serde_json::{json, Map, Value};

use super::Input;
use crate::gpu::dxil::cfg::{Cfg, Term};
use crate::gpu::dxil::{coverage, stats};
use crate::gpu::recomp::{self, dxc, Mode};
use crate::gpu::{rdna2, shader, xbox};
pub(super) use diff::diff;

/// The report format.
const FORMAT: &str = "xcl-explain/1";

/// What decides the translation (all of it is in the fingerprint), and the run's limits.
struct Opts {
    windows: u32,
    /// memory windows through the descriptor heap (`--heap`, or the runtime's `XCL_HEAP_WINDOWS`)
    heap: bool,
    /// None: what XCL picks on this machine's GPU
    split: Option<bool>,
    /// None: the level XCL picks by source size
    opt: Option<String>,
    hlsl: bool,
    direct: bool,
    /// the direct translator's time limit (`--timeout`, default 120 s)
    timeout: Duration,
    /// commit limit of this process and each dxc.exe (`--mem-gb`, default 8)
    mem_bytes: u64,
    /// `--dxc-timeout` (default 600 s)
    dxc_timeout: Duration,
    top: usize,
}

struct Prog {
    hash: u64,
    fssm: usize,
    offset: usize,
    code: Vec<u8>,
    mode: Mode,
}

impl Opts {
    /// `mem_windows` for both translators (`recomp::HEAP_WINDOWS` set for heap windows)
    fn mem_windows(&self) -> u32 {
        if self.heap && self.windows > 1 { self.windows | recomp::HEAP_WINDOWS } else { self.windows }
    }
}

impl Prog {
    fn id(&self) -> String {
        format!("{:016x}:f{}:{:x}", self.hash, self.fssm, self.offset)
    }
}

/// The value after `name` on the command line.
fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

fn arg_num(args: &[String], name: &str) -> Option<u64> {
    arg(args, name).and_then(|v| v.parse().ok())
}

pub(super) fn run(input: &Input, args: &[String], json_out: bool) -> i32 {
    let value = |name: &str| arg(args, name);
    let num = |name: &str| arg_num(args, name);
    let paths = value("--path").unwrap_or("both");
    let o = Opts {
        windows: num("--windows").unwrap_or(4).clamp(1, 4) as u32,
        split: num("--split").map(|v| v != 0),
        heap: args.iter().any(|a| a == "--heap") || recomp::pso::heap_windows(),
        opt: value("--opt").filter(|v| *v != "auto").map(|v| if v.starts_with('-') { v.to_string() } else { format!("-{v}") }),
        hlsl: paths != "direct",
        direct: paths != "hlsl",
        timeout: Duration::from_secs(num("--timeout").unwrap_or(120)),
        mem_bytes: num("--mem-gb").unwrap_or(8).max(1) << 30,
        dxc_timeout: Duration::from_secs(num("--dxc-timeout").unwrap_or(600)),
        top: num("--top").unwrap_or(8) as usize,
    };
    // the run never grows the page file: past the limit an allocation fails instead
    let own_job = limits::Job::new(o.mem_bytes);
    if own_job.as_ref().is_none_or(|j| !j.assign(unsafe { windows_sys::Win32::System::Threading::GetCurrentProcess() })) {
        eprintln!("xcl shaders --explain: cannot limit this process's memory; the direct translator is not capped");
    }
    if let Some(l) = &o.opt {
        if !matches!(l.as_str(), "-O0" | "-O1" | "-O2" | "-O3") {
            eprintln!("xcl shaders --explain: --opt {l}: use auto, O0, O1, O2 or O3");
            return 2;
        }
    }
    let progs = match select(input, args) {
        Ok(p) if p.is_empty() => {
            eprintln!("xcl shaders --explain: no console program selected");
            return 2;
        }
        Ok(p) => p,
        Err(e) => {
            eprintln!("xcl shaders --explain: {e}");
            return 2;
        }
    };
    let split_auto = o.split.is_none();
    let split = o.split.unwrap_or_else(|| recomp::pso::split_wave64_forced() || recomp::runner::wave64_supported().is_ok_and(|w| !w));
    let (fp_portable, fp_local) = fingerprint(&o, split, split_auto);
    if json_out {
        // JSON Lines, one line per program as it finishes: a run that is stopped keeps them
        println!("{}", json!({ "format": FORMAT, "kind": "header", "fingerprint": { "portable": fp_portable.clone(), "local": fp_local.clone() } }));
    } else {
        print_fingerprint(&fp_portable, &fp_local);
    }
    let mut digest_parts = Vec::new();
    for (i, p) in progs.iter().enumerate() {
        if json_out {
            eprintln!("  {}/{}: {}", i + 1, progs.len(), p.id());
        }
        let (portable, local) = explain(p, split, &o);
        if json_out {
            println!("{}", json!({ "kind": "program", "id": p.id(), "stage": coverage::stage_name(&p.mode), "portable": portable.clone(), "local": local }));
        } else {
            print_program(p, &portable, &local, &o);
        }
        digest_parts.push(json!({ "id": p.id(), "portable": portable }));
    }
    let digest = fnv64_str(&serde_json::to_string(&json!({ "fingerprint": fp_portable, "programs": digest_parts })).unwrap_or_default());
    if json_out {
        println!("{}", json!({ "kind": "footer", "programs": progs.len(), "portable_digest": format!("{digest:016x}") }));
        return 0;
    }
    println!("portable digest {digest:016x} (equal on every machine with the same portable fingerprint)");
    0
}

/// `--hash H[,H…]` (every program of those containers) or a seeded sample (`--sample N`, default
/// 20 programs; `--seed S`).
fn select(input: &Input, args: &[String]) -> Result<Vec<Prog>, String> {
    let value = |name: &str| arg(args, name);
    let num = |name: &str| arg_num(args, name);
    let mut out = Vec::new();
    if let Some(list) = value("--hash") {
        let mut want = Vec::new();
        for h in list.split(',') {
            let hex = h.trim().trim_start_matches("0x");
            want.push(u64::from_str_radix(hex, 16).map_err(|_| format!("--hash {h}: not a hexadecimal hash"))?);
        }
        let mut found = Vec::new();
        for i in 0..input.len() {
            let (_, Some(blob)) = input.get(i) else { continue };
            let h = shader::fnv64(&blob);
            if want.contains(&h) && !found.contains(&h) {
                found.push(h);
                programs_of(&blob, h, &mut out);
            }
        }
        let missing: Vec<String> = want.iter().filter(|h| !found.contains(h)).map(|h| format!("{h:016x}")).collect();
        if !missing.is_empty() {
            return Err(format!("no record with hash {}", missing.join(", ")));
        }
    } else {
        let want = num("--sample").unwrap_or(20).max(1) as usize;
        for ri in coverage::sample_indices(input.len(), want, num("--seed").unwrap_or(1)) {
            if out.len() >= want {
                break;
            }
            let (_, Some(blob)) = input.get(ri) else { continue };
            programs_of(&blob, shader::fnv64(&blob), &mut out);
        }
        out.truncate(want);
    }
    Ok(out)
}

/// The programs of a console container that have a recompiler interface.
fn programs_of(blob: &[u8], hash: u64, out: &mut Vec<Prog>) {
    let Some(sh) = xbox::parse(blob) else { return };
    for (fi, f) in sh.fssm.iter().enumerate() {
        let Some(mode) = recomp::interface(blob, &sh, f) else { continue };
        for p in &f.programs {
            out.push(Prog { hash, fssm: fi, offset: p.offset, code: f.data[p.offset..p.offset + p.len].to_vec(), mode: mode.clone() });
        }
    }
}

/// One program: (portable, local).
fn explain(p: &Prog, split: bool, o: &Opts) -> (Value, Value) {
    let wave64 = recomp::detect_wave64(&p.code);
    let mut portable = Map::new();
    let mut local = Map::new();
    portable.insert("guest".into(), guest(&p.code, wave64));
    if o.hlsl {
        let (pv, lv) = hlsl::path(p, wave64, split, o);
        portable.insert("hlsl".into(), pv);
        local.insert("hlsl".into(), lv);
    }
    if o.direct {
        let (pv, lv) = direct::path(p, wave64, split, o);
        portable.insert("direct".into(), pv);
        local.insert("direct".into(), lv);
    }
    (Value::Object(portable), Value::Object(local))
}

/// Instruction class of an RDNA2 mnemonic (lower case).
fn class_of(m: &str) -> &'static str {
    let starts = |p: &[&str]| p.iter().any(|x| m.starts_with(x));
    if m == "s_branch" || starts(&["s_cbranch", "s_setpc", "s_swappc", "s_getpc", "s_endpgm"]) {
        "branch"
    } else if starts(&["s_load", "s_buffer_load", "s_store", "s_buffer_store", "s_atomic", "s_buffer_atomic", "s_dcache", "s_gl1_inv"]) {
        "smem"
    } else if m.starts_with("s_") {
        "salu"
    } else if starts(&["v_readlane", "v_readfirstlane", "v_writelane", "v_permlane"]) {
        "cross_lane"
    } else if m.starts_with("v_") {
        "valu"
    } else if starts(&["buffer_", "tbuffer_", "global_", "flat_", "scratch_"]) {
        "vmem"
    } else if m.starts_with("ds_") {
        "lds"
    } else if m.starts_with("image_") {
        "image"
    } else if m.starts_with("exp") {
        "export"
    } else {
        "other"
    }
}

/// The console program itself: size, instruction classes and control flow.
fn guest(code: &[u8], wave64: bool) -> Value {
    let mut st = xbox::InstStats::default();
    st.add_code(code);
    let mut classes: BTreeMap<&str, u64> = BTreeMap::new();
    for (m, n) in &st.by_mnemonic {
        *classes.entry(class_of(m)).or_default() += n;
    }
    let mut v = json!({
        "bytes": code.len(),
        "instructions": st.instructions,
        "undecoded": st.undecoded,
        "wave64": wave64,
        "classes": classes,
    });
    match Cfg::build(code) {
        Ok(c) => {
            let pos: HashMap<usize, usize> = c.rpo.iter().enumerate().map(|(i, &b)| (b, i)).collect();
            let mut back = 0u64;
            for &b in &c.rpo {
                let succ = match c.blocks[b].term {
                    Term::Jump(t) => vec![t],
                    Term::Cond(_, t, f) => vec![t, f],
                    Term::End => vec![],
                };
                back += succ.iter().filter(|t| pos.get(t).is_some_and(|&pt| pt <= pos[&b])).count() as u64;
            }
            let indirect = c.insts.iter().filter(|d| d.inst.flags & rdna2::F_INDIRECT != 0 && d.far_target.is_none()).count();
            v["blocks"] = json!(c.blocks.len());
            v["loops"] = json!(c.headers.len());
            v["back_edges"] = json!(back);
            v["indirect_jumps"] = json!(indirect);
        }
        Err(e) => v["cfg_error"] = json!(e),
    }
    v
}

fn dxil_stats(container: &[u8]) -> Value {
    match stats::container_stats(container) {
        Ok(s) => s.to_json(),
        Err(e) => json!({ "bytes": container.len(), "read_error": e }),
    }
}

/// (portable, local) fingerprint: what the portable numbers depend on, and the machine.
fn fingerprint(o: &Opts, split: bool, split_auto: bool) -> (Value, Value) {
    let _ = dxc::location(); // loads DXC, so the toolchain has its versions
    let tc = dxc::toolchain();
    let (commit, dirty) = git_state();
    let portable = json!({
        "format": FORMAT,
        "xcl": env!("CARGO_PKG_VERSION"),
        "commit": commit,
        "dirty": dirty,
        "dxcompiler": tc.dxcompiler.to_string(),
        "dxil": tc.dxil.to_string(),
        "options": {
            "windows": o.windows,
            "heap_windows": o.heap && o.windows > 1,
            "split_wave64": split,
            "split_wave64_from": if split_auto { "this GPU" } else { "--split" },
            "dxc_level": o.opt.clone().unwrap_or_else(|| "auto".into()),
            "paths": match (o.hlsl, o.direct) { (true, true) => "both", (true, false) => "hlsl", _ => "direct" },
        },
    });
    let cpu = cpu_name();
    let gpu = recomp::runner::adapter().unwrap_or_else(|e| format!("none ({e})"));
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    let exe = std::env::current_exe().ok().and_then(|p| std::fs::read(p).ok()).map(|b| format!("{:016x}", shader::fnv64(&b)));
    let local = json!({
        "machine": format!("{:016x}", fnv64_str(&format!("{cpu}|{threads}|{gpu}"))),
        "cpu": cpu,
        "threads": threads,
        "gpu": gpu,
        "build": exe,
    });
    (portable, local)
}

/// The commit of the checkout this xcl.exe was built in (found above the exe, as
/// `scripts/smoke.ps1` does it) and whether the crates have uncommitted changes.
fn git_state() -> (Option<String>, Option<bool>) {
    let Some(root) = std::env::current_exe().ok().and_then(|p| p.ancestors().find(|a| a.join(".git").exists()).map(|a| a.to_path_buf())) else {
        return (None, None);
    };
    let git = |args: &[&str]| std::process::Command::new("git").arg("-C").arg(&root).args(args).output().ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    let commit = git(&["rev-parse", "HEAD"]);
    let dirty = git(&["status", "--porcelain", "--untracked-files=no", "--", "crates", "Cargo.toml", "Cargo.lock"]).map(|s| !s.is_empty());
    (commit, dirty)
}

fn cpu_name() -> String {
    #[allow(unused_unsafe)]
    let words: Vec<u32> = (0x8000_0002u32..=0x8000_0004)
        .flat_map(|leaf| {
            let r = unsafe { std::arch::x86_64::__cpuid(leaf) };
            [r.eax, r.ebx, r.ecx, r.edx]
        })
        .collect();
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    String::from_utf8_lossy(&bytes).trim_matches(|c: char| c == '\0' || c.is_whitespace()).to_string()
}

fn ms(d: Duration) -> u64 {
    d.as_millis() as u64
}

fn fnv64_str(s: &str) -> u64 {
    shader::fnv64(s.as_bytes())
}

fn print_fingerprint(p: &Value, l: &Value) {
    let s = |v: &Value| match v {
        Value::String(s) => s.clone(),
        Value::Null => "unknown".into(),
        v => v.to_string(),
    };
    println!("{FORMAT}");
    println!("  portable: xcl {} commit {}{} · dxcompiler {} · dxil {}", s(&p["xcl"]), s(&p["commit"]), if p["dirty"] == json!(true) { " (uncommitted changes)" } else { "" }, s(&p["dxcompiler"]), s(&p["dxil"]));
    println!("            options {}", p["options"]);
    println!("  local:    machine {} · {} · {} threads · {} · build {}", s(&l["machine"]), s(&l["cpu"]), s(&l["threads"]), s(&l["gpu"]), s(&l["build"]));
}

fn kb(v: &Value) -> String {
    let b = v.as_f64().unwrap_or(0.0);
    if b >= 1048576.0 { format!("{:.1} MB", b / 1048576.0) } else { format!("{:.1} KB", b / 1024.0) }
}

/// The shares of a {name: bytes} map, largest first.
fn shares(m: &Value, total: f64) -> String {
    let Some(m) = m.as_object() else { return String::new() };
    let mut v: Vec<(&String, f64)> = m.iter().map(|(k, b)| (k, b.as_f64().unwrap_or(0.0))).collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    v.iter().map(|(k, b)| format!("{k} {:.0}%", b * 100.0 / total.max(1.0))).collect::<Vec<_>>().join(" · ")
}

fn dxil_line(d: &Value) -> String {
    if let Some(e) = d.get("read_error") {
        return format!("{} (not read: {e})", kb(&d["bytes"]));
    }
    format!(
        "{} bitcode · {} blocks · {} instructions · {} values · {} phis ({} incoming, max {}) · {} selects · {} back branches · {} calls",
        kb(&d["bitcode_bytes"]), d["blocks"], d["instructions"], d["values"], d["phis"], d["phi_incoming"], d["max_phi_incoming"], d["selects"], d["backward_branches"], d["calls"]
    )
}

fn print_program(p: &Prog, pv: &Value, lv: &Value, o: &Opts) {
    let g = &pv["guest"];
    println!();
    println!("program {}  {}  {}", p.id(), coverage::stage_name(&p.mode), if g["wave64"] == json!(true) { "wave64" } else { "wave32" });
    let total = g["instructions"].as_f64().unwrap_or(0.0);
    print!("  guest    {} instructions · {} B", g["instructions"], g["bytes"]);
    if g.get("blocks").is_some() {
        print!(" · {} blocks · {} loops · {} back edges · {} indirect jumps", g["blocks"], g["loops"], g["back_edges"], g["indirect_jumps"]);
    } else {
        print!(" · CFG: {}", g["cfg_error"]);
    }
    println!();
    println!("           {}", shares(&g["classes"], total));
    if let Some(h) = pv.get("hlsl") {
        hlsl::print(h, &lv["hlsl"], o.top);
    }
    if let Some(d) = pv.get("direct") {
        direct::print(d, &lv["direct"]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::xbox::tests::{fssm, sample, CODE};

    /// A synthetic console compute container (as in `shaders::tests`).
    fn compute_sample() -> Vec<u8> {
        let mut compute = fssm(&CODE, &[(6, 3)]);
        for offset in [0x13C, 0x140, 0x144] {
            compute[offset..offset + 4].copy_from_slice(&1u32.to_le_bytes());
        }
        sample(5, &[compute])
    }

    struct Temp(std::path::PathBuf);
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn instruction_classes() {
        for (m, c) in [
            ("s_cbranch_execz", "branch"),
            ("s_setpc_b64", "branch"),
            ("s_buffer_load_dwordx4", "smem"),
            ("s_and_b64", "salu"),
            ("v_readfirstlane_b32", "cross_lane"),
            ("v_fma_f32", "valu"),
            ("buffer_load_dword", "vmem"),
            ("global_store_dword", "vmem"),
            ("ds_read_b32", "lds"),
            ("image_sample", "image"),
            ("exp", "export"),
        ] {
            assert_eq!(class_of(m), c, "{m}");
        }
    }

    /// Selection by hash, and the same program explained twice gives the same portable numbers.
    #[test]
    fn explain_is_deterministic_and_selects_by_hash() {
        let root = Temp(std::env::temp_dir().join(format!("xcl-explain-test-{}", std::process::id())));
        let _ = std::fs::remove_dir_all(&root.0);
        std::fs::create_dir_all(&root.0).unwrap();
        let blob = compute_sample();
        let hash = shader::fnv64(&blob);
        let path = root.0.join("cs.bin");
        std::fs::write(&path, &blob).unwrap();
        let input = Input::Files(vec![path]);

        let progs = select(&input, &args(&["--hash", &format!("{hash:016x}")])).unwrap();
        assert_eq!(progs.len(), 1);
        assert_eq!(progs[0].id(), format!("{hash:016x}:f0:{:x}", progs[0].offset));
        assert!(select(&input, &args(&["--hash", "1"])).err().unwrap().contains("0000000000000001"));
        assert_eq!(select(&input, &args(&["--sample", "5"])).unwrap().len(), 1);

        let o = Opts { windows: 4, heap: false, split: Some(false), opt: None, hlsl: true, direct: true, timeout: Duration::from_secs(60), mem_bytes: 8 << 30, dxc_timeout: Duration::from_secs(600), top: 4 };
        let (p1, l1) = explain(&progs[0], false, &o);
        let (p2, _) = explain(&progs[0], false, &o);
        assert_eq!(p1, p2);
        assert!(p1["guest"]["instructions"].as_u64().unwrap() > 0, "{p1}");
        assert!(p1["direct"]["dxil"]["blocks"].as_u64().unwrap() > 0, "{p1}");
        assert!(l1["direct"]["translate_ms"].is_u64(), "{l1}");
        assert!(p1["hlsl"]["hlsl_bytes"].as_u64().unwrap() > 0, "{p1}");
        assert!(l1["hlsl"]["translate_ms"].is_u64(), "{l1}");
        if dxc::create_instance().is_none() {
            return recomp::runner::skip_test("no DXC");
        }
        assert!(p1["hlsl"].get("dxil").is_some(), "{p1}");
    }
}
