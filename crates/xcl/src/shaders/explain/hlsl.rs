//! The HLSL path's part of `--explain`: the HLSL by the console instruction that wrote it,
//! DXC's time per pass and peak commit (the SDK's `dxc.exe`, run with the arguments a title's
//! compile uses, under a commit and time limit), and the structure of the DXIL.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::limits::{Job, Peak};
use super::{class_of, dxil_line, dxil_stats, kb, ms, shares, Opts, Prog};
use crate::gpu::dxil::coverage;
use crate::gpu::recomp::{self, dxc, Mode};
use crate::shaders::hlsl_bytes_by_op;

/// HLSL → DXC → DXIL: HLSL size by what wrote it, DXC per pass, the DXIL's structure.
pub(super) fn path(p: &Prog, wave64: bool, split: bool, o: &Opts) -> (Value, Value) {
    let opt = recomp::Options { mode: p.mode.clone(), wave64, split_wave64: wave64 && split, mem_windows: o.mem_windows(), ..Default::default() };
    let t = Instant::now();
    let tr = recomp::translate(&p.code, &opt);
    let translate_ms = ms(t.elapsed());
    let tr = match tr {
        Ok(tr) => tr,
        Err(e) => return (json!({ "error": coverage::reason_key(&e) }), json!({ "translate_ms": translate_ms })),
    };
    let mut by: HashMap<String, (usize, usize)> = HashMap::new();
    hlsl_bytes_by_op(&tr.hlsl, &mut by);
    let mut parts: BTreeMap<String, u64> = BTreeMap::new();
    let mut mnemonics: Vec<(&String, usize)> = Vec::new();
    for (k, (bytes, _)) in &by {
        let key = if k.starts_with('(') { k.trim_matches(|c| c == '(' || c == ')').to_string() } else { class_of(k).to_string() };
        *parts.entry(key).or_default() += *bytes as u64;
        if !k.starts_with('(') {
            mnemonics.push((k, *bytes));
        }
    }
    mnemonics.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    let top: BTreeMap<&str, usize> = mnemonics.iter().take(o.top).map(|(k, b)| (k.as_str(), *b)).collect();
    let profile = match p.mode {
        Mode::Vertex(_) => "vs_6_6",
        Mode::Pixel(_) => "ps_6_6",
        _ => "cs_6_6",
    };
    let extra: Vec<&str> = o.opt.iter().map(|s| s.as_str()).collect();
    // exactly the arguments the runtime's compile gives DXC
    let dxc_args = dxc::arguments(tr.hlsl.len(), "main", profile, &extra);
    let level = dxc_args.iter().rev().find(|a| a.starts_with("-O")).cloned();
    let mut portable = json!({
        "complete": tr.complete(),
        "unsupported": tr.unsupported.values().sum::<usize>(),
        "hlsl_bytes": tr.hlsl.len(),
        "hlsl_bytes_by_part": parts,
        "hlsl_bytes_top_instructions": top,
        "dxc_level": level.clone().unwrap_or_else(|| "-O3".into()),
        "dxc_arguments": dxc_args.join(" "),
    });
    let mut local = json!({ "translate_ms": translate_ms });
    let started = Instant::now();
    let r = match dxc_exe() {
        Some(exe) => compile_exe(&exe, &tr.hlsl, &dxc_args, o.mem_bytes, o.dxc_timeout).map(|(dxil, report, dxc_ms, peak_mb)| {
            local["dxc"] = json!("dxc.exe");
            local["dxc_ms"] = json!(dxc_ms);
            local["dxc_peak_mb"] = json!(peak_mb);
            local["dxc_passes"] = passes(&report, o.top);
            dxil
        }),
        None => {
            let mem = Peak::start();
            let t = Instant::now();
            let r = dxc::compile(&tr.hlsl, "main", profile, &extra);
            local["dxc"] = json!("in-process (no dxc.exe next to dxcompiler.dll: no pass times)");
            local["dxc_ms"] = json!(ms(t.elapsed()));
            local["dxc_peak_mb"] = json!(mem.finish());
            r
        }
    };
    match r {
        Ok(dxil) => portable["dxil"] = dxil_stats(&dxil),
        Err(e) if e.starts_with("timeout:") || e.starts_with("memory:") => {
            portable["dxc_error"] = json!("unfinished");
            local["dxc_unfinished"] = json!(e);
            local["dxc_ms"] = json!(ms(started.elapsed()));
        }
        Err(e) => portable["dxc_error"] = json!(e.lines().next().unwrap_or("").to_string()),
    }
    (portable, local)
}

/// `dxc.exe` of the folder XCL loads `dxcompiler.dll` from: the same compiler, run as a
/// process of its own. In XCL's process DXC prints `-ftime-report` only at exit, summed over
/// every compile; a process per compile gives each program its own report and peak commit.
fn dxc_exe() -> Option<std::path::PathBuf> {
    dxc::location().ok().map(|d| d.join("dxc.exe")).filter(|p| p.is_file())
}

/// One compile with `dxc.exe` and `args` (`dxc::arguments`) in a job of its own, with a commit
/// limit of `mem_bytes`, killed after `timeout`: (DXIL, the timing report, milliseconds, the
/// process's peak commit in MB). A compile that does not finish is an error starting with
/// `timeout:` or `memory:`.
fn compile_exe(exe: &Path, hlsl: &str, args: &[String], mem_bytes: u64, timeout: Duration) -> Result<(Vec<u8>, String, u64, u64), String> {
    use std::io::Read;
    use std::os::windows::io::AsRawHandle;
    // a folder per compile: compiles of one process never share files
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!("xcl-explain-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let (src, out) = (dir.join("program.hlsl"), dir.join("program.dxil"));
    std::fs::write(&src, hlsl).map_err(|e| format!("{}: {e}", src.display()))?;
    let _ = std::fs::remove_file(&out);
    let job = Job::new(mem_bytes).ok_or("cannot create a job object for dxc.exe")?;
    let mut child = std::process::Command::new(exe)
        .args(args)
        .arg("-ftime-report")
        .arg("-Fo")
        .arg(&out)
        .arg(&src)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("{}: {e}", exe.display()))?;
    let t = Instant::now();
    let handle = child.as_raw_handle() as usize;
    if !job.assign(handle as _) {
        let _ = child.kill();
        return Err("cannot put dxc.exe in its job".into());
    }
    // the pipes are drained on threads of their own, so a full pipe never stops DXC
    let drain = |mut r: Box<dyn Read + Send>| std::thread::spawn(move || {
        let mut s = Vec::new();
        let _ = r.read_to_end(&mut s);
        s
    });
    let (so, se) = (drain(Box::new(child.stdout.take().unwrap())), drain(Box::new(child.stderr.take().unwrap())));
    let mut peak = 0u64;
    let mut timed_out = false;
    let status = loop {
        peak = peak.max(process_peak(handle));
        match child.try_wait() {
            Ok(Some(s)) => break s,
            Ok(None) => {}
            Err(e) => return Err(format!("{}: {e}", exe.display())),
        }
        if t.elapsed() > timeout && !timed_out {
            timed_out = true;
            job.terminate();
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    peak = peak.max(process_peak(handle));
    let elapsed = ms(t.elapsed());
    let text = format!("{}{}", String::from_utf8_lossy(&so.join().unwrap_or_default()), String::from_utf8_lossy(&se.join().unwrap_or_default()));
    let dxil = std::fs::read(&out);
    let _ = std::fs::remove_dir_all(&dir);
    if timed_out {
        return Err(format!("timeout: over {} s", timeout.as_secs()));
    }
    if !status.success() {
        // a failed allocation at the job's limit ends DXC with an error or a crash
        if peak + (64 << 20) >= mem_bytes {
            return Err(format!("memory: over {} GB", mem_bytes >> 30));
        }
        return Err(format!("dxc: {}", text.lines().find(|l| l.contains("error")).unwrap_or("failed")));
    }
    Ok((dxil.map_err(|e| format!("{}: {e}", out.display()))?, text, elapsed, peak >> 20))
}

/// Peak commit (bytes) of a process so far.
fn process_peak(handle: usize) -> u64 {
    unsafe {
        use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
        let mut c: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
        c.cb = std::mem::size_of_val(&c) as u32;
        if K32GetProcessMemoryInfo(handle as _, &mut c as *mut _ as *mut _, c.cb) == 0 {
            return 0;
        }
        c.PeakPagefileUsage as u64
    }
}

/// DXC's `-ftime-report` text: wall seconds per pass (the last column before the name), summed
/// over the reports DXC prints, the slowest `top` with their share of all passes.
fn passes(report: &str, top: usize) -> Value {
    let mut by: BTreeMap<String, f64> = BTreeMap::new();
    for line in report.lines() {
        let Some(end) = line.rfind("%)") else { continue };
        let name = line[end + 2..].trim();
        if name.is_empty() || name == "Total" {
            continue;
        }
        let Some(open) = line[..end].rfind('(') else { continue };
        let Some(secs) = line[..open].split_whitespace().last().and_then(|s| s.parse::<f64>().ok()) else { continue };
        *by.entry(name.to_string()).or_default() += secs;
    }
    let total: f64 = by.values().sum();
    let mut v: Vec<(String, f64)> = by.into_iter().collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let list: Vec<Value> = v.iter().take(top).map(|(n, s)| json!({ "pass": n, "seconds": round3(*s), "share": round3(if total > 0.0 { s / total } else { 0.0 }) })).collect();
    json!({ "total_seconds": round3(total), "passes": v.len(), "top": list })
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

pub(super) fn print(h: &Value, lh: &Value, top: usize) {
    if let Some(e) = h.get("error") {
        println!("  hlsl     not translated: {e} ({} ms)", lh["translate_ms"]);
        return;
    }
    let hb = h["hlsl_bytes"].as_f64().unwrap_or(0.0);
    println!("  hlsl     {} HLSL{} · translate {} ms", kb(&h["hlsl_bytes"]), if h["complete"] == json!(false) { format!(" (incomplete: {} unsupported)", h["unsupported"]) } else { String::new() }, lh["translate_ms"]);
    println!("           by part: {}", shares(&h["hlsl_bytes_by_part"], hb));
    println!("           by instruction (top {top}): {}", shares(&h["hlsl_bytes_top_instructions"], hb));
    println!("  dxc {}   {} ms · peak {} MB ({})", h["dxc_level"].as_str().unwrap_or("?"), lh["dxc_ms"], lh["dxc_peak_mb"], lh["dxc"].as_str().unwrap_or("?"));
    if let Some(list) = lh["dxc_passes"]["top"].as_array() {
        let s: Vec<String> = list.iter().map(|x| format!("{} {:.0}%", x["pass"].as_str().unwrap_or("?"), x["share"].as_f64().unwrap_or(0.0) * 100.0)).collect();
        if !s.is_empty() {
            println!("           passes: {}", s.join(" · "));
        }
    }
    match (h.get("dxil"), h.get("dxc_error")) {
        (Some(d), _) => println!("  dxil     {}", dxil_line(d)),
        (_, Some(e)) => println!("  dxil     DXC failed: {}", lh.get("dxc_unfinished").unwrap_or(e)),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pass table of `dxc -ftime-report`: wall seconds per pass name, summed over the
    /// instances of a pass, without the `Total` line.
    #[test]
    fn time_report_passes_are_summed_by_name() {
        let report = "\
===-------------------------------------------------------------------------===
                      ... Pass execution timing report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 1.0000 seconds (1.0000 wall clock)

   ---User Time---   --System Time--   --User+System--   ---Wall Time---  --- Name ---
   0.4000 ( 40.0%)   0.0000 (  0.0%)   0.4000 ( 40.0%)   0.5000 ( 50.0%)  Combine redundant instructions
   0.3000 ( 30.0%)   0.0000 (  0.0%)   0.3000 ( 30.0%)   0.3000 ( 30.0%)  Jump Threading
   0.1000 ( 10.0%)   0.0000 (  0.0%)   0.1000 ( 10.0%)   0.2000 ( 20.0%)  Combine redundant instructions
   1.0000 (100.0%)   0.0000 (  0.0%)   1.0000 (100.0%)   1.0000 (100.0%)  Total
";
        let v = passes(report, 5);
        assert_eq!(v["passes"], json!(2));
        assert_eq!(v["total_seconds"], json!(1.0));
        assert_eq!(v["top"][0], json!({ "pass": "Combine redundant instructions", "seconds": 0.7, "share": 0.7 }));
        assert_eq!(v["top"][1]["pass"], json!("Jump Threading"));
        assert_eq!(passes("no report", 5)["passes"], json!(0));
    }

    /// A dxc.exe past its time or memory limit is stopped, and the error says which limit.
    #[test]
    fn dxc_exe_stops_at_its_limits() {
        let Some(exe) = dxc_exe() else {
            return recomp::runner::skip_test("no DXC");
        };
        let src = "RWByteAddressBuffer b : register(u0);\n[numthreads(64, 1, 1)] void main(uint i : SV_DispatchThreadID) { b.Store(4 * i, i); }\n";
        let args = dxc::arguments(src.len(), "main", "cs_6_6", &[]);
        let (dxil, report, _, _) = compile_exe(&exe, src, &args, 8 << 30, Duration::from_secs(600)).unwrap();
        assert_eq!(&dxil[..4], b"DXBC");
        // a program this small has no measurable pass ("-----"), but the report is there
        assert!(report.contains("Pass execution timing report"), "{report}");
        let t = compile_exe(&exe, src, &args, 8 << 30, Duration::ZERO).err().unwrap();
        assert!(t.starts_with("timeout:"), "{t}");
        let m = compile_exe(&exe, src, &args, 1 << 20, Duration::from_secs(600)).err().unwrap();
        assert!(m.starts_with("memory:"), "{m}");
    }
}
