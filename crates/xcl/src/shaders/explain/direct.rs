//! The direct DXIL translator's part of `--explain`: its time and memory, and the structure of
//! the DXIL it emits, or why it refused the program.

use std::time::Instant;

use serde_json::{json, Value};

use super::limits::Peak;
use super::{dxil_line, dxil_stats, ms, Opts, Prog};
use crate::gpu::dxil::{coverage, kernel::KernelOptions};

/// The direct translator (no HLSL, no DXC), given up after `--timeout` seconds (default 120).
pub(super) fn path(p: &Prog, wave64: bool, split: bool, o: &Opts) -> (Value, Value) {
    let ko = KernelOptions { wave64, split: wave64 && split, mem_windows: o.mem_windows(), sampler_base: 0 };
    let (code, mode) = (p.code.clone(), p.mode.clone());
    let (tx, rx) = std::sync::mpsc::channel();
    let mem = Peak::start();
    let t = Instant::now();
    let spawned = std::thread::Builder::new().name("xcl-explain-direct".into()).stack_size(256 << 20).spawn(move || {
        let _ = tx.send(coverage::translate_mode(&code, &mode, ko, false));
    });
    let r = match spawned {
        Ok(_) => rx.recv_timeout(o.timeout).unwrap_or_else(|_| Err(format!("timeout: over {} s", o.timeout.as_secs()))),
        Err(e) => Err(format!("thread: {e}")),
    };
    let local = json!({ "translate_ms": ms(t.elapsed()), "peak_mb": mem.finish() });
    match r {
        Ok(dxil) => (json!({ "dxil": dxil_stats(&dxil) }), local),
        // a timeout is local (the machine's speed), so it stays out of the portable numbers
        Err(e) if e.starts_with("timeout:") => {
            eprintln!("xcl shaders --explain: {}: the direct translator is still running in the background; later local times of this run are not clean", p.id());
            (json!({ "error": "timeout" }), local)
        }
        Err(e) => (json!({ "error": coverage::reason_key(&e) }), local),
    }
}

pub(super) fn print(d: &Value, ld: &Value) {
    match d.get("dxil") {
        Some(x) => println!("  direct   {} ms · peak +{} MB\n           {}", ld["translate_ms"], ld["peak_mb"], dxil_line(x)),
        None => println!("  direct   refused: {} ({} ms)", d["error"].as_str().unwrap_or("?"), ld["translate_ms"]),
    }
}
