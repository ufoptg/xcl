//! `xcl shaders A.json --explain-diff B.json`: what changed from one `--explain --json` report to
//! another. Portable numbers are compared always; local ones only when both reports come from
//! the same machine.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{json, Value};

use super::FORMAT;

/// Every number of a report section by its path (`hlsl.dxil.phis`); lists by index.
fn flatten(prefix: &str, v: &Value, out: &mut BTreeMap<String, f64>) {
    match v {
        Value::Object(m) => {
            for (k, x) in m {
                flatten(&if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") }, x, out);
            }
        }
        Value::Array(a) => {
            for (i, x) in a.iter().enumerate() {
                let key = x.get("pass").and_then(|p| p.as_str()).map(|p| p.to_string()).unwrap_or_else(|| i.to_string());
                flatten(&format!("{prefix}.{key}"), x, out);
            }
        }
        Value::Number(n) => {
            out.insert(prefix.to_string(), n.as_f64().unwrap_or(0.0));
        }
        Value::Bool(b) => {
            out.insert(prefix.to_string(), if *b { 1.0 } else { 0.0 });
        }
        _ => {}
    }
}

/// Changed numbers between two sections: (path, a, b), largest relative change first.
fn changes(a: &Value, b: &Value) -> Vec<(String, f64, f64)> {
    let (mut fa, mut fb) = (BTreeMap::new(), BTreeMap::new());
    flatten("", a, &mut fa);
    flatten("", b, &mut fb);
    let mut keys: Vec<&String> = fa.keys().chain(fb.keys()).collect();
    keys.sort();
    keys.dedup();
    let mut out: Vec<(String, f64, f64)> = keys
        .into_iter()
        .map(|k| (k.clone(), fa.get(k).copied().unwrap_or(f64::NAN), fb.get(k).copied().unwrap_or(f64::NAN)))
        .filter(|(_, x, y)| !(x == y || (x.is_nan() && y.is_nan())))
        .collect();
    let rel = |x: f64, y: f64| if x.is_nan() || y.is_nan() { f64::INFINITY } else { (y - x).abs() / x.abs().max(1e-9) };
    out.sort_by(|p, q| rel(q.1, q.2).total_cmp(&rel(p.1, p.2)).then_with(|| p.0.cmp(&q.0)));
    out
}

fn show(x: f64) -> String {
    if x.is_nan() { "–".into() } else if x.fract() == 0.0 { format!("{x:.0}") } else { format!("{x:.3}") }
}

fn change_line(path: &str, a: f64, b: f64) -> String {
    let pct = if a.is_nan() || b.is_nan() || a == 0.0 { String::new() } else { format!(" ({:+.1}%)", (b - a) * 100.0 / a) };
    format!("    {path}: {} → {}{pct}", show(a), show(b))
}

/// `xcl shaders A.json --explain-diff B.json`: what changed from report A to report B.
pub(in crate::shaders) fn diff(a: &Path, b: &Path) -> i32 {
    let load = |p: &Path| -> Result<Value, String> {
        let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
        let lines: Vec<Value> = text.lines().filter(|l| !l.trim().is_empty()).map(serde_json::from_str).collect::<Result<_, _>>().map_err(|e| format!("{}: {e}", p.display()))?;
        let Some(head) = lines.first().filter(|h| h["format"] == json!(FORMAT) && h["kind"] == json!("header")) else {
            return Err(format!("{}: not an {FORMAT} report (xcl shaders … --explain --json)", p.display()));
        };
        let programs: Vec<Value> = lines.iter().filter(|l| l["kind"] == json!("program")).cloned().collect();
        let complete = lines.last().is_some_and(|l| l["kind"] == json!("footer"));
        if !complete {
            println!("  note: {} has no footer (the run was stopped); {} program(s) in it", p.display(), programs.len());
        }
        Ok(json!({ "fingerprint": head["fingerprint"].clone(), "programs": programs }))
    };
    let (ra, rb) = match (load(a), load(b)) {
        (Ok(x), Ok(y)) => (x, y),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("xcl shaders --explain-diff: {e}");
            return 2;
        }
    };
    println!("{} → {}", a.display(), b.display());
    let fp = changes(&ra["fingerprint"]["portable"], &rb["fingerprint"]["portable"]);
    let fa = &ra["fingerprint"]["portable"];
    let fb = &rb["fingerprint"]["portable"];
    for k in ["commit", "dirty", "dxcompiler", "dxil"] {
        if fa[k] != fb[k] {
            println!("  {k}: {} → {}", fa[k], fb[k]);
        }
    }
    for (path, x, y) in &fp {
        println!("  {}", change_line(path, *x, *y).trim_start());
    }
    if fa["dxcompiler"] != fb["dxcompiler"] || fa["options"] != fb["options"] {
        println!("  note: DXC or the options differ too, so portable changes are not all the commit's");
    }
    let same_machine = ra["fingerprint"]["local"]["machine"] == rb["fingerprint"]["local"]["machine"];
    if !same_machine {
        println!("  local numbers: different machines ({} vs {}), not compared", ra["fingerprint"]["local"]["machine"], rb["fingerprint"]["local"]["machine"]);
    }
    let progs = |r: &Value| -> Vec<(String, Value)> { r["programs"].as_array().cloned().unwrap_or_default().into_iter().map(|p| (p["id"].as_str().unwrap_or("").to_string(), p)).collect() };
    let (pa, pb) = (progs(&ra), progs(&rb));
    let mb: BTreeMap<&str, &Value> = pb.iter().map(|(k, v)| (k.as_str(), v)).collect();
    // totals over the programs in both reports
    const TOTALS: [&str; 7] = ["guest.instructions", "hlsl.hlsl_bytes", "hlsl.dxil.bitcode_bytes", "hlsl.dxil.blocks", "hlsl.dxil.values", "direct.dxil.bitcode_bytes", "direct.dxil.blocks"];
    const LOCAL_TOTALS: [&str; 3] = ["hlsl.dxc_ms", "hlsl.dxc_peak_mb", "direct.translate_ms"];
    let (mut tot_a, mut tot_b) = (BTreeMap::<&str, f64>::new(), BTreeMap::<&str, f64>::new());
    let add = |tot: &mut BTreeMap<&'static str, f64>, keys: &[&'static str], v: &Value| {
        let mut f = BTreeMap::new();
        flatten("", v, &mut f);
        for k in keys {
            if let Some(x) = f.get(*k) {
                *tot.entry(*k).or_default() += x;
            }
        }
    };
    let mut common = 0;
    for (id, x) in &pa {
        let Some(y) = mb.get(id.as_str()) else {
            println!("\nprogram {id}: only in {}", a.display());
            continue;
        };
        common += 1;
        println!("\nprogram {id} ({})", x["stage"].as_str().unwrap_or("?"));
        let c = changes(&x["portable"], &y["portable"]);
        if c.is_empty() {
            println!("  portable: unchanged");
        } else {
            println!("  portable: {} changed", c.len());
            for (path, u, v) in c.iter().take(25) {
                println!("{}", change_line(path, *u, *v));
            }
        }
        if same_machine {
            println!("  local:");
            for k in ["hlsl.translate_ms", "hlsl.dxc_ms", "hlsl.dxc_peak_mb", "direct.translate_ms", "direct.peak_mb"] {
                let (mut u, mut v) = (BTreeMap::new(), BTreeMap::new());
                flatten("", &x["local"], &mut u);
                flatten("", &y["local"], &mut v);
                if let (Some(u), Some(v)) = (u.get(k), v.get(k)) {
                    println!("{}", change_line(k, *u, *v));
                }
            }
        }
        add(&mut tot_a, &TOTALS, &x["portable"]);
        add(&mut tot_b, &TOTALS, &y["portable"]);
        if same_machine {
            add(&mut tot_a, &LOCAL_TOTALS, &x["local"]);
            add(&mut tot_b, &LOCAL_TOTALS, &y["local"]);
        }
    }
    for (id, _) in &pb {
        if !pa.iter().any(|(k, _)| k == id) {
            println!("\nprogram {id}: only in {}", b.display());
        }
    }
    println!("\n{common} program(s) in both reports; totals over them:");
    for k in TOTALS.iter().chain(if same_machine { &LOCAL_TOTALS[..] } else { &[] }) {
        if let (Some(x), Some(y)) = (tot_a.get(k), tot_b.get(k)) {
            println!("{}", change_line(k, *x, *y));
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Temp(std::path::PathBuf);
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn changes_list_the_numbers_that_differ() {
        let a = json!({ "guest": { "instructions": 100, "wave64": true }, "hlsl": { "dxil": { "phis": 10, "blocks": 4 } } });
        let b = json!({ "guest": { "instructions": 100, "wave64": true }, "hlsl": { "dxil": { "phis": 20, "blocks": 5 } }, "direct": { "dxil": { "blocks": 3 } } });
        assert!(changes(&a, &a).is_empty());
        let c = changes(&a, &b);
        let paths: Vec<&str> = c.iter().map(|x| x.0.as_str()).collect();
        // a number only in one report first, then the largest relative change
        assert_eq!(paths, ["direct.dxil.blocks", "hlsl.dxil.phis", "hlsl.dxil.blocks"]);
        assert_eq!((c[1].1, c[1].2), (10.0, 20.0));
    }

    #[test]
    fn diff_refuses_other_files() {
        let root = Temp(std::env::temp_dir().join(format!("xcl-explain-diff-{}", std::process::id())));
        std::fs::create_dir_all(&root.0).unwrap();
        let (a, b) = (root.0.join("a.json"), root.0.join("b.json"));
        let head = json!({ "format": FORMAT, "kind": "header", "fingerprint": { "portable": {}, "local": { "machine": "1" } } });
        let prog = json!({ "kind": "program", "id": "1:f0:0", "stage": "CS", "portable": { "guest": { "instructions": 3 } }, "local": {} });
        // a report cut short (no footer) still compares
        std::fs::write(&a, format!("{head}\n{prog}\n")).unwrap();
        std::fs::write(&b, "{}").unwrap();
        assert_eq!(diff(&a, &a), 0);
        assert_eq!(diff(&a, &b), 2);
        assert_eq!(diff(&a, &root.0.join("missing.json")), 2);
    }
}
