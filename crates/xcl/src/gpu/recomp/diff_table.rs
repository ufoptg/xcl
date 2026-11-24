//! `XCL_DXIL_DIFF_TABLE=<path>` (with `XCL_DXIL_DIFF=1`): the per-opcode parity table of the
//! differential runs (`diff_kernel`). Each paired program appends one line to `<path>`:
//!
//! `<test>\t<program hash>\t<outcome>\t<opcode> <opcode> ...`
//!
//! the test is the name of the thread it ran on (libtest names a test's thread after the test;
//! `-` otherwise), the hash identifies the program (FNV-1a of its bytes), the outcome is
//! `matched`, `mismatched`, `refused` (the direct translator refused it) or `failed` (the
//! translator or the direct run itself failed), and the opcodes are the distinct RDNA2
//! mnemonics of the program, each once. A program run from many states gets one line per
//! distinct outcome. Each line is one append-mode write of the whole line, so parallel test
//! threads (and processes) do not interleave; nothing is written at exit.
//!
//! `summarise` / `table` turn the file into the table: per opcode, how many distinct programs
//! containing it matched, mismatched, were refused or failed (a program counts once, with its
//! worst outcome). With `XCL_DXIL_DIFF_SUMMARY=<path>`, the test
//! `recomp::diff_table::tests::summary_of_a_table_file` prints the table of that file and
//! writes it beside it as `<path>.summary.tsv`.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::sync::Mutex;

/// What a paired run came to, worst last.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Outcome {
    Matched,
    Refused,
    Failed,
    Mismatched,
}

impl Outcome {
    pub fn word(self) -> &'static str {
        match self {
            Outcome::Matched => "matched",
            Outcome::Refused => "refused",
            Outcome::Failed => "failed",
            Outcome::Mismatched => "mismatched",
        }
    }

    fn parse(w: &str) -> Option<Outcome> {
        [Outcome::Matched, Outcome::Refused, Outcome::Failed, Outcome::Mismatched].into_iter().find(|o| o.word() == w)
    }
}

/// FNV-1a (64 bits) of a program: the same in every process.
pub fn hash(code: &[u8]) -> u64 {
    code.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

/// The distinct mnemonics of a program, decoded from the start until the end of the code or
/// the first word that does not decode.
pub fn opcodes(code: &[u8]) -> BTreeSet<&'static str> {
    let mut ops = BTreeSet::new();
    let mut off = 0usize;
    while off + 4 <= code.len() {
        match crate::gpu::rdna2::decode(&code[off..], off as u64) {
            Ok(d) if d.len > 0 => {
                ops.insert(d.inst.name);
                off += d.len;
            }
            _ => break,
        }
    }
    ops
}

/// The table line of a program.
pub fn line(test: &str, code: &[u8], outcome: Outcome) -> String {
    let ops: Vec<&str> = opcodes(code).into_iter().collect();
    format!("{test}\t{:016x}\t{}\t{}\n", hash(code), outcome.word(), ops.join(" "))
}

/// `XCL_DXIL_DIFF_TABLE=<path>`: appends the line of `code` with `outcome`, once per program
/// and outcome in this process. Unset: nothing.
pub fn record(code: &[u8], outcome: Outcome) {
    static WRITTEN: Mutex<BTreeSet<(u64, Outcome)>> = Mutex::new(BTreeSet::new());
    let Some(path) = std::env::var_os("XCL_DXIL_DIFF_TABLE").filter(|p| !p.is_empty()) else {
        return;
    };
    if !WRITTEN.lock().unwrap_or_else(|p| p.into_inner()).insert((hash(code), outcome)) {
        return;
    }
    let test = std::thread::current().name().filter(|n| *n != "main").unwrap_or("-").replace(['\t', '\n'], " ");
    let l = line(&test, code, outcome);
    let written = std::fs::OpenOptions::new().append(true).create(true).open(&path).and_then(|mut f| f.write_all(l.as_bytes()));
    if let Err(e) = written {
        let _ = writeln!(std::io::stderr(), "XCL_DXIL_DIFF_TABLE: cannot append to {}: {e}", std::path::Path::new(&path).display());
    }
}

/// Per opcode: [matched, mismatched, refused, failed] distinct programs containing it, each
/// program (by hash) with its worst outcome over its lines. Lines that are not table lines
/// are skipped.
pub fn summarise(text: &str) -> BTreeMap<String, [usize; 4]> {
    let mut programs: BTreeMap<&str, (Outcome, BTreeSet<&str>)> = BTreeMap::new();
    for l in text.lines() {
        let f: Vec<&str> = l.split('\t').collect();
        let (Some(&h), Some(o)) = (f.get(1), f.get(2).and_then(|w| Outcome::parse(w))) else {
            continue;
        };
        let p = programs.entry(h).or_insert((o, BTreeSet::new()));
        p.0 = p.0.max(o);
        p.1.extend(f.get(3).map_or("", |s| s).split(' ').filter(|s| !s.is_empty()));
    }
    let mut table: BTreeMap<String, [usize; 4]> = BTreeMap::new();
    for (o, ops) in programs.values() {
        let col = match o {
            Outcome::Matched => 0,
            Outcome::Mismatched => 1,
            Outcome::Refused => 2,
            Outcome::Failed => 3,
        };
        for op in ops {
            table.entry(op.to_string()).or_default()[col] += 1;
        }
    }
    table
}

/// `summarise` as tab-separated text: a header, then one row per opcode, those with
/// mismatched or failed programs first (most first), then by name.
pub fn table(text: &str) -> String {
    let mut rows: Vec<(String, [usize; 4])> = summarise(text).into_iter().collect();
    rows.sort_by(|a, b| (b.1[1] + b.1[3]).cmp(&(a.1[1] + a.1[3])).then_with(|| a.0.cmp(&b.0)));
    let mut s = String::from("opcode\tmatched\tmismatched\trefused\tfailed\n");
    for (op, [m, x, r, f]) in rows {
        s.push_str(&format!("{op}\t{m}\t{x}\t{r}\t{f}\n"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_summarise_per_opcode_once_per_program_with_its_worst_outcome() {
        let text = "\
a\t01\tmatched\tS_ADD_U32 S_ENDPGM V_ADD_NC_U32
a\t01\tmismatched\tS_ADD_U32 S_ENDPGM V_ADD_NC_U32
b\t02\tmatched\tS_ENDPGM V_ADD_NC_U32
c\t03\trefused\tS_ENDPGM V_FOO
d\t04\tfailed\tS_ENDPGM
not a table line
";
        let t = summarise(text);
        assert_eq!(t["S_ADD_U32"], [0, 1, 0, 0]);
        assert_eq!(t["V_ADD_NC_U32"], [1, 1, 0, 0]);
        assert_eq!(t["S_ENDPGM"], [1, 1, 1, 1]);
        assert_eq!(t["V_FOO"], [0, 0, 1, 0]);
        assert_eq!(t.len(), 4);
        let s = table(text);
        let rows: Vec<&str> = s.lines().collect();
        assert_eq!(rows[0], "opcode\tmatched\tmismatched\trefused\tfailed");
        assert_eq!(rows[1], "S_ENDPGM\t1\t1\t1\t1");
        assert_eq!(rows[2], "S_ADD_U32\t0\t1\t0\t0");
        assert_eq!(rows[4], "V_FOO\t0\t0\t1\t0");
    }

    #[test]
    fn a_program_line_lists_its_distinct_mnemonics() {
        let ins = |name: &str, enc: &str, f: &[(&str, u64)]| crate::gpu::rdna2::encode(name, enc, f, None).unwrap();
        let code = [
            ins("S_ADD_U32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 0), ("SSRC1", 1)]),
            ins("S_ADD_U32", "ENC_SOP2", &[("SDST", 3), ("SSRC0", 0), ("SSRC1", 1)]),
            ins("S_ENDPGM", "ENC_SOPP", &[]),
        ]
        .concat();
        let l = line("t", &code, Outcome::Matched);
        assert_eq!(l, format!("t\t{:016x}\tmatched\tS_ADD_U32 S_ENDPGM\n", hash(&code)));
        assert_eq!(summarise(&l)["S_ADD_U32"], [1, 0, 0, 0]);
        assert_ne!(hash(&code), hash(&code[4..]));
    }

    /// `XCL_DXIL_DIFF_SUMMARY=<path>`: the table of that table file, printed and written to
    /// `<path>.summary.tsv`. Unset: nothing.
    #[test]
    fn summary_of_a_table_file() {
        let Some(path) = std::env::var_os("XCL_DXIL_DIFF_SUMMARY").filter(|p| !p.is_empty()) else {
            return;
        };
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", std::path::Path::new(&path).display()));
        let t = table(&text);
        let mut out = path.clone();
        out.push(".summary.tsv");
        std::fs::write(&out, &t).unwrap();
        println!("{t}");
    }
}
