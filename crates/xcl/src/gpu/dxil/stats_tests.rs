//! [`super::stats`] against the writer's modules (known shapes), DXC's
//! disassembly (an independent reading of the same bitcode), and corrupted input.

use super::module::{BinOp, FnAttrs, Module, Pred};
use super::stats::{bitcode_stats, container_stats, dxil_bitcode};

const LOOP_CS: &str = r"
RWByteAddressBuffer io : register(u0);
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint n = io.Load(tid.x * 4);
    uint acc = 0;
    for (uint i = 0; i < n; i++) {
        acc += io.Load(i * 4) * i;
        if (acc > 1000) break;
    }
    io.Store(tid.x * 4, acc);
}";

const BRANCH_CS: &str = r"
RWStructuredBuffer<float> b : register(u0);
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    float v = b[tid.x];
    float r;
    if (v > 0) {
        r = sqrt(v);
        if (v > 10) r = r * b[tid.x + 1];
    } else {
        r = -v * b[0];
    }
    b[tid.x] = (tid.x & 1) ? r : r * 2;
}";

const SWITCH_CS: &str = r"
Buffer<uint> src : register(t0);
RWBuffer<uint> dst : register(u0);
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint v = src[tid.x];
    uint r;
    switch (v & 7) {
        case 0: r = src[v]; break;
        case 1: r = src[v * 3] + 1; break;
        case 3: r = src[v + 1] ^ v; break;
        case 5: dst[v] = 7; r = 9; break;
        default: r = v;
    }
    dst[tid.x] = r;
}";

fn dxc(src: &str) -> Option<Vec<u8>> {
    if crate::gpu::recomp::dxc::create_instance().is_none() {
        crate::gpu::recomp::runner::skip_test("no DXC");
        return None;
    }
    Some(crate::gpu::recomp::dxc::compile(src, "main", "cs_6_6", &[]).expect("DXC compile"))
}

/// `entry: a = test(1, 2); c = a == 0; br c, b1, b2`, `b1: x = a + 1`,
/// `b2: y = select c, a, 5`, `join: p = phi [x, b1], [y, b2]`, then a loop
/// `q = phi [p, join], [q2, loop]; q2 = q + 1; sink(q2); br q2 < 10, loop, exit`.
fn branchy_module() -> Vec<u8> {
    let mut m = Module::new();
    m.define_entry("main");
    let (i32, void) = (m.i32(), m.void());
    let fty = m.func_ty(i32, &[i32, i32]);
    let test = m.declare("dx.op.test.i32", fty, FnAttrs::ReadNone);
    let sty = m.func_ty(void, &[i32]);
    let sink = m.declare("dx.op.sink", sty, FnAttrs::NoUnwind);
    let (b1, b2, join, lp, exit) = (m.new_block(), m.new_block(), m.new_block(), m.new_block(), m.new_block());
    let (one, two, zero, five, ten) = (m.i32c(1), m.i32c(2), m.i32c(0), m.i32c(5), m.i32c(10));
    let a = m.call(test, &[one, two]);
    let c = m.cmp(Pred::Eq, a, zero);
    m.cond_br(c, b1, b2);
    m.set_block(b1);
    let x = m.bin(BinOp::Add, a, one);
    m.br(join);
    m.set_block(b2);
    let y = m.select(c, a, five);
    m.br(join);
    m.set_block(join);
    let p = m.phi(i32);
    m.add_incoming(p, x, b1);
    m.add_incoming(p, y, b2);
    m.br(lp);
    m.set_block(lp);
    let q = m.phi(i32);
    let q2 = m.bin(BinOp::Add, q, one);
    m.add_incoming(q, p, join);
    m.add_incoming(q, q2, lp);
    m.call(sink, &[q2]);
    let more = m.cmp(Pred::ULt, q2, ten);
    m.cond_br(more, lp, exit);
    m.set_block(exit);
    m.ret();
    m.to_bitcode("dxil-ms-dx", "")
}

#[test]
fn writer_module_stats_are_exact() {
    let bc = branchy_module();
    let s = bitcode_stats(&bc).unwrap();
    let ops = [("binop", 2), ("br", 5), ("call", 2), ("cmp", 2), ("phi", 2), ("ret", 1), ("select", 1)];
    let want = super::stats::DxilStats {
        container_bytes: bc.len() as u64,
        bitcode_bytes: bc.len() as u64,
        functions: 1,
        declarations: 2,
        blocks: 6,
        instructions: 15,
        values: 8, // the call to sink is void
        phis: 2,
        phi_incoming: 4,
        max_phi_incoming: 2,
        branches: 3,
        cond_branches: 2,
        backward_branches: 1,
        selects: 1,
        calls: 2,
        constants: 5,
        opcodes: ops.into_iter().collect(),
        callees: [("dx.op.sink".to_string(), 1), ("dx.op.test.i32".to_string(), 1)].into_iter().collect(),
        ..Default::default()
    };
    assert_eq!(s, want);
    assert_eq!(s.to_json()["callees"]["dx.op.sink"], 1);
}

#[test]
fn writer_switch_and_memory_stats_are_exact() {
    let mut m = Module::new();
    m.define_entry("main");
    let i32 = m.i32();
    let arr = m.array(4, i32);
    let g = m.global("g", arr, 3, None, false, 4);
    let (c1, c2, c3) = (m.i32c(1), m.i32c(2), m.i32c(3));
    let _ = m.alloca(i32, 4);
    let zero = m.i32c(0);
    let p = m.gep(g, &[zero, zero]);
    let a = m.load(p, 4);
    let (b1, b2, b3, exit) = (m.new_block(), m.new_block(), m.new_block(), m.new_block());
    m.switch(a, exit, &[(c1, b1), (c2, b2), (c3, b3)]);
    for (b, v) in [(b1, c1), (b2, c2), (b3, c3)] {
        m.set_block(b);
        m.store(p, v, 4);
        m.br(exit);
    }
    m.set_block(exit);
    m.ret();
    let s = bitcode_stats(&m.to_bitcode("dxil-ms-dx", "")).unwrap();
    let got = (s.functions, s.declarations, s.globals, s.blocks, s.instructions, s.values, s.constants);
    assert_eq!(got, (1, 0, 1, 5, 11, 3, 4));
    let got = (s.switches, s.switch_cases, s.branches, s.backward_branches, s.loads, s.stores, s.allocas, s.opcodes["gep"]);
    assert_eq!(got, (1, 3, 3, 0, 1, 3, 1, 1));
}

/// Counts read from DXC's disassembly of a container: an independent reading
/// of the same bitcode. Fields the text does not show stay at their default.
fn stats_from_disassembly(dis: &str) -> super::stats::DxilStats {
    let mut s = super::stats::DxilStats::default();
    let (mut in_fn, mut in_switch) = (false, false);
    // per function: block labels in order, and (source block, target label) edges
    let (mut labels, mut edges): (Vec<String>, Vec<(usize, String)>) = (Vec::new(), Vec::new());
    for line in dis.lines() {
        let t = line.split("  ;").next().unwrap().trim();
        if t.starts_with("declare ") {
            s.declarations += 1;
        } else if t.starts_with("define ") {
            (in_fn, labels, edges) = (true, vec!["0".into()], Vec::new());
            s.functions += 1;
        } else if !in_fn || t.is_empty() {
        } else if t == "}" {
            in_fn = false;
            s.blocks += labels.len() as u32;
            s.backward_branches += edges.iter().filter(|(src, l)| labels.iter().position(|x| x == l).unwrap() <= *src).count() as u32;
        } else if let Some(l) = t.strip_prefix("; <label>:") {
            labels.push(l.split_whitespace().next().unwrap().to_string());
        } else if in_switch {
            if t == "]" {
                in_switch = false;
            } else {
                s.switch_cases += 1;
                edges.push((labels.len() - 1, t.rsplit('%').next().unwrap().to_string()));
            }
        } else {
            s.instructions += 1;
            let op = match t.split_once(" = ") {
                Some((_, rest)) => {
                    s.values += 1;
                    rest
                }
                None => t,
            };
            let kind = op.split_whitespace().next().unwrap();
            let targets: Vec<String> = op.split("label %").skip(1).map(|x| x.split([',', ' ']).next().unwrap().to_string()).collect();
            edges.extend(targets.iter().map(|l| (labels.len() - 1, l.clone())));
            match kind {
                "phi" => {
                    let n = op.matches("[ ").count() as u32;
                    s.phis += 1;
                    s.phi_incoming += n as u64;
                    s.max_phi_incoming = s.max_phi_incoming.max(n);
                }
                "select" => s.selects += 1,
                "br" if op.starts_with("br i1") => s.cond_branches += 1,
                "br" => s.branches += 1,
                "switch" => (s.switches, in_switch) = (s.switches + 1, true),
                "load" => s.loads += 1,
                "store" => s.stores += 1,
                "alloca" => s.allocas += 1,
                "call" => {
                    s.calls += 1;
                    let callee = op.split('@').nth(1).unwrap().split('(').next().unwrap();
                    *s.callees.entry(callee.to_string()).or_default() += 1;
                }
                _ => {}
            }
        }
    }
    s
}

#[test]
fn dxc_modules_match_their_disassembly() {
    for src in [LOOP_CS, BRANCH_CS, SWITCH_CS] {
        let Some(c) = dxc(src) else { return };
        let dis = super::validate::disassemble(&c).expect("disassembly");
        let got = container_stats(&c).unwrap_or_else(|e| panic!("{e}\n{dis}"));
        let mut want = stats_from_disassembly(&dis);
        // Not readable from the text: sizes, constants, globals, the kind histogram.
        want.container_bytes = c.len() as u64;
        want.bitcode_bytes = dxil_bitcode(&c).unwrap().len() as u64;
        (want.constants, want.globals, want.opcodes) = (got.constants, got.globals, got.opcodes.clone());
        assert_eq!(got, want, "\n{dis}");
        assert!(got.calls > 0 && got.blocks > 1, "{got:?}");
    }
    let s = container_stats(&dxc(LOOP_CS).unwrap()).unwrap();
    assert!(s.backward_branches >= 1 && s.phis >= 2, "a loop: {s:?}");
    let s = container_stats(&dxc(SWITCH_CS).unwrap()).unwrap();
    assert!(s.switches == 1 && s.switch_cases == 4, "{s:?}");
}

/// Every prefix and many byte-flipped copies parse to `Ok` or `Err`, never a
/// panic or a hang (a fixed-seed xorshift picks the flips).
fn check_corruptions(input: &[u8], parse: fn(&[u8]) -> Result<super::stats::DxilStats, String>) {
    for n in 0..input.len() {
        let _ = parse(&input[..n]);
    }
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for _ in 0..3000 {
        let mut v = input.to_vec();
        for _ in 0..1 + next() % 4 {
            let i = (next() % v.len() as u64) as usize;
            v[i] ^= 1 << (next() % 8);
        }
        let _ = parse(&v);
    }
}

#[test]
fn corrupt_input_is_an_error_not_a_panic() {
    check_corruptions(&branchy_module(), bitcode_stats);
    let Some(c) = dxc(LOOP_CS) else { return };
    check_corruptions(dxil_bitcode(&c).unwrap(), bitcode_stats);
    check_corruptions(&c, container_stats);
}
