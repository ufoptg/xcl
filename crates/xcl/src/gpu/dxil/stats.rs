//! Structure statistics of a DXIL module (blocks, phis, branches, calls by
//! callee, …), read from the bitcode DXC emits: the numbers that explain why
//! a shader is expensive for the driver's compiler.
//!
//! One pass over the bitstream ([`super::bitreader`]). Value ids are tracked
//! exactly (globals, functions, constants, arguments, value-producing
//! instructions) so that relative operands — the callee of a call — resolve.
//! Metadata, attributes and function symbol tables are skipped unread.

use std::collections::BTreeMap;

use super::bitreader::{unwrap_bitcode, BitReader, Entry};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DxilStats {
    /// The whole input.
    pub container_bytes: u64,
    /// The LLVM bitcode inside the `DXIL` part.
    pub bitcode_bytes: u64,
    /// Functions with bodies.
    pub functions: u32,
    /// Declared-only functions (`dx.op.*` and the like).
    pub declarations: u32,
    /// Basic blocks over all function bodies.
    pub blocks: u32,
    pub instructions: u32,
    /// Instructions that produce a value (SSA values).
    pub values: u32,
    pub phis: u32,
    /// Incoming edges over all phis.
    pub phi_incoming: u64,
    pub max_phi_incoming: u32,
    /// Unconditional `br`.
    pub branches: u32,
    pub cond_branches: u32,
    pub switches: u32,
    pub switch_cases: u64,
    /// `br` / `switch` edges to a block at or before their own (loop back
    /// edges in DXC's block layout).
    pub backward_branches: u32,
    pub selects: u32,
    pub loads: u32,
    pub stores: u32,
    pub allocas: u32,
    pub calls: u32,
    /// Constants (module and function constant blocks).
    pub constants: u32,
    pub globals: u32,
    /// Instruction kind (`binop`, `phi`, `call`, …) → count.
    pub opcodes: BTreeMap<&'static str, u32>,
    /// Called function name (`dx.op.bufferLoad.i32`) → calls.
    pub callees: BTreeMap<String, u32>,
}

impl DxilStats {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "container_bytes": self.container_bytes,
            "bitcode_bytes": self.bitcode_bytes,
            "functions": self.functions,
            "declarations": self.declarations,
            "blocks": self.blocks,
            "instructions": self.instructions,
            "values": self.values,
            "phis": self.phis,
            "phi_incoming": self.phi_incoming,
            "max_phi_incoming": self.max_phi_incoming,
            "branches": self.branches,
            "cond_branches": self.cond_branches,
            "switches": self.switches,
            "switch_cases": self.switch_cases,
            "backward_branches": self.backward_branches,
            "selects": self.selects,
            "loads": self.loads,
            "stores": self.stores,
            "allocas": self.allocas,
            "calls": self.calls,
            "constants": self.constants,
            "globals": self.globals,
            "opcodes": self.opcodes,
            "callees": self.callees,
        })
    }
}

/// Statistics of the program in a DXBC container's `DXIL` part.
pub fn container_stats(container: &[u8]) -> Result<DxilStats, String> {
    let mut s = module_stats(dxil_bitcode(container)?)?;
    s.container_bytes = container.len() as u64;
    Ok(s)
}

/// Statistics of raw bitcode (`BC\xC0\xDE`, optionally in a `0x0B17C0DE` wrapper).
pub fn bitcode_stats(bitcode: &[u8]) -> Result<DxilStats, String> {
    let mut s = module_stats(unwrap_bitcode(bitcode)?)?;
    s.container_bytes = bitcode.len() as u64;
    Ok(s)
}

/// The bitcode of a container's `DXIL` part (program header, bitcode header, bitcode).
pub fn dxil_bitcode(c: &[u8]) -> Result<&[u8], String> {
    let at = |b: &[u8], o: usize| {
        let word = b.get(o..o + 4).ok_or_else(|| "truncated DXBC container".to_string())?;
        Ok::<_, String>(u32::from_le_bytes(word.try_into().unwrap()) as usize)
    };
    if c.get(..4) != Some(b"DXBC") {
        return Err("not a DXBC container".into());
    }
    for i in 0..at(c, 28)? {
        let off = at(c, 32 + 4 * i)?;
        if c.get(off..off + 4) == Some(b"DXIL") {
            let part = c.get(off + 8..off + 8 + at(c, off + 4)?).ok_or("DXIL part runs past the container")?;
            if part.get(8..12) != Some(b"DXIL") {
                return Err("DXIL part without a bitcode header".into());
            }
            let (start, len) = (8 + at(part, 16)?, at(part, 20)?);
            return part.get(start..start + len).ok_or_else(|| "bitcode runs past the DXIL part".into());
        }
    }
    Err("no DXIL part in the container".into())
}

// LLVMBitCodes.h (3.7)
const BLOCK_MODULE: u32 = 8;
const BLOCK_CONSTANTS: u32 = 11;
const BLOCK_FUNCTION: u32 = 12;
const BLOCK_VALUE_SYMTAB: u32 = 14;
const BLOCK_TYPE: u32 = 17;

#[derive(Debug, Clone, Copy)]
enum Type {
    Void,
    Func { ret: u32, params: u32 },
    Ptr(u32),
    Other,
}

#[derive(Default)]
struct Walk {
    s: DxilStats,
    types: Vec<Type>,
    /// Module-level values in id order: a function's type, `None` for the rest.
    values: Vec<Option<u32>>,
    /// Value ids of the functions with bodies, in body order.
    bodies: Vec<u32>,
    bodies_seen: usize,
    relative: bool,
    names: BTreeMap<u32, String>,
    callee_ids: BTreeMap<u32, u32>,
    next_value: u32,
    cur_block: u32,
}

fn u32_of(v: u64) -> Result<u32, String> {
    u32::try_from(v).map_err(|_| format!("operand {v} over 32 bits"))
}

fn module_stats(bc: &[u8]) -> Result<DxilStats, String> {
    let mut r = BitReader::new(bc)?;
    let mut w = Walk::default();
    w.s.bitcode_bytes = bc.len() as u64;
    let mut ctx: Vec<u32> = Vec::new();
    let mut modules = 0;
    while let Some(e) = r.next_entry()? {
        match e {
            Entry::Enter(id) => {
                let keep = match (ctx.as_slice(), id) {
                    ([], BLOCK_MODULE) => {
                        modules += 1;
                        true
                    }
                    ([BLOCK_MODULE], BLOCK_TYPE | BLOCK_CONSTANTS | BLOCK_VALUE_SYMTAB) => true,
                    ([BLOCK_MODULE], BLOCK_FUNCTION) => {
                        w.enter_function()?;
                        true
                    }
                    ([BLOCK_MODULE, BLOCK_FUNCTION], BLOCK_CONSTANTS) => true,
                    _ => false,
                };
                if keep {
                    ctx.push(id);
                } else {
                    r.skip_block()?;
                }
            }
            Entry::End(_) => {
                ctx.pop();
            }
            Entry::Record(code) => {
                let ops = r.ops();
                match ctx.as_slice() {
                    [BLOCK_MODULE] => w.module_record(code, ops)?,
                    [BLOCK_MODULE, BLOCK_TYPE] => w.type_record(code, ops)?,
                    [BLOCK_MODULE, BLOCK_VALUE_SYMTAB] => w.symbol(code, ops),
                    [BLOCK_MODULE, BLOCK_FUNCTION] => w.inst(code, ops)?,
                    [BLOCK_MODULE, .., BLOCK_CONSTANTS] if code != 1 => {
                        // Every record but SETTYPE defines one constant.
                        w.s.constants += 1;
                        if ctx.len() == 2 {
                            w.values.push(None);
                        } else {
                            w.next_value = w.next_value.wrapping_add(1);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    if !ctx.is_empty() || modules != 1 {
        return Err(format!("bitcode with {modules} module blocks"));
    }
    for (id, n) in w.callee_ids {
        let name = w.names.get(&id).cloned().unwrap_or_else(|| format!("#{id}"));
        *w.s.callees.entry(name).or_default() += n;
    }
    Ok(w.s)
}

impl Walk {
    /// The function type behind a type id (through a pointer): (return type, parameters).
    fn fn_type(&self, t: u64) -> Result<(u32, u32), String> {
        let ty = |t: u64| self.types.get(t as usize).copied().ok_or_else(|| format!("type {t} out of range"));
        let resolved = match ty(t)? {
            Type::Ptr(p) => ty(p as u64)?,
            other => other,
        };
        match resolved {
            Type::Func { ret, params } if (ret as usize) < self.types.len() => Ok((ret, params)),
            _ => Err(format!("type {t} is not a function type")),
        }
    }

    fn type_record(&mut self, code: u32, ops: &[u64]) -> Result<(), String> {
        let t = match (code, ops) {
            (1 | 19, _) => return Ok(()), // NUMENTRY, STRUCT_NAME
            (2, _) => Type::Void,
            (8, [p, ..]) => Type::Ptr(u32_of(*p)?),
            (21, [_, ret, params @ ..]) => Type::Func { ret: u32_of(*ret)?, params: params.len() as u32 },
            (9, [_, _, ret, params @ ..]) => Type::Func { ret: u32_of(*ret)?, params: params.len() as u32 },
            (8 | 9 | 21, _) => return Err(format!("type record {code} with {} operands", ops.len())),
            _ => Type::Other,
        };
        self.types.push(t);
        Ok(())
    }

    fn module_record(&mut self, code: u32, ops: &[u64]) -> Result<(), String> {
        match code {
            1 => self.relative = ops.first().is_some_and(|&v| v >= 1), // VERSION
            7 => {
                self.s.globals += 1;
                self.values.push(None);
            }
            8 => {
                // [type, cc, isproto, ...]
                let [ty, _, proto, ..] = *ops else { return Err("short FUNCTION record".into()) };
                self.fn_type(ty)?;
                let ty = match self.types[ty as usize] {
                    Type::Ptr(p) => p,
                    _ => ty as u32,
                };
                if proto == 0 {
                    self.s.functions += 1;
                    self.bodies.push(self.values.len() as u32);
                } else {
                    self.s.declarations += 1;
                }
                self.values.push(Some(ty));
            }
            9 | 14 => self.values.push(None), // aliases
            _ => {}
        }
        Ok(())
    }

    /// Module symbol table: ENTRY [valueid, name...], FNENTRY [valueid, offset, name...].
    fn symbol(&mut self, code: u32, ops: &[u64]) {
        let (id, name) = match (code, ops) {
            (1, [id, name @ ..]) | (3, [id, _, name @ ..]) => (*id, name),
            _ => return,
        };
        if self.values.get(id as usize).is_some_and(Option::is_some) {
            let bytes: Vec<u8> = name.iter().map(|&c| c as u8).collect();
            self.names.insert(id as u32, String::from_utf8_lossy(&bytes).into_owned());
        }
    }

    fn enter_function(&mut self) -> Result<(), String> {
        let f = *self.bodies.get(self.bodies_seen).ok_or("more function bodies than defined functions")?;
        self.bodies_seen += 1;
        let ty = self.values[f as usize].unwrap();
        let (_, params) = self.fn_type(ty as u64)?;
        self.next_value = (self.values.len() as u32).checked_add(params).ok_or("value ids over 32 bits")?;
        self.cur_block = 0;
        Ok(())
    }

    fn edge(&mut self, target: u64) {
        if target <= self.cur_block as u64 {
            self.s.backward_branches += 1;
        }
    }

    fn inst(&mut self, code: u32, ops: &[u64]) -> Result<(), String> {
        let op = |i: usize| ops.get(i).copied().ok_or_else(|| format!("instruction record {code} with {} operands", ops.len()));
        // (kind, produces a value, ends a block)
        let (kind, value, term) = match code {
            1 => {
                self.s.blocks = self.s.blocks.checked_add(u32_of(op(0)?)?).ok_or("block count over 32 bits")?;
                return Ok(());
            }
            33 | 35 => return Ok(()), // debug locations
            2 => ("binop", true, false),
            3 => ("cast", true, false),
            4 | 30 | 43 => ("gep", true, false),
            5 | 29 => {
                self.s.selects += 1;
                ("select", true, false)
            }
            6 => ("extractelt", true, false),
            7 => ("insertelt", true, false),
            8 => ("shufflevec", true, false),
            9 | 28 => ("cmp", true, false),
            10 => ("ret", false, true),
            11 => {
                // [bb] or [truebb, falsebb, cond]
                self.edge(op(0)?);
                if ops.len() >= 3 {
                    self.s.cond_branches += 1;
                    self.edge(op(1)?);
                } else {
                    self.s.branches += 1;
                }
                ("br", false, true)
            }
            12 => {
                // [opty, cond, defaultbb, (caseval, bb)*]; the case-range
                // format (magic in operand 0) is counted without its cases.
                self.s.switches += 1;
                self.edge(op(2)?);
                if op(0)? >> 16 != 0x4B5 {
                    let cases = ops.len().saturating_sub(3) / 2;
                    self.s.switch_cases += cases as u64;
                    for k in 0..cases {
                        self.edge(ops[4 + 2 * k]);
                    }
                }
                ("switch", false, true)
            }
            15 => ("unreachable", false, true),
            16 => {
                // [ty, (val, bb)*]
                let n = (ops.len().max(1) - 1) / 2;
                self.s.phis += 1;
                self.s.phi_incoming += n as u64;
                self.s.max_phi_incoming = self.s.max_phi_incoming.max(n as u32);
                ("phi", true, false)
            }
            19 => {
                self.s.allocas += 1;
                ("alloca", true, false)
            }
            20 | 41 => {
                self.s.loads += 1;
                ("load", true, false)
            }
            23 => ("vaarg", true, false),
            24 | 42 | 44 | 45 => {
                self.s.stores += 1;
                ("store", false, false)
            }
            26 => ("extractval", true, false),
            27 => ("insertval", true, false),
            31 => ("indirectbr", false, true),
            34 => {
                // [paramattrs, cc, fmf if cc bit 17, fnty if cc bit 15, callee, args...]
                let cc = op(1)?;
                let mut i = 2 + (cc >> 17 & 1) as usize;
                let explicit = if cc >> 15 & 1 == 1 {
                    i += 1;
                    Some(op(i - 1)?)
                } else {
                    None
                };
                let rel = u32_of(op(i)?)?;
                let callee = if self.relative { self.next_value.wrapping_sub(rel) } else { rel };
                let fty = match explicit {
                    Some(t) => t,
                    None => self.values.get(callee as usize).copied().flatten().ok_or("call through a value of unknown type")? as u64,
                };
                let (ret, _) = self.fn_type(fty)?;
                self.s.calls += 1;
                *self.callee_ids.entry(callee).or_default() += 1;
                ("call", !matches!(self.types[ret as usize], Type::Void), false)
            }
            36 => ("fence", false, false),
            37 | 46 => ("cmpxchg", true, false),
            38 => ("atomicrmw", true, false),
            39 => ("resume", false, true),
            40 | 47 => ("landingpad", true, false),
            _ => return Err(format!("unsupported instruction record {code}")),
        };
        self.s.instructions += 1;
        *self.s.opcodes.entry(kind).or_default() += 1;
        if value {
            self.s.values += 1;
            self.next_value = self.next_value.wrapping_add(1);
        }
        if term {
            self.cur_block += 1;
        }
        Ok(())
    }
}
