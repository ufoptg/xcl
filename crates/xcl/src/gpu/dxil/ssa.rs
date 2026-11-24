//! On-the-fly SSA construction (Braun, Buchwald, Hack, Leißa, Mallon, Zwinkau:
//! "Simple and Efficient Construction of Static Single Assignment Form", CC 2013)
//! over the module's basic blocks. Variables are register dwords (SGPRs, SCC,
//! VCC, M0, EXEC, VGPRs) numbered by the translator.
//!
//! Blocks are registered as they are created; predecessors are added as the
//! branches are emitted. A block whose predecessors are not all known yet is
//! *unsealed*: reads there create incomplete phis, completed on [`Ssa::seal`].
//! A read that reaches a block without predecessors asks the caller for the
//! variable's initial value (the kernel loads it from `io` in the entry block).

use std::collections::HashMap;

use super::module::{Block, Ty, Value};
use super::shader::Shader;

/// Multiplicative hash for the small integer keys of the definition maps (SipHash
/// dominated the reads of long programs).
#[derive(Default, Clone, Copy)]
struct VarHasher(u64);

impl std::hash::Hasher for VarHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
        }
    }
    fn write_u32(&mut self, v: u32) {
        self.0 = (self.0.rotate_left(5) ^ v as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

type VarMap = HashMap<u32, Value, std::hash::BuildHasherDefault<VarHasher>>;

#[derive(Default)]
struct SBlock {
    defs: VarMap,
    preds: Vec<Block>,
    sealed: bool,
    /// predecessors to wait for before sealing automatically (None: sealed explicitly)
    expect: Option<usize>,
    incomplete: Vec<(u32, Value)>,
    /// a region join ([`Ssa::seal_region_join`]): the region's last block and the
    /// variables it writes
    region: Option<(Block, Vec<u32>)>,
}

/// A sealed join waiting for the values from its predecessors.
struct ReadJoin {
    block: Block,
    preds: Vec<Block>,
    values: Vec<Value>,
    chain_start: usize,
}

pub struct Ssa {
    blocks: Vec<SBlock>,
    ty: Ty,
    /// (block, variable) of the sealed joins whose operands are being read, with the
    /// phi created if a read came back around to the join (a loop)
    visiting: HashMap<(u32, u32), Option<Value>>,
    /// open write logs (innermost last)
    logs: Vec<Vec<u32>>,
}

impl Ssa {
    /// All variables have type `ty`.
    pub fn new(ty: Ty) -> Ssa {
        Ssa { blocks: Vec::new(), ty, visiting: HashMap::new(), logs: Vec::new() }
    }

    fn at(&mut self, b: Block) -> &mut SBlock {
        let i = b.0 as usize;
        if self.blocks.len() <= i {
            self.blocks.resize_with(i + 1, SBlock::default);
        }
        &mut self.blocks[i]
    }

    /// `b` seals itself once `n` predecessors have been added.
    pub fn expect_preds(&mut self, s: &mut Shader, b: Block, n: usize, init: &mut dyn FnMut(&mut Shader, u32) -> Value) {
        self.at(b).expect = Some(n);
        if self.at(b).preds.len() >= n && !self.at(b).sealed {
            self.seal(s, b, init);
        }
    }

    pub fn add_pred(&mut self, s: &mut Shader, b: Block, p: Block, init: &mut dyn FnMut(&mut Shader, u32) -> Value) {
        let sb = self.at(b);
        assert!(!sb.sealed, "edge into the sealed block {b:?}");
        sb.preds.push(p);
        if let Some(n) = sb.expect {
            if sb.preds.len() >= n {
                self.seal(s, b, init);
            }
        }
    }

    pub fn is_sealed(&mut self, b: Block) -> bool {
        self.at(b).sealed
    }

    pub fn preds(&mut self, b: Block) -> Vec<Block> {
        self.at(b).preds.clone()
    }

    /// A definition by the program (logged for an open [`Ssa::begin_log`]).
    pub fn write(&mut self, b: Block, var: u32, v: Value) {
        if let Some(log) = self.logs.last_mut() {
            log.push(var);
        }
        self.at(b).defs.insert(var, v);
    }

    /// A read's result remembered in `b` (not a definition).
    fn cache(&mut self, b: Block, var: u32, v: Value) {
        self.at(b).defs.insert(var, v);
    }

    /// Starts logging the variables written (e.g. in a guarded region).
    pub fn begin_log(&mut self) {
        self.logs.push(Vec::new());
    }

    /// Ends the innermost log: the variables written since `begin_log` (also
    /// counted in the enclosing log).
    pub fn end_log(&mut self) -> Vec<u32> {
        let mut log = self.logs.pop().expect("begin_log");
        log.sort_unstable();
        log.dedup();
        if let Some(outer) = self.logs.last_mut() {
            outer.extend_from_slice(&log);
        }
        log
    }

    /// Seals the join `j` of a region `cond → (then … end) → j` / `cond → j` whose
    /// only writes are `written` (sorted): a written variable merges over both edges,
    /// every other one reads through `cond` directly, as if `cond` were `j`'s only
    /// predecessor. Reads then skip the region instead of walking both edges of every
    /// guard (guarded memory accesses make thousands of such joins).
    pub fn seal_region_join(&mut self, j: Block, cond: Block, end: Block, written: Vec<u32>) {
        let sb = self.at(j);
        assert!(!sb.sealed && sb.incomplete.is_empty(), "region join already read");
        sb.preds = vec![cond];
        sb.sealed = true;
        if !written.is_empty() {
            sb.region = Some((end, written));
        }
    }

    /// The predecessors a read of `var` in `b` merges over.
    fn preds_of(&mut self, b: Block, var: u32) -> Vec<Block> {
        let sb = self.at(b);
        match &sb.region {
            Some((end, w)) if w.binary_search(&var).is_ok() => vec![sb.preds[0], *end],
            _ => sb.preds.clone(),
        }
    }

    /// The definition of `var` reaching the current end of block `b`.
    pub fn read(&mut self, s: &mut Shader, b: Block, var: u32, init: &mut dyn FnMut(&mut Shader, u32) -> Value) -> Value {
        let mut block = b;
        let mut chain = Vec::new();
        let mut joins: Vec<ReadJoin> = Vec::new();
        let mut chain_start = 0;

        'read: loop {
            let mut value = loop {
                let sb = self.at(block);
                if let Some(&value) = sb.defs.get(&var) {
                    break value;
                }
                if !sb.sealed {
                    let phi = s.m.phi_at(block, self.ty);
                    self.at(block).incomplete.push((var, phi));
                    self.cache(block, var, phi);
                    break phi;
                }
                if sb.preds.len() == 1 && !sb.region.as_ref().is_some_and(|(_, w)| w.binary_search(&var).is_ok()) {
                    chain.push(block);
                    block = sb.preds[0];
                    continue;
                }
                if sb.preds.is_empty() {
                    let value = init(s, var);
                    self.cache(block, var, value);
                    break value;
                }

                let key = (block.0, var);
                if let Some(slot) = self.visiting.get(&key).copied() {
                    let phi = slot.unwrap_or_else(|| s.m.detached_phi(self.ty));
                    self.visiting.insert(key, Some(phi));
                    break phi;
                }
                self.visiting.insert(key, None);
                let preds = self.preds_of(block, var);
                let next = preds[0];
                let values = Vec::with_capacity(preds.len());
                joins.push(ReadJoin { block, preds, values, chain_start });
                // Each operand owns the chain suffix it traverses.
                chain_start = chain.len();
                block = next;
                continue 'read;
            };

            loop {
                for traversed in chain.drain(chain_start..) {
                    self.cache(traversed, var, value);
                }
                let Some(join) = joins.last_mut() else {
                    return value;
                };
                join.values.push(value);
                if let Some(&next) = join.preds.get(join.values.len()) {
                    block = next;
                    continue 'read;
                }

                let join = joins.pop().unwrap();
                let marker = self.visiting.remove(&(join.block.0, var)).flatten();
                // Identical operands need no phi unless a loop has already used it.
                value = if marker.is_none() && join.values.iter().all(|&v| v == join.values[0]) {
                    join.values[0]
                } else {
                    let phi = marker.unwrap_or_else(|| s.m.detached_phi(self.ty));
                    for (&v, &p) in join.values.iter().zip(&join.preds) {
                        s.m.add_incoming(phi, v, p);
                    }
                    s.m.place_phi(join.block, phi);
                    phi
                };
                self.cache(join.block, var, value);
                chain_start = join.chain_start;
            }
        }
    }

    fn add_operands(&mut self, s: &mut Shader, phi: Value, preds: &[Block], var: u32, init: &mut dyn FnMut(&mut Shader, u32) -> Value) {
        for &p in preds {
            let v = self.read(s, p, var, init);
            s.m.add_incoming(phi, v, p);
        }
    }

    pub fn seal(&mut self, s: &mut Shader, b: Block, init: &mut dyn FnMut(&mut Shader, u32) -> Value) {
        if self.at(b).sealed {
            return;
        }
        let preds = self.at(b).preds.clone();
        let inc = std::mem::take(&mut self.at(b).incomplete);
        for (var, phi) in inc {
            self.add_operands(s, phi, &preds, var, init);
        }
        self.at(b).sealed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::dxil::container::Stage;

    #[test]
    fn deep_joins_read_on_a_small_stack() {
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let mut shader = Shader::new(Stage::Compute, (6, 6));
                let mut ssa = Ssa::new(shader.m.i32());
                let mut init = |shader: &mut Shader, _| shader.m.i32c(42);
                let mut previous = shader.m.new_block();
                ssa.seal(&mut shader, previous, &mut init);
                for _ in 0..8192 {
                    let branches = [shader.m.new_block(), shader.m.new_block()];
                    let join = shader.m.new_block();
                    for branch in branches {
                        ssa.add_pred(&mut shader, branch, previous, &mut init);
                        ssa.seal(&mut shader, branch, &mut init);
                        ssa.add_pred(&mut shader, join, branch, &mut init);
                    }
                    ssa.seal(&mut shader, join, &mut init);
                    previous = join;
                }

                let value = ssa.read(&mut shader, previous, 0, &mut init);
                assert_eq!(value, shader.m.i32c(42));
                assert!(shader.m.block_insts(previous).is_empty());
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
