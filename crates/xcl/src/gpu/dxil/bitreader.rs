//! LLVM bitstream reader: fixed, VBR and char6 fields, nested blocks,
//! abbreviation definitions (local and from `BLOCKINFO`), abbreviated and
//! unabbreviated records, arrays and blobs — everything DXC's LLVM 3.7 writer
//! emits. Iterative: [`BitReader::next_entry`] yields one entry at a time and the
//! block stack lives on the heap, so nesting depth costs no recursion.
//!
//! Malformed input is an `Err`, never a panic: every length field is checked
//! against the bits left in its block before anything is read or allocated.

use std::collections::BTreeMap;

/// One step of the stream. `BLOCKINFO` blocks are consumed by the reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Entry {
    /// A sub-block starts (its id); [`BitReader::skip_block`] skips it.
    Enter(u32),
    /// The current block ends (its id).
    End(u32),
    /// A record (its code); operands in [`BitReader::ops`], blob in [`BitReader::blob`].
    Record(u32),
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Literal(u64),
    Fixed(u32),
    Vbr(u32),
    Array,
    Char6,
    Blob,
}

struct Frame {
    id: u32,
    width: u32,
    /// Bit position of the block's end (the stream's end for the top level).
    end: usize,
    /// How many `BLOCKINFO` abbreviations of this block id existed on entry.
    inherited: usize,
    local: Vec<Vec<Op>>,
}

pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    len: usize,
    stack: Vec<Frame>,
    blockinfo: BTreeMap<u32, Vec<Vec<Op>>>,
    /// The block a `BLOCKINFO` `SETBID` record selected.
    info_target: Option<u32>,
    ops: Vec<u64>,
    blob: Option<(usize, usize)>,
}

const MAGIC: &[u8; 4] = b"BC\xC0\xDE";
const WRAPPER_MAGIC: u32 = 0x0B17_C0DE;
const BLOCKINFO: u32 = 0;
const BLOCKINFO_SETBID: u32 = 1;

/// The bitcode inside an optional `0x0B17C0DE` wrapper header.
pub fn unwrap_bitcode(buf: &[u8]) -> Result<&[u8], String> {
    let word = |i: usize| buf.get(i * 4..i * 4 + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize);
    if word(0) == Some(WRAPPER_MAGIC as usize) {
        let (off, size) = (word(2).ok_or("truncated bitcode wrapper")?, word(3).ok_or("truncated bitcode wrapper")?);
        return off.checked_add(size).and_then(|e| buf.get(off..e)).ok_or_else(|| "bitcode wrapper points past its buffer".to_string());
    }
    Ok(buf)
}

impl<'a> BitReader<'a> {
    /// A reader over bitcode that starts with the `BC\xC0\xDE` magic.
    pub fn new(bitcode: &'a [u8]) -> Result<Self, String> {
        if bitcode.len() < 4 || &bitcode[..4] != MAGIC {
            return Err("not LLVM bitcode (no BC 0xC0DE magic)".into());
        }
        if !bitcode.len().is_multiple_of(4) {
            return Err(format!("bitcode of {} bytes, not whole 32-bit words", bitcode.len()));
        }
        let len = bitcode.len() * 8;
        let top = Frame { id: u32::MAX, width: 2, end: len, inherited: 0, local: Vec::new() };
        Ok(BitReader { data: bitcode, pos: 32, len, stack: vec![top], blockinfo: BTreeMap::new(), info_target: None, ops: Vec::new(), blob: None })
    }

    /// The operands of the last record (without its code or blob bytes).
    pub fn ops(&self) -> &[u64] {
        &self.ops
    }

    /// The blob of the last record, if its abbreviation has one.
    pub fn blob(&self) -> Option<&'a [u8]> {
        self.blob.map(|(at, n)| &self.data[at..at + n])
    }

    fn word(&self, byte: usize) -> u64 {
        match self.data.get(byte..byte + 8) {
            Some(b) => u64::from_le_bytes(b.try_into().unwrap()),
            None => self.data.get(byte..).unwrap_or(&[]).iter().rev().fold(0, |v, &b| v << 8 | b as u64),
        }
    }

    fn fixed(&mut self, w: u32) -> Result<u64, String> {
        if w > 56 {
            let lo = self.fixed(32)?;
            return Ok(lo | self.fixed(w - 32)? << 32);
        }
        if self.len.saturating_sub(self.pos) < w as usize {
            return Err(format!("bitcode ends inside a field at bit {}", self.pos));
        }
        let v = (self.word(self.pos / 8) >> (self.pos % 8)) & ((1u64 << w) - 1);
        self.pos += w as usize;
        Ok(v)
    }

    fn vbr(&mut self, w: u32) -> Result<u64, String> {
        let hi = 1u64 << (w - 1);
        let (mut v, mut shift) = (0u64, 0u32);
        loop {
            let c = self.fixed(w)?;
            if shift >= 64 || (c & (hi - 1)) << shift >> shift != c & (hi - 1) {
                return Err(format!("VBR value over 64 bits at bit {}", self.pos));
            }
            v |= (c & (hi - 1)) << shift;
            if c & hi == 0 {
                return Ok(v);
            }
            shift += w - 1;
        }
    }

    fn align32(&mut self) {
        self.pos = self.pos.div_ceil(32) * 32;
    }

    fn left(&self) -> usize {
        self.stack.last().map_or(0, |f| f.end).saturating_sub(self.pos)
    }

    fn abbrev(&self, id: u64) -> Option<&[Op]> {
        let f = self.stack.last()?;
        let k = usize::try_from(id - 4).ok()?;
        if k < f.inherited {
            return self.blockinfo.get(&f.id).map(|v| &v[k][..]);
        }
        f.local.get(k - f.inherited).map(|v| &v[..])
    }

    /// The next entry, `None` at the end of the stream.
    pub fn next_entry(&mut self) -> Result<Option<Entry>, String> {
        loop {
            let e = match self.step()? {
                Some(e) => e,
                None => return Ok(None),
            };
            // The block holding this entry (for `Enter`, the one around the new block).
            let depth = self.stack.len() - matches!(e, Entry::Enter(_)) as usize;
            let in_info = depth >= 1 && self.stack[depth - 1].id == BLOCKINFO;
            match e {
                Entry::Enter(BLOCKINFO) => self.info_target = None,
                Entry::Enter(_) if in_info => self.skip_block()?,
                Entry::End(BLOCKINFO) => {}
                Entry::Record(c) if in_info => {
                    if c == BLOCKINFO_SETBID {
                        let id = self.ops.first().ok_or("SETBID without a block id")?;
                        self.info_target = Some(u32::try_from(*id).map_err(|_| "SETBID block id over 32 bits")?);
                    }
                }
                e => return Ok(Some(e)),
            }
        }
    }

    /// Skips the rest of the block just entered (or the current one).
    pub fn skip_block(&mut self) -> Result<(), String> {
        if self.stack.len() < 2 {
            return Err("skip_block outside a block".into());
        }
        let f = self.stack.pop().unwrap();
        self.pos = f.end;
        Ok(())
    }

    fn step(&mut self) -> Result<Option<Entry>, String> {
        loop {
            let width = self.stack.last().unwrap().width;
            let top = self.stack.len() == 1;
            if top && self.len.saturating_sub(self.pos) < 32 {
                return Ok(None);
            }
            if self.left() < width as usize {
                return Err(format!("block runs past its end at bit {}", self.pos));
            }
            let id = self.fixed(width)?;
            match id {
                0 if top => return Ok(None),
                0 => {
                    self.align32();
                    let f = self.stack.pop().unwrap();
                    if self.pos != f.end {
                        return Err(format!("block {} ends at bit {}, its length says {}", f.id, self.pos, f.end));
                    }
                    return Ok(Some(Entry::End(f.id)));
                }
                1 => {
                    let bid = u32::try_from(self.vbr(8)?).map_err(|_| "block id over 32 bits")?;
                    let w = self.vbr(4)?;
                    if w > 32 {
                        return Err(format!("abbreviation width {w} in block {bid}"));
                    }
                    self.align32();
                    let words = self.fixed(32)? as usize;
                    if words * 32 > self.left() {
                        return Err(format!("block {bid} ({words} words) runs past its parent"));
                    }
                    let inherited = self.blockinfo.get(&bid).map_or(0, Vec::len);
                    self.stack.push(Frame { id: bid, width: w as u32, end: self.pos + words * 32, inherited, local: Vec::new() });
                    return Ok(Some(Entry::Enter(bid)));
                }
                2 if !top => {
                    let a = self.define_abbrev()?;
                    let f = self.stack.last_mut().unwrap();
                    if f.id == BLOCKINFO {
                        let t = self.info_target.ok_or("BLOCKINFO abbreviation before SETBID")?;
                        self.blockinfo.entry(t).or_default().push(a);
                    } else {
                        f.local.push(a);
                    }
                }
                3 if !top => {
                    let code = self.vbr(6)?;
                    let n = self.vbr(6)? as usize;
                    if n > self.left() / 6 {
                        return Err(format!("record with {n} operands runs past its block"));
                    }
                    self.ops.clear();
                    self.blob = None;
                    for _ in 0..n {
                        let v = self.vbr(6)?;
                        self.ops.push(v);
                    }
                    return Ok(Some(Entry::Record(u32::try_from(code).map_err(|_| "record code over 32 bits")?)));
                }
                _ if !top => return self.abbreviated(id).map(Some),
                _ => return Err(format!("abbreviation id {id} at the top level")),
            }
        }
    }

    fn define_abbrev(&mut self) -> Result<Vec<Op>, String> {
        let n = self.vbr(5)? as usize;
        if n == 0 || n > self.left() / 4 {
            return Err(format!("abbreviation with {n} operands"));
        }
        let mut a = Vec::with_capacity(n);
        for _ in 0..n {
            let op = if self.fixed(1)? == 1 {
                Op::Literal(self.vbr(8)?)
            } else {
                match self.fixed(3)? {
                    e @ (1 | 2) => match (e, self.vbr(5)?) {
                        (_, 0) => Op::Literal(0),
                        (1, w) if w <= 64 => Op::Fixed(w as u32),
                        (2, w) if (2..=32).contains(&w) => Op::Vbr(w as u32),
                        (_, w) => return Err(format!("abbreviation field width {w}")),
                    },
                    3 => Op::Array,
                    4 => Op::Char6,
                    5 => Op::Blob,
                    e => return Err(format!("abbreviation encoding {e}")),
                }
            };
            a.push(op);
        }
        // An array is followed by its element (a scalar, last); a blob comes last.
        for (i, op) in a.iter().enumerate() {
            let ok = match op {
                Op::Array => i + 2 == a.len() && matches!(a[i + 1], Op::Fixed(_) | Op::Vbr(_) | Op::Char6),
                Op::Blob => i + 1 == a.len(),
                _ => true,
            };
            if !ok || (i == 0 && matches!(op, Op::Array | Op::Blob)) {
                return Err("malformed array or blob in an abbreviation".into());
            }
        }
        Ok(a)
    }

    fn scalar(&mut self, op: Op) -> Result<u64, String> {
        match op {
            Op::Literal(v) => Ok(v),
            Op::Fixed(w) => self.fixed(w),
            Op::Vbr(w) => self.vbr(w),
            Op::Char6 => Ok(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._"[self.fixed(6)? as usize] as u64),
            Op::Array | Op::Blob => Err("array or blob where a scalar belongs".into()),
        }
    }

    fn abbreviated(&mut self, id: u64) -> Result<Entry, String> {
        let a = self.abbrev(id).ok_or_else(|| format!("undefined abbreviation {id} at bit {}", self.pos))?;
        // Abbreviations are small; a copy frees `self` for reading.
        let mut buf = [Op::Literal(0); 16];
        let heap;
        let a: &[Op] = if a.len() <= buf.len() {
            buf[..a.len()].copy_from_slice(a);
            &buf[..a.len()]
        } else {
            heap = a.to_vec();
            &heap
        };
        let code = self.scalar(a[0])?;
        self.ops.clear();
        self.blob = None;
        let mut i = 1;
        while i < a.len() {
            match a[i] {
                Op::Array => {
                    let n = self.vbr(6)? as usize;
                    if n > self.left() {
                        return Err(format!("array of {n} elements runs past its block"));
                    }
                    for _ in 0..n {
                        let v = self.scalar(a[i + 1])?;
                        self.ops.push(v);
                    }
                    i += 1;
                }
                Op::Blob => {
                    let n = self.vbr(6)? as usize;
                    self.align32();
                    if n > self.left() / 8 {
                        return Err(format!("blob of {n} bytes runs past its block"));
                    }
                    self.blob = Some((self.pos / 8, n));
                    self.pos += n * 8;
                    self.align32();
                }
                op => {
                    let v = self.scalar(op)?;
                    self.ops.push(v);
                }
            }
            i += 1;
        }
        Ok(Entry::Record(u32::try_from(code).map_err(|_| "record code over 32 bits")?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::dxil::bitstream::BitWriter;

    /// A stream with a `BLOCKINFO` abbreviation (literal code, fixed, VBR and a
    /// char6 array) and a local one (fixed code, blob), as DXC's writer uses them.
    fn abbreviated_stream() -> Vec<u8> {
        let mut w = BitWriter::new();
        for b in *b"BC\xC0\xDE" {
            w.emit(b as u64, 8);
        }
        let abbrev = |w: &mut BitWriter, ops: &[(u64, u64)]| {
            // (encoding, value): encoding 0 is a literal
            w.emit(2, 3);
            w.vbr(ops.len() as u64, 5);
            for &(e, v) in ops {
                w.emit((e == 0) as u64, 1);
                match e {
                    0 => w.vbr(v, 8),
                    1 | 2 => {
                        w.emit(e, 3);
                        w.vbr(v, 5)
                    }
                    _ => w.emit(e, 3),
                }
            }
        };
        w.enter_block(0);
        w.record(1, &[9]); // SETBID 9
        abbrev(&mut w, &[(0, 7), (1, 3), (2, 4), (3, 0), (4, 0)]);
        w.end_block();
        w.enter_block(9);
        abbrev(&mut w, &[(1, 8), (5, 0)]);
        w.emit(4, 3); // abbreviation 4: code 7, fixed 5, vbr 100, "dx."
        w.emit(5, 3);
        w.vbr(100, 4);
        w.vbr(3, 6);
        for c in [3, 23, 62] {
            w.emit(c, 6);
        }
        w.emit(5, 3); // abbreviation 5: code 42, blob "hi!"
        w.emit(42, 8);
        w.vbr(3, 6);
        w.align32();
        for &b in b"hi!" {
            w.emit(b as u64, 8);
        }
        w.align32();
        w.record(3, &[1, 2]);
        w.end_block();
        w.finish()
    }

    #[test]
    fn reader_decodes_abbreviations_from_blockinfo_and_blobs() {
        let bc = abbreviated_stream();
        let mut r = BitReader::new(&bc).unwrap();
        assert_eq!(r.next_entry(), Ok(Some(Entry::Enter(9))));
        assert_eq!(r.next_entry(), Ok(Some(Entry::Record(7))));
        assert_eq!(r.ops(), &[5, 100, b'd' as u64, b'x' as u64, b'.' as u64]);
        assert_eq!(r.next_entry(), Ok(Some(Entry::Record(42))));
        assert_eq!((r.ops(), r.blob()), (&[][..], Some(&b"hi!"[..])));
        assert_eq!(r.next_entry(), Ok(Some(Entry::Record(3))));
        assert_eq!(r.ops(), &[1, 2]);
        assert_eq!(r.next_entry(), Ok(Some(Entry::End(9))));
        assert_eq!(r.next_entry(), Ok(None));
    }

    #[test]
    fn truncated_or_bit_flipped_streams_are_errors_not_panics() {
        let bc = abbreviated_stream();
        let drain = |b: &[u8]| -> Result<(), String> {
            let mut r = BitReader::new(b)?;
            while r.next_entry()?.is_some() {}
            Ok(())
        };
        assert_eq!(drain(&bc), Ok(()));
        // A prefix reads only if it ends between top-level blocks (a word boundary).
        for n in 0..bc.len() {
            assert!(drain(&bc[..n]).is_err() || n % 4 == 0, "prefix of {n} bytes");
        }
        assert!(drain(&bc[..bc.len() - 4]).is_err());
        for bit in 32..bc.len() * 8 {
            let mut v = bc.clone();
            v[bit / 8] ^= 1 << (bit % 8);
            let _ = drain(&v);
        }
    }
}
