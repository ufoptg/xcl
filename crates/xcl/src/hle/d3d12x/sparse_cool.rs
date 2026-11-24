//! Sparse-window tiles the CPU has not written for a while leave the write-watch look (#582).
//!
//! Before every title submit `title_textures::refresh_once` asks `GetWriteWatch` which pages of
//! the watched ranges were written. That costs about 0.45 ms per GiB on a fast CPU and about
//! 1.4 ms per GiB on a Coffee Lake one, for every committed page whether written or not, and
//! the sparse windows are most of those bytes. A tile (64 KiB) with no written page for
//! `QUIET_LOOKS` looks is made read-only here and the look leaves it out. Its first CPU write
//! faults; the vectored handler makes it writable again and the look takes it back, and the
//! write itself is then seen by the write watch as before.
//!
//! Kernel calls do not fault in the title, they fail (ERROR_NOACCESS), so the places that let
//! the kernel write title memory warm the range first, as they commit lazy pages
//! (`kernelx::lazy::ensure`), and so do the title's own commit, decommit and protect calls.
//!
//! Off unless `XCL_SPARSE_COOL=1`.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, Once, OnceLock, RwLock};

use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};
use windows_sys::Win32::System::Memory::{
    GetWriteWatch, VirtualProtect, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_READONLY, PAGE_READWRITE,
};

pub(super) const TILE: usize = 0x1_0000;
/// looks in a row with no written page before a tile is made read-only
const QUIET_LOOKS: u64 = 8;
/// tiles made read-only in one look at most, so a look's extra cost stays bounded
const COOL_PER_LOOK: usize = 64;
/// tiles examined in one look at most (a cursor goes round the mapped tiles)
const EXAMINE_PER_LOOK: usize = 1024;
/// `Quiet::last` of a tile this module made read-only (skipped until a write marks it again)
const COOLED: u64 = u64::MAX;

const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

/// `XCL_SPARSE_COOL=1`
pub fn enabled() -> bool {
    #[cfg(test)]
    if tests::ON.with(|on| on.get()) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SPARSE_COOL").is_ok_and(|v| v == "1"))
}

/// tiles read-only now (left out of the look)
static COOL: RwLock<Option<HashSet<usize>>> = RwLock::new(None);
/// changes when a tile is cooled or warmed: the look's cached ranges are made again
static GEN: AtomicU64 = AtomicU64::new(0);
/// tiles cooled and warmed so far, for the timing line
static COUNTS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

struct Quiet {
    look: u64,
    /// the mapped tiles with no hole, their index, and where the cursor is in them
    tiles: Vec<usize>,
    index: HashMap<usize, u32>,
    /// per tile: the look it was last seen written (or first seen) in, or `COOLED`
    last: Vec<u64>,
    tiles_gen: u64,
    cursor: usize,
}

static QUIET: Mutex<Option<Quiet>> = Mutex::new(None);

pub fn gen() -> u64 {
    GEN.load(Ordering::Acquire)
}

/// (tiles read-only now, cooled so far, warmed so far)
pub fn counts() -> (usize, u64, u64) {
    let now = COOL.read().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(0, |s| s.len());
    (now, COUNTS[0].load(Ordering::Relaxed), COUNTS[1].load(Ordering::Relaxed))
}

fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(handler));
    });
}

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    let r = &*(*info).ExceptionRecord;
    // a write (ExceptionInformation[0] == 1) to a read-only tile of ours
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 || r.ExceptionInformation[0] != 1 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let tile = r.ExceptionInformation[1] & !(TILE - 1);
    if warm_tile(tile) || writable(r.ExceptionInformation[1]) {
        // warmed here, or by another thread a moment ago: the write goes on
        EXCEPTION_CONTINUE_EXECUTION
    } else {
        EXCEPTION_CONTINUE_SEARCH
    }
}

/// `a` can be written now: a tile another thread warmed between this thread's fault and its look
unsafe fn writable(a: usize) -> bool {
    let tile = a & !(TILE - 1);
    if !was_ours(tile) {
        return false;
    }
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    VirtualQuery(a as *const c_void, &mut m, std::mem::size_of_val(&m)) != 0 && m.State == MEM_COMMIT && m.Protect == PAGE_READWRITE
}

/// tiles ever cooled (so the handler only lets writes through to its own tiles)
static EVER: RwLock<Option<HashSet<usize>>> = RwLock::new(None);

fn was_ours(tile: usize) -> bool {
    EVER.read().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|s| s.contains(&tile))
}

/// Make `tile` writable again if it is read-only (true: it was).
fn warm_tile(tile: usize) -> bool {
    let mut g = COOL.write().unwrap_or_else(|e| e.into_inner());
    let Some(set) = g.as_mut() else { return false };
    if !set.remove(&tile) {
        return false;
    }
    let mut old = 0u32;
    // SAFETY: `tile` is a committed 64 KiB tile of a sparse window that this module made
    // read-only; restoring read-write changes no contents. A tile decommitted meanwhile makes
    // this fail, which leaves nothing to restore.
    unsafe { VirtualProtect(tile as *const c_void, TILE, PAGE_READWRITE, &mut old) };
    drop(g);
    COUNTS[1].fetch_add(1, Ordering::Relaxed);
    GEN.fetch_add(1, Ordering::Release);
    true
}

/// The read-only tiles of `[p, p + len)` made writable – memory the kernel, XCL or the title's
/// own memory calls are about to write, commit, decommit or protect.
pub fn warm(p: usize, len: usize) {
    if len == 0 || !enabled() {
        return;
    }
    let tiles: Vec<usize> = {
        let g = COOL.read().unwrap_or_else(|e| e.into_inner());
        let Some(set) = g.as_ref() else { return };
        if set.is_empty() {
            return;
        }
        let first = p & !(TILE - 1);
        let last = p.saturating_add(len - 1) & !(TILE - 1);
        if (last - first) / TILE < set.len() {
            (0..=(last - first) / TILE).map(|i| first + i * TILE).filter(|t| set.contains(t)).collect()
        } else {
            set.iter().copied().filter(|&t| t >= first && t <= last).collect()
        }
    };
    for t in tiles {
        warm_tile(t);
    }
}

/// `runs` (sorted, disjoint `[start, end)`) without the read-only tiles.
pub fn without_cool(runs: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if !enabled() {
        return runs;
    }
    let g = COOL.read().unwrap_or_else(|e| e.into_inner());
    let Some(set) = g.as_ref().filter(|s| !s.is_empty()) else { return runs };
    let mut out = Vec::with_capacity(runs.len());
    for (a, b) in runs {
        let mut start = a;
        let mut t = a & !(TILE - 1);
        while t < b {
            if set.contains(&t) {
                if start < t {
                    out.push((start, t));
                }
                start = t + TILE;
            }
            t += TILE;
        }
        if start < b {
            out.push((start, b));
        }
    }
    out
}

/// After a look: `dirty` (the written pages, sorted) mark their tiles written; then up to
/// `EXAMINE_PER_LOOK` of `tiles()` (the mapped tiles with no hole, made again when `tiles_gen`
/// changes) are examined and quiet ones made read-only.
pub unsafe fn after_look(dirty: &[usize], tiles_gen: u64, tiles: impl FnOnce() -> Vec<usize>) {
    if !enabled() {
        return;
    }
    let mut g = QUIET.lock().unwrap_or_else(|e| e.into_inner());
    let q = g.get_or_insert_with(|| Quiet {
        look: 0,
        tiles: Vec::new(),
        index: HashMap::new(),
        last: Vec::new(),
        tiles_gen: u64::MAX,
        cursor: 0,
    });
    q.look += 1;
    let look = q.look;
    if q.tiles_gen != tiles_gen {
        let tiles = tiles();
        let last: Vec<u64> = tiles.iter().map(|t| q.index.get(t).map_or(look, |&i| q.last[i as usize])).collect();
        q.index = tiles.iter().enumerate().map(|(i, &t)| (t, i as u32)).collect();
        q.tiles = tiles;
        q.last = last;
        q.tiles_gen = tiles_gen;
        q.cursor = 0;
    }
    // a written tile (a cooled one too: its write faulted and warmed it) is quiet again from now
    let mut prev = usize::MAX;
    for &p in dirty {
        let t = p & !(TILE - 1);
        if t != prev {
            if let Some(&i) = q.index.get(&t) {
                q.last[i as usize] = look;
            }
            prev = t;
        }
    }
    if q.tiles.is_empty() {
        return;
    }
    let mut cooled = 0usize;
    for _ in 0..q.tiles.len().min(EXAMINE_PER_LOOK) {
        if cooled == COOL_PER_LOOK {
            break;
        }
        q.cursor = (q.cursor + 1) % q.tiles.len();
        let i = q.cursor;
        let seen = q.last[i];
        if seen == COOLED || look - seen < QUIET_LOOKS {
            continue;
        }
        if cool_tile(q.tiles[i]) {
            q.last[i] = COOLED;
            cooled += 1;
        } else {
            q.last[i] = look;
        }
    }
    if cooled > 0 {
        COUNTS[0].fetch_add(cooled as u64, Ordering::Relaxed);
        GEN.fetch_add(1, Ordering::Release);
    }
}

/// Make `tile` read-only if it is all committed read-write and not written since the last look
/// (true: it is read-only now).
unsafe fn cool_tile(tile: usize) -> bool {
    {
        let g = COOL.read().unwrap_or_else(|e| e.into_inner());
        if g.as_ref().is_some_and(|s| s.contains(&tile)) {
            return false;
        }
    }
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    if VirtualQuery(tile as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0
        || m.State != MEM_COMMIT
        || m.Protect != PAGE_READWRITE
        || (m.BaseAddress as usize).saturating_add(m.RegionSize) < tile + TILE
    {
        return false;
    }
    install();
    EVER.write().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashSet::new).insert(tile);
    let mut g = COOL.write().unwrap_or_else(|e| e.into_inner());
    let mut old = 0u32;
    if VirtualProtect(tile as *const c_void, TILE, PAGE_READONLY, &mut old) == 0 {
        return false;
    }
    // A write between the look and the protect is in the write watch but the look will leave
    // this tile out from now on: such a tile stays writable (not reset, so the next look sees it).
    let mut buf = [std::ptr::null_mut::<c_void>(); 1];
    let mut count = buf.len();
    let mut gran = 0u32;
    if GetWriteWatch(0, tile as *const c_void, TILE, buf.as_mut_ptr(), &mut count, &mut gran) != 0 || count != 0 {
        VirtualProtect(tile as *const c_void, TILE, old, &mut old);
        return false;
    }
    g.get_or_insert_with(HashSet::new).insert(tile);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_RELEASE, MEM_RESERVE};

    const MEM_WRITE_WATCH: u32 = 0x0020_0000;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;

    thread_local! {
        /// `enabled()` on this test's thread only (other tests run beside it with it off)
        pub(super) static ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    unsafe fn watched(tiles: usize) -> usize {
        ON.with(|on| on.set(true));
        let p = VirtualAlloc(std::ptr::null(), tiles * TILE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
        assert_ne!(p, 0);
        p
    }

    unsafe fn written(base: usize, len: usize) -> usize {
        let mut buf = vec![std::ptr::null_mut::<c_void>(); len / 0x1000];
        let mut n = buf.len();
        let mut gran = 0u32;
        assert_eq!(GetWriteWatch(WRITE_WATCH_FLAG_RESET, base as *const c_void, len, buf.as_mut_ptr(), &mut n, &mut gran), 0);
        n
    }

    unsafe fn protect(a: usize) -> u32 {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        VirtualQuery(a as *const c_void, &mut m, std::mem::size_of_val(&m));
        m.Protect
    }

    /// A cooled tile leaves the runs; its first write faults, warms it and is still seen by the
    /// write watch afterwards.
    #[test]
    fn a_cooled_tile_is_warmed_by_its_first_write_and_the_write_is_watched() {
        unsafe {
            let p = watched(4);
            written(p, 4 * TILE);
            assert!(cool_tile(p + TILE));
            assert_eq!(protect(p + TILE), PAGE_READONLY);
            assert_eq!(without_cool(vec![(p, p + 4 * TILE)]), vec![(p, p + TILE), (p + 2 * TILE, p + 4 * TILE)]);
            std::ptr::write_volatile((p + TILE + 8) as *mut u32, 7);
            assert_eq!(std::ptr::read_volatile((p + TILE + 8) as *const u32), 7);
            assert_eq!(protect(p + TILE), PAGE_READWRITE);
            assert_eq!(without_cool(vec![(p, p + 4 * TILE)]), vec![(p, p + 4 * TILE)]);
            assert_eq!(written(p + TILE, TILE), 1, "the write after the fault is in the write watch");
            VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// A tile written after the look and before the protect is not cooled (the look would miss it).
    #[test]
    fn a_tile_written_since_the_look_is_not_cooled() {
        unsafe {
            let p = watched(1);
            written(p, TILE);
            std::ptr::write_volatile(p as *mut u32, 1);
            assert!(!cool_tile(p));
            assert_eq!(protect(p), PAGE_READWRITE);
            assert_eq!(written(p, TILE), 1, "still reported to the next look");
            VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// `warm` makes a range's read-only tiles writable without a fault (kernel writes).
    #[test]
    fn warm_restores_write_access_for_kernel_writes() {
        unsafe {
            let p = watched(3);
            written(p, 3 * TILE);
            assert!(cool_tile(p));
            assert!(cool_tile(p + TILE));
            assert!(cool_tile(p + 2 * TILE));
            warm(p + 100, TILE);
            assert_eq!(protect(p), PAGE_READWRITE);
            assert_eq!(protect(p + TILE), PAGE_READWRITE);
            assert_eq!(protect(p + 2 * TILE), PAGE_READONLY, "outside the range");
            warm(p + 2 * TILE, 1);
            VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// Quiet tiles are cooled after `QUIET_LOOKS` looks, a tile written every look is not.
    #[test]
    fn only_tiles_quiet_for_several_looks_are_cooled() {
        unsafe {
            let p = watched(2);
            written(p, 2 * TILE);
            // a generation of its own so this test's tile list is used
            let tiles_gen = p as u64;
            for _ in 0..QUIET_LOOKS * 2 {
                std::ptr::write_volatile(p as *mut u32, 1);
                assert!(written(p, 2 * TILE) >= 1);
                after_look(&[p], tiles_gen, || vec![p, p + TILE]);
            }
            assert_eq!(protect(p), PAGE_READWRITE, "written every look");
            assert_eq!(protect(p + TILE), PAGE_READONLY, "never written");
            warm(p, 2 * TILE);
            VirtualFree(p as *mut c_void, 0, MEM_RELEASE);
        }
    }
}
