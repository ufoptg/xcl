//! 4 KiB sub-allocated default heaps for small (< 64 KiB) sampled textures.
//!
//! A console title packs thousands of small mip-tails, icons, decals and LUTs at 256 B – 4 KiB
//! offsets inside its texture pools. Backing each one with a 64 KiB reserved resource in
//! `vram::Pages` wastes up to 16× video memory (one full 64 KiB `extra_pages` tile per small
//! texture) and submits a `CreateReservedResource` + `GetResourceTiling` + `UpdateTileMappings`
//! round trip for every placement.
//!
//! Direct3D 12 supports 4 KiB placement (`D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT`) for
//! single-sample non-RT/DS textures whose allocation info fits under 64 KiB. Placing those in
//! shared 16 MiB default heaps packs up to 4 096 textures into one heap with zero tile mappings,
//! shares the exact `(heap, offset)` across repeated placements at the same `(va, size)` so
//! same-address aliases see each other's uploads, and recycles released blocks by size class
//! after `IDLE` + `COOL`. `XCL_VRAM_SMALL=0` disables this path.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::util::HRESULT;

/// Allocation unit: 4 KiB (`D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT`).
const UNIT: usize = 0x1000;
/// Largest small-resource size handled here: 60 KiB (`15 * 4 KiB < 64 KiB`).
const MAX_UNITS: u32 = 15;
/// Units per shared default heap: 16 MiB (`4096 * 4 KiB`).
const POOL_UNITS: u32 = 4096;
/// Keep an unreferenced `(va, units)` slot this long so immediate re-placements reuse it.
const IDLE: Duration = Duration::from_secs(2);
/// Wait before reusing a freed block for a different address (in-flight GPU work).
const COOL: Duration = Duration::from_secs(2);

/// Key identifying a shared small-texture backing slot: `(title_va, size_in_4k_units)`.
type SlotKey = (usize, u32);

/// Block location inside a device's small pools: `(pool_index, offset_in_4k_units, size_in_4k_units)`.
type Block = (u32, u32, u32);

struct Pool<H> {
    heap: H,
    /// Bump pointer in 4 KiB units (`0..=POOL_UNITS`).
    next_unit: u32,
}

struct Slot {
    pool: u32,
    offset: u32,
    users: u32,
    idle_since: Option<Instant>,
}

struct SmallPools<H> {
    pools: Vec<Pool<H>>,
    /// Per-size-class (`1..=MAX_UNITS`) reusable blocks `(pool, offset)`.
    free: [Vec<(u32, u32)>; (MAX_UNITS as usize) + 1],
    /// Blocks given back, reusable after `COOL`.
    cooling: VecDeque<(Block, Instant)>,
    /// Live or idle-kept `(va, units)` → backing block.
    slots: HashMap<SlotKey, Slot>,
    /// Idle slots, oldest first (`sweep_idle`).
    idle: VecDeque<(SlotKey, Instant)>,
    /// Live resource raw pointer → `(va, units)`.
    resources: HashMap<usize, SlotKey>,
    /// Total 4 KiB units currently held by `slots` (live + idle-kept).
    used_units: usize,
}

impl<H: Clone> SmallPools<H> {
    fn new() -> Self {
        SmallPools {
            pools: Vec::new(),
            free: std::array::from_fn(|_| Vec::new()),
            cooling: VecDeque::new(),
            slots: HashMap::new(),
            idle: VecDeque::new(),
            resources: HashMap::new(),
            used_units: 0,
        }
    }

    fn sweep_idle(&mut self, now: Instant) {
        while self.idle.front().is_some_and(|&(_, since)| now.duration_since(since) >= IDLE) {
            let Some((key, since)) = self.idle.pop_front() else { break };
            if self.slots.get(&key).is_some_and(|s| s.idle_since == Some(since)) {
                if let Some(s) = self.slots.remove(&key) {
                    self.used_units -= key.1 as usize;
                    self.cooling.push_back(((s.pool, s.offset, key.1), now));
                }
            }
        }
    }

    fn drain_cooling(&mut self, now: Instant) {
        while self.cooling.front().is_some_and(|&(_, at)| now.duration_since(at) >= COOL) {
            if let Some(((pool, offset, units), _)) = self.cooling.pop_front() {
                self.free[units as usize].push((pool, offset));
            }
        }
    }

    /// Try to acquire an existing shared slot for `(va, units)` or allocate a block from existing
    /// pools / free lists without creating a new D3D12 heap.
    fn try_acquire(&mut self, key: SlotKey, now: Instant) -> Option<(H, u32)> {
        self.sweep_idle(now);
        if let Some(s) = self.slots.get_mut(&key) {
            s.users += 1;
            s.idle_since = None;
            return Some((self.pools[s.pool as usize].heap.clone(), s.offset));
        }
        let units = key.1;
        debug_assert!((1..=MAX_UNITS).contains(&units));
        self.drain_cooling(now);
        let (pool, offset) = if let Some(b) = self.free[units as usize].pop() {
            b
        } else if let Some((idx, p)) = self.pools.iter_mut().enumerate().rev().find(|(_, p)| p.next_unit + units <= POOL_UNITS) {
            let off = p.next_unit;
            p.next_unit += units;
            (idx as u32, off)
        } else {
            return None;
        };
        let heap = self.pools[pool as usize].heap.clone();
        self.slots.insert(key, Slot { pool, offset, users: 1, idle_since: None });
        self.used_units += units as usize;
        Some((heap, offset))
    }

    /// Register a newly created pool heap and acquire `(va, units)`.
    /// Re-checks `try_acquire` first under the lock so two threads racing to create a new 16 MiB
    /// pool do not push a redundant empty heap (the extra `heap` drops cleanly if another thread
    /// already added room or created the same slot).
    fn add_pool_and_acquire(&mut self, heap: H, key: SlotKey, now: Instant) -> (H, u32) {
        if let Some(got) = self.try_acquire(key, now) {
            return got;
        }
        let pool_idx = self.pools.len() as u32;
        self.pools.push(Pool { heap, next_unit: 0 });
        if let Some(got) = self.try_acquire(key, now) {
            return got;
        }
        // `pool_idx` has `next_unit == 0` and `units <= MAX_UNITS < POOL_UNITS`, so `try_acquire`
        // above always succeeds.
        unreachable!("fresh pool {pool_idx} must fit {} units", key.1);
    }

    fn note_resource(&mut self, res: usize, key: SlotKey) {
        self.resources.insert(res, key);
    }

    /// Release one user of `key`. When `rollback` is true (creation failed before the GPU ever saw
    /// the resource), the block returns directly to `free` if this was its only user.
    fn release_key(&mut self, key: SlotKey, now: Instant, rollback: bool) {
        let Some(s) = self.slots.get_mut(&key) else { return };
        s.users = s.users.saturating_sub(1);
        if s.users == 0 {
            if rollback {
                if let Some(s) = self.slots.remove(&key) {
                    self.used_units -= key.1 as usize;
                    self.free[key.1 as usize].push((s.pool, s.offset));
                }
            } else {
                s.idle_since = Some(now);
                self.idle.push_back((key, now));
            }
        }
    }

    fn forget(&mut self, res: usize, now: Instant) -> bool {
        let Some(key) = self.resources.remove(&res) else {
            return false;
        };
        self.release_key(key, now, false);
        true
    }

    fn bytes(&self) -> (usize, usize) {
        (self.used_units * UNIT, self.pools.len() * POOL_UNITS as usize * UNIT)
    }
}

// D3D12 heaps are free-threaded.
unsafe impl Send for SmallPools<ID3D12Heap> {}

static SMALL_POOLS: Mutex<BTreeMap<usize, SmallPools<ID3D12Heap>>> = Mutex::new(BTreeMap::new());

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_VRAM_SMALL").map_or(true, |v| v.trim() != "0"))
}

/// Whether `desc` and `console_bytes` are eligible for 4 KiB small-resource sub-allocation.
fn candidate(desc: &D3D12_RESOURCE_DESC, console_bytes: Option<u64>) -> bool {
    const PAGE: u64 = 0x1_0000;
    desc.Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN
        && desc.SampleDesc.Count <= 1
        && desc.Flags.0 & 0x7 == 0
        && matches!(desc.Dimension, D3D12_RESOURCE_DIMENSION_TEXTURE2D | D3D12_RESOURCE_DIMENSION_TEXTURE3D)
        && console_bytes.is_some_and(|b| b > 0 && b < PAGE)
}

/// Place a sub-64 KiB sampled texture at 4 KiB granularity in a shared default heap.
/// Returns `None` if the texture is not a small resource (the caller falls back to `vram::Pages`
/// or a committed resource).
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn place(
    dev: &ID3D12Device,
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    console_bytes: Option<u64>,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    if !enabled() || !candidate(desc, console_bytes) {
        return None;
    }
    let small_desc = D3D12_RESOURCE_DESC { Alignment: D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT as u64, ..*desc };
    let info = dev.GetResourceAllocationInfo(0, &[small_desc]);
    if info.SizeInBytes == u64::MAX || info.SizeInBytes == 0 || info.Alignment != UNIT as u64 || info.SizeInBytes >= 0x1_0000 {
        return None;
    }
    let units = (info.SizeInBytes as usize).div_ceil(UNIT) as u32;
    if !(1..=MAX_UNITS).contains(&units) {
        return None;
    }
    let dev_key = dev.as_raw() as usize;
    let key = (va, units);
    let now = Instant::now();

    let (heap, offset_units) = {
        let mut g = SMALL_POOLS.lock().unwrap();
        let pools = g.entry(dev_key).or_insert_with(SmallPools::new);
        pools.try_acquire(key, now)
    }
    .or_else(|| {
        // Create a new 16 MiB default heap outside `SMALL_POOLS.lock()`.
        let heap_desc = D3D12_HEAP_DESC {
            SizeInBytes: POOL_UNITS as u64 * UNIT as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            // 64 KiB heap alignment is valid for 4 KiB-aligned small resources at any 4 KiB offset.
            Alignment: D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
        };
        let mut new_heap: Option<ID3D12Heap> = None;
        if let Err(e) = super::title_textures::own_work(|| dev.CreateHeap(&heap_desc, &mut new_heap)) {
            crate::xlog_warn!("D3D12.X: small texture pool CreateHeap: {e}");
            return None;
        }
        let new_heap = new_heap?;
        let mut g = SMALL_POOLS.lock().unwrap();
        let pools = g.entry(dev_key).or_insert_with(SmallPools::new);
        Some(pools.add_pool_and_acquire(new_heap, key, Instant::now()))
    })?;

    let hr = super::real::dev_create_placed_resource(
        this,
        heap.as_raw() as usize,
        offset_units as usize * UNIT,
        &small_desc as *const _ as usize,
        state as usize,
        clear as usize,
        riid,
        ppv,
    );
    if hr < 0 || (*ppv).is_null() {
        let mut g = SMALL_POOLS.lock().unwrap();
        if let Some(pools) = g.get_mut(&dev_key) {
            pools.release_key(key, Instant::now(), true);
        }
        return None;
    }
    let raw = *ppv as usize;
    SMALL_POOLS.lock().unwrap().entry(dev_key).or_insert_with(SmallPools::new).note_resource(raw, key);
    Some(hr)
}

/// Release a small-pool texture on COM release. Returns `true` if `resource` belonged to a small
/// pool (so `vram::forget` can skip locking `PAGES`).
pub(super) fn forget(resource: usize) -> bool {
    let mut g = SMALL_POOLS.lock().unwrap();
    let now = Instant::now();
    for pools in g.values_mut() {
        if pools.forget(resource, now) {
            return true;
        }
    }
    false
}

/// `(used_bytes, pool_bytes)` across all devices.
pub(super) fn bytes() -> (usize, usize) {
    let g = SMALL_POOLS.lock().unwrap();
    g.values().fold((0, 0), |(used, pools), p| {
        let (u, t) = p.bytes();
        (used + u, pools + t)
    })
}

#[cfg(test)]
pub(super) fn bytes_for(device: usize) -> (usize, usize) {
    let g = SMALL_POOLS.lock().unwrap();
    g.get(&device).map_or((0, 0), SmallPools::bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_pool_packs_shares_and_recycles_by_size_class() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut p: SmallPools<u32> = SmallPools::new();
        let alloc = |p: &mut SmallPools<u32>, va: usize, units: u32, res: usize, now: Instant| -> (u32, u32) {
            let got = p.try_acquire((va, units), now).unwrap_or_else(|| {
                let id = p.pools.len() as u32;
                p.add_pool_and_acquire(id, (va, units), now)
            });
            p.note_resource(res, (va, units));
            got
        };

        // Sixteen 4 KiB textures pack into 64 KiB (not 16 * 64 KiB = 1 MiB).
        for i in 0..16usize {
            let (h, off) = alloc(&mut p, 0x10_0000 + i * UNIT, 1, 100 + i, at(0));
            assert_eq!((h, off), (0, i as u32));
        }
        assert_eq!(p.bytes(), (16 * UNIT, POOL_UNITS as usize * UNIT));

        // Re-placing the exact same (va, units) shares the exact same (heap, offset) with zero growth.
        let (h_dup, off_dup) = alloc(&mut p, 0x10_4000, 1, 200, at(10));
        assert_eq!((h_dup, off_dup), (0, 4));
        assert_eq!(p.bytes().0, 16 * UNIT);

        // Releasing one of the two aliases keeps the slot alive; releasing both makes it idle.
        assert!(p.forget(200, at(20)));
        assert_eq!(p.slots[&(0x10_4000, 1)].users, 1);
        assert!(p.forget(104, at(30)));
        assert_eq!(p.slots[&(0x10_4000, 1)].users, 0);

        // Re-acquiring within IDLE reuses the exact same slot.
        let (h_re, off_re) = alloc(&mut p, 0x10_4000, 1, 201, at(1000));
        assert_eq!((h_re, off_re), (0, 4));
        assert!(p.forget(201, at(1100)));

        // After IDLE (2 s) + COOL (2 s), the 4 KiB block is recycled for a new address.
        let (h_wait, off_wait) = alloc(&mut p, 0x20_0000, 1, 202, at(3200));
        assert_eq!((h_wait, off_wait), (0, 16), "still cooling at +2.1s, must not reuse yet");
        let (h_rec, off_rec) = alloc(&mut p, 0x20_1000, 1, 203, at(5300));
        assert_eq!((h_rec, off_rec), (0, 4), "block (0, 4) recycled after IDLE + COOL");

        // Rollback on creation failure immediately returns the block to the free list.
        let (h_fail, off_fail) = p.try_acquire((0x30_0000, 4), at(5400)).unwrap();
        assert_eq!((h_fail, off_fail), (0, 17));
        p.release_key((0x30_0000, 4), at(5400), true);
        let (h_next, off_next) = alloc(&mut p, 0x30_4000, 4, 204, at(5400));
        assert_eq!((h_next, off_next), (0, 17));

        // Concurrent CreateHeap race: if room already exists in pool 0, `add_pool_and_acquire`
        // must reuse existing room instead of pushing a redundant empty 16 MiB pool.
        let pools_before = p.pools.len();
        let (h_raced, _) = p.add_pool_and_acquire(999, (0x40_0000, 1), at(5500));
        assert_eq!(h_raced, 0);
        assert_eq!(p.pools.len(), pools_before, "must not push redundant pool when room already exists");
    }

    #[test]
    fn candidate_filters_large_gpu_owned_and_msaa_textures() {
        let mut d = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 64,
            Height: 64,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_NONE,
            ..Default::default()
        };
        assert!(candidate(&d, Some(16 * 1024)));
        assert!(!candidate(&d, Some(64 * 1024)), "64 KiB stays on vram::Pages");
        assert!(!candidate(&d, None));
        assert!(!candidate(&d, Some(0)));
        d.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        assert!(!candidate(&d, Some(16 * 1024)));
        d.Flags = D3D12_RESOURCE_FLAG_NONE;
        d.SampleDesc.Count = 4;
        assert!(!candidate(&d, Some(16 * 1024)));
        d.SampleDesc.Count = 1;
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        assert!(!candidate(&d, Some(16 * 1024)));
    }

    /// On the GPU: 512 small 4 KiB (64x64 BC1) and 16 KiB (64x64 RGBA8) sampled textures pack at
    /// 4 KiB granularity inside a single 16 MiB heap (16x and 4x smaller than 64 KiB tiles),
    /// same-address placements alias their pixels on hardware, adjacent 4 KiB textures have zero
    /// crosstalk, and a small sampled texture over an existing 64 KiB GPU target falls back to
    /// `vram::Pages` so target aliasing is preserved.
    #[test]
    fn gpu_small_textures_pack_at_4k_granularity_and_preserve_pixels() {
        let _vram_guard = super::super::vram::VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory6 = CreateDXGIFactory1().expect("DXGI factory");
            let sel = crate::gpu::adapter::Selector::from_env_or("auto");
            let mut dev: Option<ID3D12Device> = None;
            for (adapter, _) in crate::gpu::adapter::candidates(&factory, &sel) {
                if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev).is_ok() && dev.is_some() {
                    break;
                }
            }
            let Some(dev) = dev else { return };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let dev_key = dev.as_raw() as usize;
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;

            let desc_of = |w: u64, h: u32, fmt: DXGI_FORMAT, flags: D3D12_RESOURCE_FLAGS| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: w,
                Height: h,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: fmt,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: flags,
                ..Default::default()
            };
            let place_sampled = |va: usize, d: &D3D12_RESOURCE_DESC, cbytes: u64| -> ID3D12Resource {
                let mut p = std::ptr::null_mut();
                let hr = super::super::vram::place_pages(dev.as_raw(), va, d, D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), Some(cbytes), riid, &mut p);
                assert!(hr.is_some_and(|h| h >= 0) && !p.is_null(), "place_pages at {va:#x} failed: {hr:x?}");
                assert!(super::super::vram::is_placed_texture(p as usize));
                ID3D12Resource::from_raw(p)
            };

            // 1. 512 x 64x64 BC1 (2 KiB on console -> 4 KiB small resource on PC): must pack into
            //    2 MiB (16x smaller than 512 x 64 KiB = 32 MiB!) inside a single 16 MiB heap.
            let bc1 = desc_of(64, 64, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_NONE);
            let (small0, _) = bytes_for(dev_key);
            let (pages0, _) = super::super::vram::page_bytes_for(dev_key);
            let bc1_base = 0x60_0000_0000usize;
            let mut bc1_res = Vec::with_capacity(512);
            for k in 0..512usize {
                bc1_res.push(place_sampled(bc1_base + k * 0x1000, &bc1, 2048));
            }
            let (small_after_bc1, small_pool_after_bc1) = bytes_for(dev_key);
            let bc1_small_delta = small_after_bc1 - small0;
            let bc1_total_delta = super::super::vram::page_bytes_for(dev_key).0 - pages0;
            println!("512 x 64x64 BC1: +{bc1_small_delta} B in small pools (pool={small_pool_after_bc1} B), +{bc1_total_delta} B total (vs 33554432 B as 64 KiB tiles)");
            assert_eq!(bc1_small_delta, 512 * UNIT, "each 64x64 BC1 must take 4 KiB in small pool");
            assert_eq!(bc1_total_delta, 512 * UNIT, "zero 64 KiB Pages tiles allocated for 4 KiB BC1 textures");
            drop(bc1_res);

            // 2. Hardware upload + readback: two adjacent 64x64 RGBA8 (16 KiB) textures and a
            //    same-address alias verify zero crosstalk and exact pixel sharing on the GPU.
            let rgba = desc_of(64, 64, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
            let va_a = 0x61_0000_1000usize;
            let va_b = 0x61_0000_5000usize;
            let a1 = place_sampled(va_a, &rgba, 16384);
            let (small_after_a1, _) = bytes_for(dev_key);
            let a2 = place_sampled(va_a, &rgba, 16384);
            assert_eq!(bytes_for(dev_key).0, small_after_a1, "same-va small texture must share slot");
            let b = place_sampled(va_b, &rgba, 16384);
            assert_eq!(bytes_for(dev_key).0 - small_after_a1, 16384, "64x64 RGBA8 must take 16 KiB");

            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let fill_a = vec![0x3Cu8; 64 * 64 * 4];
            let fill_b = vec![0xA5u8; 64 * 64 * 4];
            let sub = |data: Vec<u8>| crate::gpu::texture::Subresource { rows: 64, row_bytes: 64 * 4, depth: 1, data };
            let batch = vec![
                (a2.clone(), vec![sub(fill_a.clone())], vec![D3D12_RESOURCE_STATE_COMMON]),
                (b.clone(), vec![sub(fill_b.clone())], vec![D3D12_RESOURCE_STATE_COMMON]),
            ];
            let failed = crate::gpu::texture::upload_batch(&dev, &queue, &batch).unwrap();
            assert!(failed.is_empty(), "upload failed: {failed:?}");

            let readback_tex = |r: &ID3D12Resource| -> Vec<u8> {
                let rd = r.GetDesc();
                let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
                let (mut rows, mut row_bytes, mut total) = (0u32, 0u64, 0u64);
                dev.GetCopyableFootprints(&rd, 0, 1, 0, Some(&mut fp), Some(&mut rows), Some(&mut row_bytes), Some(&mut total));
                let mut rb: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb,
                ).unwrap();
                let rb = rb.unwrap();
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                list.Close().unwrap();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                crate::gpu::texture::wait(&dev, &queue).unwrap();
                let mut p = std::ptr::null_mut();
                rb.Map(0, None, Some(&mut p)).unwrap();
                let mut out = Vec::with_capacity(rows as usize * row_bytes as usize);
                for y in 0..rows as usize {
                    out.extend_from_slice(std::slice::from_raw_parts((p as *const u8).add(y * fp.Footprint.RowPitch as usize), row_bytes as usize));
                }
                rb.Unmap(0, None);
                out
            };

            assert_eq!(readback_tex(&a1), fill_a, "same-va alias a1 must see pixels uploaded to a2");
            assert_eq!(readback_tex(&b), fill_b, "adjacent 4 KiB texture b must not crosstalk with a");
            drop((b, a2, a1));

            // 3. A small sampled texture at a 64 KiB-aligned va occupied by an existing GPU-owned
            //    render target must stay on vram::Pages (not vram_small) so it aliases the target.
            let rt_va = 0x62_0000_0000usize;
            let rt_desc = desc_of(64, 64, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
            let mut rt_raw = std::ptr::null_mut();
            let hr = super::super::vram::place(dev.as_raw(), rt_va, &rt_desc, D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32, std::ptr::null(), Some(16384), riid, &mut rt_raw);
            assert!(hr.is_some_and(|h| h >= 0) && !rt_raw.is_null());
            let rt = ID3D12Resource::from_raw(rt_raw);
            let small_before_alias = bytes_for(dev_key).0;
            let sampled_view = place_sampled(rt_va, &rgba, 16384);
            assert_eq!(bytes_for(dev_key).0, small_before_alias, "sampled view over live GPU target must alias vram::Pages, not vram_small");
            drop((sampled_view, rt));
        }
    }
}
