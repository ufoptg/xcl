//! Write32/64BitValueBottomOfPipeX into memory the GPU does not see: done by the CPU once the GPU
//! has finished the work before them.
//!
//! A console title marks its GPU progress with bottom-of-pipe writes into its own memory and
//! waits on them (a retail game: plain `VirtualAlloc` memory, outside XCL's unified memory).
//! The PC GPU cannot write there; dropped, the title waited for ever after its second frame.
//! Here each queue gets a fence of XCL's: after a submission whose lists carry such writes the
//! queue signals it, and a worker thread does the writes, in submission order, when the GPU is
//! there – what bottom of pipe means.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Mutex};

use windows::core::Interface;
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::Win32::System::Threading as th;

/// A bottom-of-pipe write into memory outside unified memory, applied by the CPU once the
/// queue's fence reaches the submission that recorded it.
#[derive(Clone)]
pub(super) enum Write {
    /// Immediate dwords to store starting at `va`.
    Immediate(usize, Vec<u32>),
    /// A 64-bit value resolved by the GPU into a persistently mapped READBACK buffer slot
    /// (`src`), copied atomically to `va` once the queue's fence completes. Holding `_buf` keeps
    /// the mapping alive until the worker finishes.
    ReadbackU64 { va: usize, src: usize, _buf: ID3D12Resource },
}

struct Job {
    fence: ID3D12Fence,
    value: u64,
    writes: Vec<Write>,
}

// SAFETY: the fence is a free-threaded COM object; the writes are plain data
unsafe impl Send for Job {}

/// queue → its fence, last value signalled, and dedicated FIFO worker channel
struct QueueFence {
    fence: ID3D12Fence,
    value: u64,
    tx: mpsc::Sender<Job>,
}

unsafe impl Send for QueueFence {}

static FENCES: Mutex<Option<HashMap<usize, QueueFence>>> = Mutex::new(None);

/// how long the worker waits for one submission before writing anyway (a removed device, a
/// queue that never gets there)
const WAIT_MS: u32 = 10_000;

fn spawn_worker() -> Option<mpsc::Sender<Job>> {
    let event = unsafe { th::CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
    if event.is_null() {
        crate::xlog_error!("D3D12.X: CreateEventW failed for cpu_writes worker");
        return None;
    }
    let event_raw = event as usize;
    let (tx, rx) = mpsc::channel::<Job>();
    let spawned = std::thread::Builder::new().name("xcl-cpu-writes".into()).spawn(move || unsafe {
        let event = event_raw as *mut c_void;
        for job in rx {
            if job.fence.GetCompletedValue() < job.value {
                th::ResetEvent(event);
                if job.fence.SetEventOnCompletion(job.value, HANDLE(event)).is_ok() {
                    th::WaitForSingleObject(event, WAIT_MS);
                    if job.fence.GetCompletedValue() < job.value {
                        static LATE: AtomicU64 = AtomicU64::new(0);
                        let k = LATE.fetch_add(1, Ordering::Relaxed) + 1;
                        if k.is_power_of_two() {
                            crate::xlog_warn!("D3D12.X: {k} writes after the GPU so far done although their queue had not got there after {WAIT_MS} ms (fence at {}, wanted {}; {} writes)", job.fence.GetCompletedValue(), job.value, job.writes.len());
                        }
                    }
                }
            }
            for w in &job.writes {
                match w {
                    Write::Immediate(va, words) => write_now(*va, words),
                    Write::ReadbackU64 { va, src, .. } => {
                        let value = (*(*src as *const AtomicU64)).load(Ordering::Acquire);
                        write_u64_now(*va, value);
                    }
                }
            }
        }
        windows_sys::Win32::Foundation::CloseHandle(event);
    });
    if let Err(e) = spawned {
        unsafe { windows_sys::Win32::Foundation::CloseHandle(event) };
        crate::xlog_error!("D3D12.X: no thread for the writes after the GPU: {e}");
        return None;
    }
    Some(tx)
}

fn is_region_writable(va: usize, bytes: usize) -> bool {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_READWRITE};
    let Some(end) = va.checked_add(bytes) else { return false };
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    if unsafe { VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) } == 0 || m.State != MEM_COMMIT {
        return false;
    }
    m.Protect & PAGE_GUARD == 0
        && m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) != 0
        && (m.BaseAddress as usize).saturating_add(m.RegionSize) >= end
}

/// The dwords at `va`, if the memory is still there and writable.
/// An 8-byte aligned 64-bit immediate (`words.len() == 2`) is stored with a single 64-bit
/// atomic write so spinning CPU readers never observe a torn half-written value.
unsafe fn write_now(va: usize, words: &[u32]) {
    let Some(bytes) = words.len().checked_mul(4) else { return };
    if bytes == 0 {
        return;
    }
    if words.len() == 2 && va.is_multiple_of(8) {
        let value = (words[0] as u64) | ((words[1] as u64) << 32);
        write_u64_now(va, value);
        return;
    }
    if !va.is_multiple_of(4) || !is_region_writable(va, bytes) {
        return;
    }
    for (i, &w) in words.iter().enumerate() {
        (*((va + 4 * i) as *const AtomicU32)).store(w, Ordering::Release);
    }
}

/// An 8-byte aligned 64-bit value at `va`, if the memory is still there and writable.
unsafe fn write_u64_now(va: usize, value: u64) {
    if !va.is_multiple_of(8) || !is_region_writable(va, 8) {
        return;
    }
    (*(va as *const AtomicU64)).store(value, Ordering::Release);
}

/// Whether `[va, va + bytes)` is memory the CPU can write for the GPU later.
pub(super) fn writable(va: usize, bytes: usize) -> bool {
    crate::hle::kernelx::lazy::ensure(va, bytes);
    is_region_writable(va, bytes)
}

/// `writes` of lists just submitted to `queue`: done when the GPU has finished them.
pub(super) unsafe fn after(queue: *mut c_void, writes: Vec<Write>) {
    if writes.is_empty() || queue.is_null() {
        return;
    }
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return };
    let (tx, job) = {
        let mut g = FENCES.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        let entry = match m.entry(queue as usize) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let mut dev: Option<ID3D12Device> = None;
                if q.GetDevice(&mut dev).is_err() {
                    return;
                }
                let Some(dev) = dev else { return };
                let Ok(fence) = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) else { return };
                let Some(tx) = spawn_worker() else { return };
                v.insert(QueueFence { fence, value: 0, tx })
            }
        };
        let next = entry.value + 1;
        if q.Signal(&entry.fence, next).is_err() {
            return;
        }
        entry.value = next;
        (entry.tx.clone(), Job { fence: entry.fence.clone(), value: next, writes })
    };
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| crate::xlog_info!("D3D12.X: bottom-of-pipe writes outside unified memory are done by the CPU after the GPU ({} in the first submission)", job.writes.len()));
    let _ = tx.send(job);
}

/// Remove a released command queue from `FENCES`, dropping its sender so its worker exits
/// cleanly and preventing stale fence reuse if the queue address is later reallocated.
pub(super) fn on_queue_release(queue: usize) {
    if let Some(m) = FENCES.lock().unwrap().as_mut() {
        m.remove(&queue);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Dxgi::*;

    struct QueueCleanup(usize);
    impl Drop for QueueCleanup {
        fn drop(&mut self) {
            on_queue_release(self.0);
        }
    }

    /// The value lands after the queue has done what came before it.
    #[test]
    fn writes_land_after_the_queue() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let _cleanup = QueueCleanup(q.as_raw() as usize);
            let cell = Box::new([0u32; 2]);
            let va = cell.as_ptr() as usize;
            assert!(writable(va, 8));
            after(q.as_raw(), vec![Write::Immediate(va, vec![7, 9])]);
            let t = std::time::Instant::now();
            while (*(va as *const AtomicU32)).load(Ordering::Acquire) != 7 {
                assert!(t.elapsed().as_secs() < 5, "never written");
                std::thread::yield_now();
            }
            assert_eq!(*cell, [7, 9]);
        }
    }

    /// A ReadbackU64 write copies the 64-bit value from the mapped readback buffer after the queue finishes.
    #[test]
    fn readback_u64_lands_after_the_queue() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let _cleanup = QueueCleanup(q.as_raw() as usize);
            let hp = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 8,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut rb: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).unwrap();
            let rb = rb.unwrap();
            let mut ptr: *mut c_void = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut ptr)).unwrap();
            let expected = 0x1234_5678_9ABC_DEF0u64;
            (*(ptr as *const AtomicU64)).store(expected, Ordering::Release);

            let dst = Box::new(0u64);
            let va = dst.as_ref() as *const u64 as usize;
            assert!(writable(va, 8));
            after(q.as_raw(), vec![Write::ReadbackU64 { va, src: ptr as usize, _buf: rb }]);
            let t = std::time::Instant::now();
            while (*(va as *const AtomicU64)).load(Ordering::Acquire) != expected {
                assert!(t.elapsed().as_secs() < 5, "readback value never written");
                std::thread::yield_now();
            }
            assert_eq!(*dst, expected);
        }
    }

    #[test]
    fn independent_queues_do_not_hol_block_each_other() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let q_a: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let q_b: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let _cleanup_a = QueueCleanup(q_a.as_raw() as usize);
            let _cleanup_b = QueueCleanup(q_b.as_raw() as usize);
            let blocker: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

            // Block queue A on `blocker` so its bottom-of-pipe fence stays unsignaled.
            q_a.Wait(&blocker, 1).unwrap();
            let cell_a = Box::new(0u64);
            let cell_b = Box::new(0u64);
            let va_a = cell_a.as_ref() as *const u64 as usize;
            let va_b = cell_b.as_ref() as *const u64 as usize;
            let t_a = std::time::Instant::now();
            after(q_a.as_raw(), vec![Write::Immediate(va_a, vec![1, 0])]);
            after(q_b.as_raw(), vec![Write::Immediate(va_b, vec![0x1122_3344, 0x5566_7788])]);

            // Queue B must complete well before WAIT_MS (10_000 ms) while queue A is still blocked.
            let t = std::time::Instant::now();
            while (*(va_b as *const AtomicU64)).load(Ordering::Acquire) != 0x5566_7788_1122_3344 {
                assert!(t.elapsed().as_secs() < 3, "queue B was HOL-blocked behind queue A");
                std::thread::yield_now();
            }
            if t_a.elapsed() < std::time::Duration::from_millis(WAIT_MS as u64) {
                assert_eq!((*(va_a as *const AtomicU64)).load(Ordering::Acquire), 0);
            }

            // Unblock queue A and confirm its write lands cleanly too.
            blocker.Signal(1).unwrap();
            let t = std::time::Instant::now();
            while (*(va_a as *const AtomicU64)).load(Ordering::Acquire) != 1 {
                assert!(t.elapsed().as_secs() < 5, "queue A never wrote after unblock");
                std::thread::yield_now();
            }
        }
    }

    #[test]
    fn write_now_stores_64bit_immediates_atomically() {
        let cell = Box::new(0u64);
        let va = cell.as_ref() as *const u64 as usize;
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_r = stop.clone();
        let reader = std::thread::spawn(move || unsafe {
            let ptr = va as *const AtomicU64;
            while !stop_r.load(Ordering::Relaxed) {
                let v = (*ptr).load(Ordering::Acquire);
                if v != 0 {
                    let lo = v as u32;
                    let hi = (v >> 32) as u32;
                    assert_eq!(hi, !lo, "observed torn 64-bit immediate write: {v:#018x}");
                }
            }
        });
        unsafe {
            for i in 1u32..=20_000 {
                write_now(va, &[i, !i]);
            }
        }
        stop.store(true, Ordering::Relaxed);
        reader.join().unwrap();
        assert_eq!(*cell, (20_000u64) | ((!20_000u64) << 32));
    }
}
