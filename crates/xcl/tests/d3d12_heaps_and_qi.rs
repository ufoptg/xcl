//! Host-driver parity and contract tests for `ID3D12Device::CreateHeap` (#28),
//! `ID3D12Device4::CreateHeap1` (#54), `QueryInterface` (#0), and XCL's
//! `IDXGIDevice`/`1`/`2` adapter surface without console game files (Part of #480).

use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::null_mut;
use std::sync::{Arc, Once};

use windows::core::{Interface, IUnknown};
use windows::Win32::Foundation::E_POINTER;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::*;
use windows_sys::core::GUID as SysGuid;
use windows_sys::Win32::Foundation::CloseHandle;
use windows_sys::Win32::System::Diagnostics::Debug::RtlLookupFunctionEntry;
use windows_sys::Win32::System::Threading::CreateEventW;

use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::{self, layout, real};
use xcl::modules::{self, MappedModule};
use xcl::util::{E_INVALIDARG, E_NOINTERFACE, HRESULT, S_OK};

const E_NOTIMPL: HRESULT = 0x8000_4001u32 as i32;

/// Unknown IID used to test `creator!` fallback to `ID3D12Heap` and `QueryInterface` parity.
const CONSOLE_UNKNOWN_IID: SysGuid = SysGuid::from_u128(0x22334455_6677_8899_aabb_ccddeeff0011);

type QiFn = unsafe extern "system" fn(*mut c_void, *const SysGuid, *mut *mut c_void) -> HRESULT;
type CreateHeapFn = unsafe extern "system" fn(*mut c_void, *const D3D12_HEAP_DESC, *const SysGuid, *mut *mut c_void) -> HRESULT;
type CreateHeap1Fn = unsafe extern "system" fn(*mut c_void, *const D3D12_HEAP_DESC, *mut c_void, *const SysGuid, *mut *mut c_void) -> HRESULT;

#[inline(never)]
unsafe fn title_qi(f: QiFn, dev: *mut c_void, riid: *const SysGuid, ppv: *mut *mut c_void) -> HRESULT {
    // Distinct `black_box` tag kept live across `f` prevents `/OPT:ICF` folding and tail calls in release builds.
    let tag = std::hint::black_box(0x5449_544Cu32);
    let r = f(dev, riid, ppv);
    std::hint::black_box((r, tag)).0
}

#[inline(never)]
unsafe fn pc_qi(f: QiFn, dev: *mut c_void, riid: *const SysGuid, ppv: *mut *mut c_void) -> HRESULT {
    let tag = std::hint::black_box(0x5043_5F51u32);
    let r = f(dev, riid, ppv);
    std::hint::black_box((r, tag)).0
}

/// Registers `title_qi` as a synthetic mapped module so `guard::make_split` sends its calls to
/// `dev_qi_x` while `pc_qi` keeps taking `qi_override` (`orig_fn`).
fn ensure_title_callers() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let lookup = |addr: u64| -> (usize, usize) {
            let mut image_base = 0u64;
            let entry = unsafe {
                RtlLookupFunctionEntry(addr, &mut image_base, null_mut())
                    .as_ref()
                    .expect("unwind entry for caller")
            };
            ((image_base + u64::from(entry.BeginAddress)) as usize, (entry.EndAddress - entry.BeginAddress) as usize)
        };
        let (base, size) = lookup(title_qi as *const () as u64);
        let (pc_base, _) = lookup(pc_qi as *const () as u64);
        assert_ne!(base, pc_base, "title_qi and pc_qi must remain distinct code blocks");
        modules::register_mapped(Arc::new(MappedModule {
            name: "d3d12-heaps-qi-fixture".into(),
            path: Default::default(),
            base,
            size,
            entry_rva: 0,
            is_dll: false,
            exports: Default::default(),
            tls_callbacks: Default::default(),
            no_thread_calls: Default::default(),
            initialized: Default::default(),
            stack_reserve: 0,
        }));
    });
}

fn init_device() -> Option<ID3D12Device> {
    if d3d12x::backend() != d3d12x::Backend::Real {
        runner::skip_test("no GPU");
        return None;
    }
    ensure_title_callers();
    let mut raw = null_mut();
    let hr = unsafe { real::create_device(&ID3D12Device::IID as *const _ as *const c_void, &mut raw) };
    assert!(hr >= 0 && !raw.is_null(), "real::create_device failed: {hr:#x}");
    Some(unsafe { ID3D12Device::from_raw(raw) })
}

fn slot_of(iface_name: &str, method: &str) -> usize {
    let iface = layout::find(iface_name).unwrap_or_else(|| panic!("missing {iface_name} in layout"));
    layout::slot(iface, method).unwrap_or_else(|| panic!("missing {iface_name}::{method} in layout"))
}

unsafe fn vtable_fn<F: Copy>(obj: *mut c_void, slot: usize) -> F {
    assert_eq!(size_of::<F>(), size_of::<usize>());
    let vt = *(obj as *const *const usize);
    let addr = *vt.add(slot);
    assert_ne!(addr, 0, "null vtable entry at slot {slot}");
    std::mem::transmute_copy(&addr)
}

unsafe fn orig_slot_fn<F: Copy>(obj: *mut c_void, slot: usize) -> F {
    assert_eq!(size_of::<F>(), size_of::<usize>());
    let addr = real::orig_fn(obj, slot);
    assert_ne!(addr, 0, "null original vtable entry at slot {slot}");
    std::mem::transmute_copy(&addr)
}

fn sys_guid(g: windows::core::GUID) -> SysGuid {
    SysGuid { data1: g.data1, data2: g.data2, data3: g.data3, data4: g.data4 }
}

fn heap_desc(ty: D3D12_HEAP_TYPE, size: u64, align: u64, flags: D3D12_HEAP_FLAGS) -> D3D12_HEAP_DESC {
    D3D12_HEAP_DESC { SizeInBytes: size, Properties: D3D12_HEAP_PROPERTIES { Type: ty, ..Default::default() }, Alignment: align, Flags: flags }
}

unsafe fn call_heap(f: CreateHeapFn, dev: *mut c_void, desc: *const D3D12_HEAP_DESC, riid: &SysGuid) -> (HRESULT, Option<ID3D12Heap>) {
    let mut out = null_mut();
    let hr = f(dev, desc, riid, &mut out);
    (hr, (hr >= 0 && !out.is_null()).then(|| ID3D12Heap::from_raw(out)))
}

unsafe fn call_heap1(f: CreateHeap1Fn, dev: *mut c_void, desc: *const D3D12_HEAP_DESC, riid: &SysGuid) -> (HRESULT, Option<ID3D12Heap>) {
    let mut out = null_mut();
    let hr = f(dev, desc, null_mut(), riid, &mut out);
    (hr, (hr >= 0 && !out.is_null()).then(|| ID3D12Heap::from_raw(out)))
}

#[test]
fn create_heap_and_heap1_match_host_driver_and_console_fallbacks() {
    let Some(dev) = init_device() else {
        return;
    };
    unsafe {
        let raw_dev = dev.as_raw();
        let slot_heap = slot_of("ID3D12Device", "CreateHeap");
        let hook_heap: CreateHeapFn = vtable_fn(raw_dev, slot_heap);
        let orig_heap: CreateHeapFn = orig_slot_fn(raw_dev, slot_heap);
        let iid_heap = sys_guid(ID3D12Heap::IID);

        let k64 = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64;
        let m4 = D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT as u64;

        let custom_wb = D3D12_HEAP_DESC {
            SizeInBytes: k64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
        };
        // Host-driver parity across standard heap types, flags, alignments, and invalid inputs.
        let cases = [
            ("default_buffers", heap_desc(D3D12_HEAP_TYPE_DEFAULT, k64, 0, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS)),
            ("default_non_rt_tex", heap_desc(D3D12_HEAP_TYPE_DEFAULT, k64, k64, D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES)),
            ("default_rt_ds_tex", heap_desc(D3D12_HEAP_TYPE_DEFAULT, k64, 0, D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES)),
            ("default_all_tier2", heap_desc(D3D12_HEAP_TYPE_DEFAULT, k64, 0, D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES)),
            ("default_msaa_4m", heap_desc(D3D12_HEAP_TYPE_DEFAULT, m4, m4, D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES)),
            ("upload_buffers", heap_desc(D3D12_HEAP_TYPE_UPLOAD, k64, 0, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS)),
            ("readback_buffers", heap_desc(D3D12_HEAP_TYPE_READBACK, k64, 0, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS)),
            ("custom_wb_l0", custom_wb),
            ("zero_size", heap_desc(D3D12_HEAP_TYPE_DEFAULT, 0, 0, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS)),
        ];
        for (name, desc) in cases {
            let (hr_x, heap_x) = call_heap(hook_heap, raw_dev, &desc, &iid_heap);
            let (hr_o, heap_o) = call_heap(orig_heap, raw_dev, &desc, &iid_heap);
            assert_eq!(hr_x, hr_o, "{name}: HRESULT mismatch");
            assert_eq!(heap_x.is_some(), heap_o.is_some(), "{name}: object presence mismatch");
            if let (Some(hx), Some(ho)) = (heap_x, heap_o) {
                assert_eq!(hx.GetDesc(), ho.GetDesc(), "{name}: GetDesc mismatch");
            }
        }

        // NULL ppvHeap capability-probe parity (`S_FALSE` without creating an object).
        // Skip under the D3D12 debug layer (`ID3D12InfoQueue`), whose validation wrapper crashes on null `ppvHeap`.
        let valid = heap_desc(D3D12_HEAP_TYPE_DEFAULT, k64, 0, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS);
        let has_debug_layer = dev.cast::<ID3D12InfoQueue>().is_ok();
        if !has_debug_layer {
            let hr_probe_x = hook_heap(raw_dev, &valid, &iid_heap, null_mut());
            let hr_probe_o = orig_heap(raw_dev, &valid, &iid_heap, null_mut());
            assert_eq!(hr_probe_x, hr_probe_o, "null ppvHeap probe HRESULT mismatch");
        }

        // Console heap fallbacks in `dev_create_heap`: console flags (0x2 / 0x10 masked out),
        // non-standard alignments (<= 64 KiB -> 64 KiB, > 64 KiB -> 4 MiB), unknown heap type,
        // unaligned SizeInBytes rounding, and unknown IID fallback (`ID3D12Heap`).
        let console_cases = [
            ("console_flag_0x2", heap_desc(D3D12_HEAP_TYPE_DEFAULT, k64, 0, D3D12_HEAP_FLAGS(0xC2)), D3D12_HEAP_TYPE_DEFAULT, k64, k64, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS),
            ("console_flag_0x10", heap_desc(D3D12_HEAP_TYPE_DEFAULT, k64, 0, D3D12_HEAP_FLAGS(0xD0)), D3D12_HEAP_TYPE_DEFAULT, k64, k64, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS),
            ("small_align_4k", heap_desc(D3D12_HEAP_TYPE_DEFAULT, k64, 4096, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS), D3D12_HEAP_TYPE_DEFAULT, k64, k64, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS),
            ("large_align_128k", heap_desc(D3D12_HEAP_TYPE_DEFAULT, m4, 0x2_0000, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS), D3D12_HEAP_TYPE_DEFAULT, m4, m4, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS),
            ("unknown_type_unaligned_sz", heap_desc(D3D12_HEAP_TYPE(99), 0x1_2340, 0, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS), D3D12_HEAP_TYPE_DEFAULT, 2 * k64, k64, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS),
        ];
        for (name, desc, want_ty, want_sz, want_align, want_flags) in console_cases {
            let (hr_o, _) = call_heap(orig_heap, raw_dev, &desc, &iid_heap);
            assert!(hr_o < 0, "{name}: expected host driver to reject console heap desc, got {hr_o:#x}");
            let (hr_x, heap_x) = call_heap(hook_heap, raw_dev, &desc, &iid_heap);
            assert!(hr_x >= 0, "{name}: CreateHeap fallback failed: {hr_x:#x}");
            let got = heap_x.expect(name).GetDesc();
            assert_eq!(got.Properties.Type, want_ty, "{name}: Properties.Type");
            assert_eq!(got.SizeInBytes, want_sz, "{name}: SizeInBytes");
            assert_eq!(got.Alignment, want_align, "{name}: Alignment");
            assert_eq!(got.Flags, want_flags, "{name}: Flags");
        }

        let (hr_o, _) = call_heap(orig_heap, raw_dev, &valid, &CONSOLE_UNKNOWN_IID);
        assert_eq!(hr_o, E_NOINTERFACE, "host CreateHeap must reject unknown IID");
        let (hr_x, heap_x) = call_heap(hook_heap, raw_dev, &valid, &CONSOLE_UNKNOWN_IID);
        assert_eq!(hr_x, S_OK, "CreateHeap unknown IID fallback failed: {hr_x:#x}");
        assert_eq!(heap_x.expect("fallback heap").GetDesc().SizeInBytes, k64);

        // `CreateHeap1` (`ID3D12Device4` #54): parity + console IID fallback.
        if let Ok(dev4) = dev.cast::<ID3D12Device4>() {
            let raw4 = dev4.as_raw();
            let slot_heap1 = slot_of("ID3D12Device4", "CreateHeap1");
            let hook_heap1: CreateHeap1Fn = vtable_fn(raw4, slot_heap1);
            let orig_heap1: CreateHeap1Fn = orig_slot_fn(raw4, slot_heap1);

            for (name, desc) in [
                ("heap1_default", heap_desc(D3D12_HEAP_TYPE_DEFAULT, k64, 0, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS)),
                ("heap1_upload", heap_desc(D3D12_HEAP_TYPE_UPLOAD, k64, 0, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS)),
                ("heap1_zero", heap_desc(D3D12_HEAP_TYPE_DEFAULT, 0, 0, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS)),
            ] {
                let (hr_x, hx) = call_heap1(hook_heap1, raw4, &desc, &iid_heap);
                let (hr_o, ho) = call_heap1(orig_heap1, raw4, &desc, &iid_heap);
                assert_eq!(hr_x, hr_o, "{name}: HRESULT mismatch");
                if let (Some(hx), Some(ho)) = (hx, ho) {
                    assert_eq!(hx.GetDesc(), ho.GetDesc(), "{name}: GetDesc mismatch");
                }
            }
            if !has_debug_layer {
                assert_eq!(hook_heap1(raw4, &valid, null_mut(), &iid_heap, null_mut()), orig_heap1(raw4, &valid, null_mut(), &iid_heap, null_mut()));
            }
            let (hr_o1, _) = call_heap1(orig_heap1, raw4, &valid, &CONSOLE_UNKNOWN_IID);
            assert_eq!(hr_o1, E_NOINTERFACE);
            let (hr_x1, hx1) = call_heap1(hook_heap1, raw4, &valid, &CONSOLE_UNKNOWN_IID);
            assert_eq!(hr_x1, S_OK);
            assert_eq!(hx1.expect("heap1 fallback").GetDesc().SizeInBytes, k64);
        }
    }
}

#[test]
fn query_interface_splits_dxgi_device_between_title_and_pc_callers() {
    let Some(dev) = init_device() else {
        return;
    };
    unsafe {
        let raw_dev = dev.as_raw();
        let hook_qi: QiFn = vtable_fn(raw_dev, 0);
        let orig_qi: QiFn = orig_slot_fn(raw_dev, 0);

        // Standard D3D12 interfaces and unknown IID match `orig_fn` for both PC and title callers.
        for (name, guid) in [
            ("IUnknown", sys_guid(IUnknown::IID)),
            ("ID3D12Device", sys_guid(ID3D12Device::IID)),
            ("ID3D12Device1", sys_guid(ID3D12Device1::IID)),
            ("ID3D12Device4", sys_guid(ID3D12Device4::IID)),
            ("CONSOLE_UNKNOWN_IID", CONSOLE_UNKNOWN_IID),
        ] {
            let (mut out_pc, mut out_title, mut out_orig) = (null_mut(), null_mut(), null_mut());
            let hr_pc = pc_qi(hook_qi, raw_dev, &guid, &mut out_pc);
            let hr_title = title_qi(hook_qi, raw_dev, &guid, &mut out_title);
            let hr_orig = orig_qi(raw_dev, &guid, &mut out_orig);
            assert_eq!((hr_pc, hr_title), (hr_orig, hr_orig), "{name}: QI HRESULT mismatch");
            assert_eq!((out_pc, out_title), (out_orig, out_orig), "{name}: QI pointer mismatch");
            for p in [out_pc, out_title, out_orig] {
                if !p.is_null() {
                    drop(IUnknown::from_raw(p));
                }
            }
        }

        // DXGI device interfaces stay E_NOINTERFACE for host/PC callers, while title callers get XCL's `dxgi_device`.
        for (name, guid) in [
            ("IDXGIObject", sys_guid(IDXGIObject::IID)),
            ("IDXGIDevice", sys_guid(IDXGIDevice::IID)),
            ("IDXGIDevice1", sys_guid(IDXGIDevice1::IID)),
            ("IDXGIDevice2", sys_guid(IDXGIDevice2::IID)),
        ] {
            let (mut out_pc, mut out_orig) = (null_mut(), null_mut());
            assert_eq!(pc_qi(hook_qi, raw_dev, &guid, &mut out_pc), E_NOINTERFACE, "{name}: PC caller must not see dxgi_device");
            assert_eq!(orig_qi(raw_dev, &guid, &mut out_orig), E_NOINTERFACE);
            assert!(out_pc.is_null() && out_orig.is_null());

            let mut out_title = null_mut();
            let hr = title_qi(hook_qi, raw_dev, &guid, &mut out_title);
            assert_eq!(hr, S_OK, "{name}: title QI failed: {hr:#x}");
            assert!(!out_title.is_null());
            drop(IUnknown::from_raw(out_title));
        }

        // Exercise the full `IDXGIDevice2` surface returned to a title caller.
        let mut raw_dxgi = null_mut();
        let iid_dxgi2 = sys_guid(IDXGIDevice2::IID);
        assert_eq!(title_qi(hook_qi, raw_dev, &iid_dxgi2, &mut raw_dxgi), S_OK);
        let dxgi2 = IDXGIDevice2::from_raw(raw_dxgi);

        // COM identity & QI back to ID3D12Device / DXGI interfaces / unknown IID.
        let back_dev: ID3D12Device = dxgi2.cast().expect("IDXGIDevice2 -> ID3D12Device");
        assert_eq!(back_dev.as_raw(), raw_dev);
        let _as_dxgi0: IDXGIDevice = dxgi2.cast().expect("IDXGIDevice");
        let _as_dxgi1: IDXGIDevice1 = dxgi2.cast().expect("IDXGIDevice1");
        let mut out_unk = null_mut();
        assert_eq!(dxgi2.query(&CONSOLE_UNKNOWN_IID as *const SysGuid as *const windows::core::GUID, &mut out_unk).0, E_NOINTERFACE);
        assert!(out_unk.is_null());

        // Adapter & GetParent LUID match the D3D12 device's adapter LUID.
        let want_luid = dev.GetAdapterLuid();
        let adapter = dxgi2.GetAdapter().expect("GetAdapter");
        assert_eq!(adapter.GetDesc().expect("GetDesc").AdapterLuid, want_luid);
        let parent_adapter: IDXGIAdapter = dxgi2.GetParent().expect("GetParent(IDXGIAdapter)");
        assert_eq!(parent_adapter.GetDesc().expect("parent GetDesc").AdapterLuid, want_luid);

        // PrivateData forwards to the underlying D3D12 device.
        let priv_guid = windows::core::GUID::from_u128(0xdeadbeef_1234_5678_9abc_001122334455);
        let payload = [0x11u8, 0x22, 0x33, 0x44];
        dxgi2.SetPrivateData(&priv_guid, payload.len() as u32, payload.as_ptr() as *const c_void).expect("SetPrivateData");
        let mut got_priv = [0u8; 4];
        let mut got_sz = got_priv.len() as u32;
        dev.GetPrivateData(&priv_guid, &mut got_sz, Some(got_priv.as_mut_ptr() as *mut c_void)).expect("dev.GetPrivateData");
        assert_eq!((got_sz, got_priv), (4, payload));

        // GPU thread priority bounds [-7, 7].
        assert_eq!(dxgi2.GetGPUThreadPriority().expect("default priority"), 0);
        dxgi2.SetGPUThreadPriority(-7).expect("min priority");
        assert_eq!(dxgi2.GetGPUThreadPriority().expect("read -7"), -7);
        dxgi2.SetGPUThreadPriority(7).expect("max priority");
        assert_eq!(dxgi2.GetGPUThreadPriority().expect("read 7"), 7);
        assert_eq!(dxgi2.SetGPUThreadPriority(-8).unwrap_err().code().0, E_INVALIDARG);
        assert_eq!(dxgi2.SetGPUThreadPriority(8).unwrap_err().code().0, E_INVALIDARG);
        assert_eq!(dxgi2.GetGPUThreadPriority().expect("unchanged after invalid"), 7);

        // Maximum frame latency: default 3, 0 -> 3, >16 clamped to 16.
        assert_eq!(dxgi2.GetMaximumFrameLatency().expect("default latency"), 3);
        dxgi2.SetMaximumFrameLatency(5).expect("latency 5");
        assert_eq!(dxgi2.GetMaximumFrameLatency().expect("read 5"), 5);
        dxgi2.SetMaximumFrameLatency(0).expect("latency 0 resets to 3");
        assert_eq!(dxgi2.GetMaximumFrameLatency().expect("read 3"), 3);
        dxgi2.SetMaximumFrameLatency(25).expect("latency 25 clamps");
        assert_eq!(dxgi2.GetMaximumFrameLatency().expect("read 16"), 16);

        // Residency, OfferResources / ReclaimResources, CreateSurface, EnqueueSetEvent, and E_POINTER checks.
        let mut buf: Option<ID3D12Resource> = None;
        let buf_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 256,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
            ..Default::default()
        };
        let def_props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        dev.CreateCommittedResource(&def_props, D3D12_HEAP_FLAG_NONE, &buf_desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buf).expect("committed buffer");
        let buf_unk: IUnknown = buf.expect("buffer").cast().unwrap();
        let resources = [Some(buf_unk.clone())];
        let mut status = [DXGI_RESIDENCY(0)];
        dxgi2.QueryResourceResidency(resources.as_ptr(), status.as_mut_ptr(), 1).expect("QueryResourceResidency");
        assert_eq!(status[0], DXGI_RESIDENCY_FULLY_RESIDENT);

        let raw2 = dxgi2.as_raw();
        let get_parent: unsafe extern "system" fn(*mut c_void, *const SysGuid, *mut *mut c_void) -> HRESULT = vtable_fn(raw2, 6);
        let get_adapter: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> HRESULT = vtable_fn(raw2, 7);
        let create_surface: unsafe extern "system" fn(*mut c_void, *const c_void, u32, u32, *const c_void, *mut *mut c_void) -> HRESULT = vtable_fn(raw2, 8);
        let query_residency: unsafe extern "system" fn(*mut c_void, *const *mut c_void, *mut u32, u32) -> HRESULT = vtable_fn(raw2, 9);
        let get_prio: unsafe extern "system" fn(*mut c_void, *mut i32) -> HRESULT = vtable_fn(raw2, 11);
        let get_lat: unsafe extern "system" fn(*mut c_void, *mut u32) -> HRESULT = vtable_fn(raw2, 13);
        let offer: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void, i32) -> HRESULT = vtable_fn(raw2, 14);
        let reclaim: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void, *mut i32) -> HRESULT = vtable_fn(raw2, 15);
        let enqueue_event: unsafe extern "system" fn(*mut c_void, *mut c_void) -> HRESULT = vtable_fn(raw2, 16);

        let mut dummy_out = raw2;
        assert_eq!(get_parent(raw2, std::ptr::null(), &mut dummy_out), E_POINTER.0);
        assert_eq!(get_parent(raw2, &sys_guid(IDXGIAdapter::IID), null_mut()), E_POINTER.0);
        assert_eq!(get_adapter(raw2, null_mut()), E_POINTER.0);
        assert_eq!(query_residency(raw2, &buf_unk.as_raw(), null_mut(), 1), E_POINTER.0);
        assert_eq!(get_prio(raw2, null_mut()), E_POINTER.0);
        assert_eq!(get_lat(raw2, null_mut()), E_POINTER.0);

        let res_ptrs = [buf_unk.as_raw()];
        assert_eq!(offer(raw2, 1, res_ptrs.as_ptr(), DXGI_OFFER_RESOURCE_PRIORITY_NORMAL.0), S_OK);
        let mut discarded = [1i32];
        assert_eq!(reclaim(raw2, 1, res_ptrs.as_ptr(), discarded.as_mut_ptr()), S_OK);
        assert_eq!(discarded, [0]);

        let mut surface = raw2;
        assert_eq!(create_surface(raw2, std::ptr::null(), 1, 0, std::ptr::null(), &mut surface), E_NOTIMPL);
        assert!(surface.is_null());

        let ev = CreateEventW(std::ptr::null(), 0, 0, std::ptr::null());
        assert!(!ev.is_null());
        assert_eq!(enqueue_event(raw2, ev), E_NOTIMPL);
        CloseHandle(ev);
    }
}
