//! Host D3D12 driver parity and console descriptor-heap contract tests for XCL's
//! `ID3D12Device` view, sampler, and descriptor-copy hooks (slots 17–24; part of #480):
//! `CreateConstantBufferView`, `CreateShaderResourceView`, `CreateUnorderedAccessView`,
//! `CreateRenderTargetView`, `CreateDepthStencilView`, `CreateSampler`, `CopyDescriptors`,
//! and `CopyDescriptorsSimple`.

use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::{Arc, Once};

use windows::core::{Interface, GUID, HRESULT};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows_sys::Win32::System::Diagnostics::Debug::RtlLookupFunctionEntry;

use xcl::gpu::desc::{self, ImageType};
use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::{self, console_heap, real};
use xcl::modules::{self, MappedModule};

const XBOX_EXTRA_CBV_SRV_UAV: u32 = 4;

const SLOT_CREATE_DESCRIPTOR_HEAP: usize = 14;
const SLOT_GET_INCREMENT_SIZE: usize = 15;
const SLOT_CREATE_CBV: usize = 17;
const SLOT_CREATE_SRV: usize = 18;
const SLOT_CREATE_UAV: usize = 19;
const SLOT_CREATE_RTV: usize = 20;
const SLOT_CREATE_DSV: usize = 21;
const SLOT_CREATE_SAMPLER: usize = 22;
const SLOT_COPY_DESCRIPTORS: usize = 23;
const SLOT_COPY_DESCRIPTORS_SIMPLE: usize = 24;
const SLOT_CREATE_COMMITTED_RESOURCE: usize = 27;

const DXGI_FORMAT_B5G5R5A1_CONSOLE: DXGI_FORMAT = DXGI_FORMAT(86);
const DXGI_FORMAT_B4G4R4A4_CONSOLE: DXGI_FORMAT = DXGI_FORMAT(115);

type CreateHeapFn = unsafe extern "system" fn(*mut c_void, *const [u32; 4], *const GUID, *mut *mut c_void) -> HRESULT;
type IncrementSizeFn = unsafe extern "system" fn(*mut c_void, u32) -> u32;
type CreateCbvFn = unsafe extern "system" fn(*mut c_void, *const D3D12_CONSTANT_BUFFER_VIEW_DESC, u64);
type CreateSrvFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_SHADER_RESOURCE_VIEW_DESC, u64);
type CreateUavFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *const D3D12_UNORDERED_ACCESS_VIEW_DESC, u64);
type CreateRtvFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_RENDER_TARGET_VIEW_DESC, usize);
type CreateDsvFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_DEPTH_STENCIL_VIEW_DESC, u64);
type CreateSamplerFn = unsafe extern "system" fn(*mut c_void, *const D3D12_SAMPLER_DESC, u64);
type CopyRangesFn = unsafe extern "system" fn(*mut c_void, u32, *const u64, *const u32, u32, *const u64, *const u32, u32);
type CopySimpleFn = unsafe extern "system" fn(*mut c_void, u32, u64, u64, u32);
type CreateCommittedFn = unsafe extern "system" fn(
    *mut c_void,
    *const D3D12_HEAP_PROPERTIES,
    D3D12_HEAP_FLAGS,
    *const D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_STATES,
    *const D3D12_CLEAR_VALUE,
    *const GUID,
    *mut *mut c_void,
) -> HRESULT;
type ConsoleStartFn = unsafe extern "system" fn(*mut c_void) -> u64;

#[inline(never)]
unsafe fn title_create_heap(f: CreateHeapFn, dev: *mut c_void, desc: &[u32; 4], out: &mut *mut c_void) -> HRESULT {
    let hr = f(dev, desc, &ID3D12DescriptorHeap::IID, out);
    std::hint::black_box(hr)
}

#[inline(never)]
unsafe fn title_increment_size(f: IncrementSizeFn, dev: *mut c_void, ty: u32) -> u32 {
    let inc = f(dev, ty);
    std::hint::black_box(inc)
}

#[inline(never)]
unsafe fn title_create_cbv(f: CreateCbvFn, dev: *mut c_void, desc: *const D3D12_CONSTANT_BUFFER_VIEW_DESC, cpu: u64) {
    f(dev, desc, cpu);
    std::hint::black_box(0u8);
}

#[inline(never)]
unsafe fn title_create_srv(f: CreateSrvFn, dev: *mut c_void, res: *mut c_void, desc: *const D3D12_SHADER_RESOURCE_VIEW_DESC, cpu: u64) {
    f(dev, res, desc, cpu);
    std::hint::black_box(0u8);
}

#[inline(never)]
unsafe fn title_create_uav(
    f: CreateUavFn,
    dev: *mut c_void,
    res: *mut c_void,
    counter: *mut c_void,
    desc: *const D3D12_UNORDERED_ACCESS_VIEW_DESC,
    cpu: u64,
) {
    f(dev, res, counter, desc, cpu);
    std::hint::black_box(0u8);
}

#[inline(never)]
unsafe fn title_create_dsv(f: CreateDsvFn, dev: *mut c_void, res: *mut c_void, desc: *const D3D12_DEPTH_STENCIL_VIEW_DESC, cpu: u64) {
    f(dev, res, desc, cpu);
    std::hint::black_box(0u8);
}

#[inline(never)]
unsafe fn title_create_sampler(f: CreateSamplerFn, dev: *mut c_void, desc: *const D3D12_SAMPLER_DESC, cpu: u64) {
    f(dev, desc, cpu);
    std::hint::black_box(0u8);
}

#[inline(never)]
unsafe fn title_copy_simple(f: CopySimpleFn, dev: *mut c_void, n: u32, dst: u64, src: u64, ty: u32) {
    f(dev, n, dst, src, ty);
    std::hint::black_box(0u8);
}

#[inline(never)]
unsafe fn title_copy_ranges(
    f: CopyRangesFn,
    dev: *mut c_void,
    nd: u32,
    dsts: *const u64,
    dsizes: *const u32,
    ns: u32,
    srcs: *const u64,
    ssizes: *const u32,
    ty: u32,
) {
    f(dev, nd, dsts, dsizes, ns, srcs, ssizes, ty);
    std::hint::black_box(0u8);
}

#[inline(never)]
unsafe fn title_create_committed(
    f: CreateCommittedFn,
    dev: *mut c_void,
    props: &D3D12_HEAP_PROPERTIES,
    flags: D3D12_HEAP_FLAGS,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
    clear: *const D3D12_CLEAR_VALUE,
    out: &mut *mut c_void,
) -> HRESULT {
    let hr = f(dev, props, flags, desc, state, clear, &ID3D12Resource::IID, out);
    std::hint::black_box(hr)
}

#[inline(never)]
unsafe fn title_heap_cpu_start(f: ConsoleStartFn, heap: *mut c_void) -> u64 {
    let ptr = f(heap);
    std::hint::black_box(ptr)
}

#[inline(never)]
unsafe fn title_heap_gpu_start(f: ConsoleStartFn, heap: *mut c_void) -> u64 {
    let ptr = f(heap);
    std::hint::black_box(ptr)
}

#[inline(never)]
unsafe fn title_gpu_va(f: ConsoleStartFn, res: *mut c_void) -> u64 {
    let va = f(res);
    std::hint::black_box(va)
}

/// Registers only the `title_*` trampoline code ranges with `modules` once so test bodies remain
/// host callers while calls inside `title_*` route through XCL's console slot handlers.
fn ensure_title_callers() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let callers: [u64; 13] = [
            title_create_heap as *const () as u64,
            title_increment_size as *const () as u64,
            title_create_cbv as *const () as u64,
            title_create_srv as *const () as u64,
            title_create_uav as *const () as u64,
            title_create_dsv as *const () as u64,
            title_create_sampler as *const () as u64,
            title_copy_simple as *const () as u64,
            title_copy_ranges as *const () as u64,
            title_create_committed as *const () as u64,
            title_heap_cpu_start as *const () as u64,
            title_heap_gpu_start as *const () as u64,
            title_gpu_va as *const () as u64,
        ];
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for addr in callers {
            let mut image_base = 0u64;
            let entry = unsafe {
                RtlLookupFunctionEntry(addr, &mut image_base, null_mut())
                    .as_ref()
                    .expect("unwind entry for synthetic title caller")
            };
            let base = (image_base + u64::from(entry.BeginAddress)) as usize;
            let size = (entry.EndAddress - entry.BeginAddress) as usize;
            if !ranges.iter().any(|&(b, _)| b == base) {
                ranges.push((base, size));
            }
        }
        for (i, (base, size)) in ranges.into_iter().enumerate() {
            modules::register_mapped(Arc::new(MappedModule {
                name: format!("d3d12-views-fixture-{i}"),
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
        }
    });
}

fn init_device() -> Option<*mut c_void> {
    if d3d12x::backend() != d3d12x::Backend::Real {
        runner::skip_test("no GPU");
        return None;
    }
    ensure_title_callers();
    Some(real::device_raw())
}

unsafe fn slot_pair<F: Copy>(dev: *mut c_void, slot: usize, name: &str) -> (F, F) {
    assert_eq!(std::mem::size_of::<F>(), std::mem::size_of::<usize>());
    let vt = *(dev as *const *const usize);
    let hooked = *vt.add(slot);
    let host = real::orig_fn(dev, slot);
    assert_ne!(hooked, host, "{name} slot {slot} must be hooked");
    (std::mem::transmute_copy(&hooked), std::mem::transmute_copy(&host))
}

unsafe fn vtable_fn<F: Copy>(obj: *mut c_void, slot: usize) -> F {
    assert_eq!(std::mem::size_of::<F>(), std::mem::size_of::<usize>());
    let vt = *(obj as *const *const usize);
    let f = *vt.add(slot);
    std::mem::transmute_copy(&f)
}

fn buffer_desc(bytes: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

fn tex2d_desc(width: u64, height: u32, array: u16, mips: u16, format: DXGI_FORMAT, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width,
        Height: height,
        DepthOrArraySize: array,
        MipLevels: mips,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: flags,
        ..Default::default()
    }
}

fn default_heap_props() -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }
}

unsafe fn host_committed(dev: &ID3D12Device, desc: &D3D12_RESOURCE_DESC, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
    let mut res: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&default_heap_props(), D3D12_HEAP_FLAG_NONE, desc, state, None, &mut res)
        .expect("CreateCommittedResource");
    res.unwrap()
}

unsafe fn console_committed(
    dev: *mut c_void,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let (hooked, _): (CreateCommittedFn, CreateCommittedFn) =
        slot_pair(dev, SLOT_CREATE_COMMITTED_RESOURCE, "CreateCommittedResource");
    let mut raw = null_mut();
    let hr = title_create_committed(hooked, dev, &default_heap_props(), D3D12_HEAP_FLAG_NONE, desc, state, std::ptr::null(), &mut raw);
    assert!(hr.is_ok() && !raw.is_null(), "console CreateCommittedResource failed: {hr:?}");
    ID3D12Resource::from_raw(raw)
}

unsafe fn pc_heap(dev: &ID3D12Device, ty: D3D12_DESCRIPTOR_HEAP_TYPE, n: u32, flags: D3D12_DESCRIPTOR_HEAP_FLAGS) -> ID3D12DescriptorHeap {
    dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: ty, NumDescriptors: n, Flags: flags, NodeMask: 0 })
        .expect("PC CreateDescriptorHeap")
}

unsafe fn console_heap_pair(dev: *mut c_void, ty: u32, n: u32, visible: bool) -> (ID3D12DescriptorHeap, u64, u64) {
    let (hooked, _): (CreateHeapFn, CreateHeapFn) = slot_pair(dev, SLOT_CREATE_DESCRIPTOR_HEAP, "CreateDescriptorHeap");
    let flags = if visible { D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE.0 as u32 } else { 0 };
    let desc = [ty, n, flags, 0];
    let mut raw = null_mut();
    let hr = title_create_heap(hooked, dev, &desc, &mut raw);
    assert!(hr.is_ok() && !raw.is_null(), "console CreateDescriptorHeap({ty}) failed: {hr:?}");
    let heap = ID3D12DescriptorHeap::from_raw(raw);
    let cpu_fn: ConsoleStartFn = vtable_fn(raw, 9);
    let gpu_fn: ConsoleStartFn = vtable_fn(raw, 10);
    let cpu = title_heap_cpu_start(cpu_fn, raw);
    let gpu = title_heap_gpu_start(gpu_fn, raw);
    assert_eq!(Some(cpu), console_heap::start(raw as usize), "console heap start mismatch");
    assert_eq!(cpu, gpu, "console CPU and GPU heap starts must agree");
    (heap, cpu, gpu)
}

unsafe fn drain_queue(dev: &ID3D12Device, queue: &ID3D12CommandQueue) {
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
    queue.Signal(&fence, 1).unwrap();
    let start = std::time::Instant::now();
    while fence.GetCompletedValue() < 1 {
        assert!(start.elapsed().as_secs() < 10, "GPU fence timed out");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    dev.GetDeviceRemovedReason().expect("device removed after queue execution");
}

fn rel32(uma: &xcl::gpu::uma::Uma, addr: u64) -> u32 {
    u32::try_from((addr as usize).checked_sub(uma.base()).expect("address below UMA base"))
        .expect("console heap must live in window 0 (< 4 GiB)")
}

#[test]
fn pc_views_samplers_and_copies_match_host_driver() {
    let Some(raw_dev) = init_device() else { return };
    unsafe {
        let dev = ID3D12Device::from_raw_borrowed(&raw_dev).expect("backend device");
        let (hook_cbv, host_cbv): (CreateCbvFn, CreateCbvFn) = slot_pair(raw_dev, SLOT_CREATE_CBV, "CreateConstantBufferView");
        let (hook_srv, host_srv): (CreateSrvFn, CreateSrvFn) = slot_pair(raw_dev, SLOT_CREATE_SRV, "CreateShaderResourceView");
        let (hook_uav, host_uav): (CreateUavFn, CreateUavFn) = slot_pair(raw_dev, SLOT_CREATE_UAV, "CreateUnorderedAccessView");
        let (hook_rtv, host_rtv): (CreateRtvFn, CreateRtvFn) = slot_pair(raw_dev, SLOT_CREATE_RTV, "CreateRenderTargetView");
        let (hook_dsv, host_dsv): (CreateDsvFn, CreateDsvFn) = slot_pair(raw_dev, SLOT_CREATE_DSV, "CreateDepthStencilView");
        let (hook_smp, host_smp): (CreateSamplerFn, CreateSamplerFn) = slot_pair(raw_dev, SLOT_CREATE_SAMPLER, "CreateSampler");
        let (hook_copy_r, host_copy_r): (CopyRangesFn, CopyRangesFn) = slot_pair(raw_dev, SLOT_COPY_DESCRIPTORS, "CopyDescriptors");
        let (hook_copy_s, host_copy_s): (CopySimpleFn, CopySimpleFn) =
            slot_pair(raw_dev, SLOT_COPY_DESCRIPTORS_SIMPLE, "CopyDescriptorsSimple");

        let inc_view = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as u64;
        let inc_smp = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) as u64;
        let inc_rtv = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as u64;
        let inc_dsv = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV) as u64;

        let pc_views_src = pc_heap(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8, D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
        let pc_views_dst = pc_heap(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
        let pc_smp_src = pc_heap(dev, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 4, D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
        let pc_smp_dst = pc_heap(dev, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
        let pc_rtv_heap = pc_heap(dev, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
        let pc_dsv_heap = pc_heap(dev, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_NONE);

        let v_src = pc_views_src.GetCPUDescriptorHandleForHeapStart().ptr as u64;
        let v_dst = pc_views_dst.GetCPUDescriptorHandleForHeapStart().ptr as u64;
        let s_src = pc_smp_src.GetCPUDescriptorHandleForHeapStart().ptr as u64;
        let s_dst = pc_smp_dst.GetCPUDescriptorHandleForHeapStart().ptr as u64;
        let rtv0 = pc_rtv_heap.GetCPUDescriptorHandleForHeapStart().ptr;
        let dsv0 = pc_dsv_heap.GetCPUDescriptorHandleForHeapStart().ptr as u64;

        let buf = host_committed(dev, &buffer_desc(512, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_COMMON);
        let tex_uav = host_committed(
            dev,
            &tex2d_desc(16, 16, 1, 1, DXGI_FORMAT_R32_UINT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let tex_rt = host_committed(
            dev,
            &tex2d_desc(16, 16, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        let tex_depth_console = console_committed(
            raw_dev,
            &tex2d_desc(16, 16, 1, 1, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
        );

        let cbv = D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: buf.GetGPUVirtualAddress(), SizeInBytes: 256 };
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV { FirstElement: 0, NumElements: 16, StructureByteStride: 0, Flags: D3D12_BUFFER_SRV_FLAG_RAW },
            },
        };
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_UAV { MipSlice: 0, PlaneSlice: 0 } },
        };
        let smp = D3D12_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
            MaxLOD: f32::MAX,
            ..Default::default()
        };

        // Host-caller path through hooked slots vs unhooked orig_fn on PC heaps.
        host_cbv(raw_dev, &cbv, v_src);
        hook_cbv(raw_dev, &cbv, v_src + inc_view);
        host_srv(raw_dev, buf.as_raw(), &srv, v_src + 2 * inc_view);
        hook_srv(raw_dev, buf.as_raw(), &srv, v_src + 3 * inc_view);
        host_uav(raw_dev, tex_uav.as_raw(), null_mut(), &uav, v_src + 4 * inc_view);
        hook_uav(raw_dev, tex_uav.as_raw(), null_mut(), &uav, v_src + 5 * inc_view);

        host_smp(raw_dev, &smp, s_src);
        hook_smp(raw_dev, &smp, s_src + inc_smp);

        host_copy_s(raw_dev, 1, v_dst, v_src, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32);
        hook_copy_s(raw_dev, 2, v_dst + inc_view, v_src + 4 * inc_view, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32);
        let (d_arr, s_arr, sz_arr) = ([v_dst + 4 * inc_view, v_dst + 5 * inc_view], [v_src, v_src + 5 * inc_view], [1u32, 1u32]);
        host_copy_r(raw_dev, 1, d_arr.as_ptr(), sz_arr.as_ptr(), 1, s_arr.as_ptr(), sz_arr.as_ptr(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32);
        hook_copy_r(raw_dev, 1, &d_arr[1], &sz_arr[1], 1, &s_arr[1], &sz_arr[1], D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32);
        host_copy_s(raw_dev, 1, s_dst, s_src, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32);
        hook_copy_s(raw_dev, 1, s_dst + inc_smp, s_src + inc_smp, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32);

        // RTV (slot 20 override): native format + console B5G5R5A1/B4G4R4A4 on R8G8B8A8_UNORM target.
        host_rtv(raw_dev, tex_rt.as_raw(), std::ptr::null(), rtv0);
        hook_rtv(raw_dev, tex_rt.as_raw(), std::ptr::null(), rtv0 + inc_rtv as usize);
        let rtv_b5g5r5a1 = D3D12_RENDER_TARGET_VIEW_DESC { Format: DXGI_FORMAT_B5G5R5A1_CONSOLE, ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D, ..Default::default() };
        let rtv_b4g4r4a4 = D3D12_RENDER_TARGET_VIEW_DESC { Format: DXGI_FORMAT_B4G4R4A4_CONSOLE, ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D, ..Default::default() };
        hook_rtv(raw_dev, tex_rt.as_raw(), &rtv_b5g5r5a1, rtv0 + 2 * inc_rtv as usize);
        hook_rtv(raw_dev, tex_rt.as_raw(), &rtv_b4g4r4a4, rtv0 + 3 * inc_rtv as usize);

        // DSV (slot 21): host D32_FLOAT + console null-desc default_dsv on typeless depth resource.
        let dsv_d32 = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: DXGI_FORMAT_D32_FLOAT, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D, ..Default::default() };
        host_dsv(raw_dev, tex_depth_console.as_raw(), &dsv_d32, dsv0);
        hook_dsv(raw_dev, tex_depth_console.as_raw(), &dsv_d32, dsv0 + inc_dsv);
        title_create_dsv(hook_dsv, raw_dev, tex_depth_console.as_raw(), std::ptr::null(), dsv0 + 2 * inc_dsv);
        title_create_dsv(hook_dsv, raw_dev, tex_depth_console.as_raw(), &dsv_d32, dsv0 + 3 * inc_dsv);

        // UAV (slot 19) on a depth resource lacking ALLOW_UNORDERED_ACCESS: console hook substitutes a null UAV.
        title_create_uav(hook_uav, raw_dev, tex_depth_console.as_raw(), null_mut(), &uav, v_src + 6 * inc_view);

        // Execute clears on a real queue using the descriptors produced by slots 19, 20, 21, and 24.
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
        list.ClearRenderTargetView(D3D12_CPU_DESCRIPTOR_HANDLE { ptr: rtv0 + 2 * inc_rtv as usize }, &[0.25, 0.5, 0.75, 1.0], None);
        list.ClearDepthStencilView(D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (dsv0 + 2 * inc_dsv) as usize }, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
        list.SetDescriptorHeaps(&[Some(pc_views_dst.clone())]);
        let dst_uav_gpu = D3D12_GPU_DESCRIPTOR_HANDLE { ptr: pc_views_dst.GetGPUDescriptorHandleForHeapStart().ptr + 2 * inc_view };
        let src_uav_cpu = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (v_src + 5 * inc_view) as usize };
        list.ClearUnorderedAccessViewUint(dst_uav_gpu, src_uav_cpu, &tex_uav, &[7, 8, 9, 10], &[]);
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        drain_queue(dev, &queue);
    }
}

#[test]
fn console_heap_cbv_srv_uav_and_sampler_write_expected_abi_descriptors() {
    let Some(raw_dev) = init_device() else { return };
    let Some(uma) = xcl::gpu::uma::get() else {
        runner::skip_test("no UMA");
        return;
    };
    unsafe {
        let dev = ID3D12Device::from_raw_borrowed(&raw_dev).expect("backend device");
        let (hook_inc, _): (IncrementSizeFn, IncrementSizeFn) = slot_pair(raw_dev, SLOT_GET_INCREMENT_SIZE, "GetDescriptorHandleIncrementSize");
        let (hook_cbv, _): (CreateCbvFn, CreateCbvFn) = slot_pair(raw_dev, SLOT_CREATE_CBV, "CreateConstantBufferView");
        let (hook_srv, _): (CreateSrvFn, CreateSrvFn) = slot_pair(raw_dev, SLOT_CREATE_SRV, "CreateShaderResourceView");
        let (hook_uav, _): (CreateUavFn, CreateUavFn) = slot_pair(raw_dev, SLOT_CREATE_UAV, "CreateUnorderedAccessView");
        let (hook_smp, _): (CreateSamplerFn, CreateSamplerFn) = slot_pair(raw_dev, SLOT_CREATE_SAMPLER, "CreateSampler");

        assert_eq!(title_increment_size(hook_inc, raw_dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32), console_heap::SLOT);
        assert_eq!(title_increment_size(hook_inc, raw_dev, XBOX_EXTRA_CBV_SRV_UAV), console_heap::SLOT);
        assert_eq!(title_increment_size(hook_inc, raw_dev, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32), console_heap::SAMPLER_SLOT);

        let (view_heap, view_cpu, _) = console_heap_pair(raw_dev, XBOX_EXTRA_CBV_SRV_UAV, 8, true);
        let (smp_heap, smp_cpu, _) = console_heap_pair(raw_dev, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32, 4, true);
        let view_raw = view_heap.as_raw() as usize;
        let smp_raw = smp_heap.as_raw() as usize;
        assert_eq!(uma.window_of(view_cpu as usize), Some(0), "console view heap must live in window 0");
        assert_eq!(uma.window_of(smp_cpu as usize), Some(0), "console sampler heap must live in window 0");

        let vslot = |k: u64| view_cpu + k * console_heap::SLOT as u64;
        let sslot = |k: u64| smp_cpu + k * console_heap::SAMPLER_SLOT as u64;

        // 1) CBV (slot 17): in-UMA address -> cbv_words; non-UMA address -> zeros.
        let uma_va = uma.base() as u64 + 0x1000;
        assert!(uma.contains(uma_va as usize));
        let cbv_in = D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: uma_va, SizeInBytes: 256 };
        let cbv_out = D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: 0x1000_0000, SizeInBytes: 256 };
        title_create_cbv(hook_cbv, raw_dev, &cbv_in, vslot(0));
        title_create_cbv(hook_cbv, raw_dev, &cbv_out, vslot(1));
        assert_eq!(console_heap::words_at(rel32(uma, vslot(0))).unwrap()[..4], desc::cbv_words(uma_va, 256));
        assert_eq!(console_heap::words_at(rel32(uma, vslot(1))).unwrap()[..4], desc::cbv_words(0, 0));

        // 2) SRV (slot 18): console buffer, default Texture2D, TextureCube -> CubeArray, RT AS, null, depth.
        let c_buf = console_committed(raw_dev, &buffer_desc(256, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_COMMON);
        let va_fn: ConsoleStartFn = vtable_fn(c_buf.as_raw(), 11);
        let c_buf_va = title_gpu_va(va_fn, c_buf.as_raw());
        let raw_srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV { FirstElement: 0, NumElements: 16, StructureByteStride: 0, Flags: D3D12_BUFFER_SRV_FLAG_RAW },
            },
        };
        title_create_srv(hook_srv, raw_dev, c_buf.as_raw(), &raw_srv_desc, vslot(2));
        let want_buf_srv = if uma.contains(c_buf_va as usize) {
            desc::buffer_words(c_buf_va, 0, 64, 22)
        } else {
            [0; 4]
        };
        assert_eq!(console_heap::words_at(rel32(uma, vslot(2))).unwrap()[..4], want_buf_srv);

        let tex2d = host_committed(dev, &tex2d_desc(64, 32, 1, 2, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON);
        title_create_srv(hook_srv, raw_dev, tex2d.as_raw(), std::ptr::null(), vslot(3));
        let slot3 = console_heap::slot(vslot(3)).unwrap();
        let fmt_rgba8 = desc::gfx10_format(DXGI_FORMAT_R8G8B8A8_UNORM).unwrap();
        assert_eq!(
            console_heap::words_at(rel32(uma, vslot(3))).unwrap(),
            desc::xcl_image_for_view(slot3.index, fmt_rgba8, ImageType::D2, 64, 32, 1, 2)
        );

        let cube_tex = host_committed(dev, &tex2d_desc(32, 32, 6, 1, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON);
        let cube_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBE,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                TextureCube: D3D12_TEXCUBE_SRV { MostDetailedMip: 0, MipLevels: 1, ResourceMinLODClamp: 0.0 },
            },
        };
        title_create_srv(hook_srv, raw_dev, cube_tex.as_raw(), &cube_srv, vslot(4));
        let slot4 = console_heap::slot(vslot(4)).unwrap();
        assert_eq!(
            console_heap::words_at(rel32(uma, vslot(4))).unwrap(),
            desc::xcl_image_for_view(slot4.index, fmt_rgba8, ImageType::Cube, 32, 32, 6, 1)
        );

        let bvh_in = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                RaytracingAccelerationStructure: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV { Location: uma_va },
            },
        };
        let bvh_out = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                RaytracingAccelerationStructure: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV { Location: 0x2000_0000 },
            },
            ..bvh_in
        };
        title_create_srv(hook_srv, raw_dev, null_mut(), &bvh_in, vslot(5));
        let want_bvh = desc::bvh_words(uma_va, true);
        assert_eq!(
            console_heap::words_at(rel32(uma, vslot(5))).unwrap(),
            [want_bvh[0], want_bvh[1], want_bvh[2], want_bvh[3], 0, 0, 0, 0]
        );
        title_create_srv(hook_srv, raw_dev, null_mut(), &bvh_out, vslot(6));
        assert_eq!(console_heap::words_at(rel32(uma, vslot(6))).unwrap(), [0; 8]);

        let depth = console_committed(
            raw_dev,
            &tex2d_desc(16, 16, 1, 1, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
        );
        title_create_srv(hook_srv, raw_dev, depth.as_raw(), std::ptr::null(), vslot(7));
        let slot7 = console_heap::slot(vslot(7)).unwrap();
        let fmt_r32f = desc::gfx10_format(DXGI_FORMAT_R32_FLOAT).unwrap();
        assert_eq!(
            console_heap::words_at(rel32(uma, vslot(7))).unwrap(),
            desc::xcl_image_for_view(slot7.index, fmt_r32f, ImageType::D2, 16, 16, 1, 1)
        );

        // Null resource SRV zeroes the slot.
        let null_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            ..Default::default()
        };
        title_create_srv(hook_srv, raw_dev, null_mut(), &null_srv, vslot(6));
        assert_eq!(console_heap::words_at(rel32(uma, vslot(6))).unwrap(), [0; 8]);

        // 3) UAV (slot 19): buffer V#, Texture2D mip-scaled T# + GPU clear, and depth null-UAV fallback.
        let (uav_heap, uav_cpu, _) = console_heap_pair(raw_dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32, 4, true);
        let uslot = |k: u64| uav_cpu + k * console_heap::SLOT as u64;
        let raw_uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV { FirstElement: 0, NumElements: 16, StructureByteStride: 0, CounterOffsetInBytes: 0, Flags: D3D12_BUFFER_UAV_FLAG_RAW },
            },
        };
        title_create_uav(hook_uav, raw_dev, c_buf.as_raw(), null_mut(), &raw_uav_desc, uslot(0));
        assert_eq!(console_heap::words_at(rel32(uma, uslot(0))).unwrap()[..4], want_buf_srv);

        let rw_tex = host_committed(
            dev,
            &tex2d_desc(64, 32, 1, 2, DXGI_FORMAT_R32_UINT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let tex_uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_UAV { MipSlice: 1, PlaneSlice: 0 } },
        };
        title_create_uav(hook_uav, raw_dev, rw_tex.as_raw(), null_mut(), &tex_uav_desc, uslot(1));
        let slot_u1 = console_heap::slot(uslot(1)).unwrap();
        let fmt_r32u = desc::gfx10_format(DXGI_FORMAT_R32_UINT).unwrap();
        assert_eq!(
            console_heap::words_at(rel32(uma, uslot(1))).unwrap(),
            desc::xcl_image_for_view(slot_u1.index, fmt_r32u, ImageType::D2, 32, 16, 1, 1)
        );
        if let Some(shared_heap) = console_heap::global_heap() {
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            list.SetDescriptorHeaps(&[Some(shared_heap)]);
            list.ClearUnorderedAccessViewUint(
                D3D12_GPU_DESCRIPTOR_HANDLE { ptr: slot_u1.pc_gpu_uav },
                D3D12_CPU_DESCRIPTOR_HANDLE { ptr: slot_u1.pc_uav as usize },
                &rw_tex,
                &[1, 2, 3, 4],
                &[],
            );
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            drain_queue(dev, &queue);
        }

        // Depth texture lacks ALLOW_UNORDERED_ACCESS: console hook replaces PC call with null UAV.
        let depth_uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        title_create_uav(hook_uav, raw_dev, depth.as_raw(), null_mut(), &depth_uav_desc, uslot(2));
        dev.GetDeviceRemovedReason().expect("depth UAV must not remove device");

        // 4) Samplers (slot 22): S# word 3 [11:0] = sampler slot index.
        let smp_desc = D3D12_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
            MaxLOD: 1.0,
            ..Default::default()
        };
        title_create_sampler(hook_smp, raw_dev, &smp_desc, sslot(0));
        title_create_sampler(hook_smp, raw_dev, &smp_desc, sslot(1));
        assert_eq!(console_heap::words_at(rel32(uma, sslot(0))).unwrap()[..4], [0, 0, 0, 0]);
        assert_eq!(console_heap::words_at(rel32(uma, sslot(1))).unwrap()[..4], [0, 0, 0, 1]);

        drop(uav_heap);
        drop(view_heap);
        drop(smp_heap);
        assert!(console_heap::start(view_raw).is_none(), "released view heap must unregister");
        assert!(console_heap::start(smp_raw).is_none(), "released sampler heap must unregister");
    }
}

#[test]
fn console_copy_descriptors_renumbers_images_and_samplers_and_keeps_buffers() {
    let Some(raw_dev) = init_device() else { return };
    let Some(uma) = xcl::gpu::uma::get() else {
        runner::skip_test("no UMA");
        return;
    };
    unsafe {
        let dev = ID3D12Device::from_raw_borrowed(&raw_dev).expect("backend device");
        let (hook_cbv, _): (CreateCbvFn, CreateCbvFn) = slot_pair(raw_dev, SLOT_CREATE_CBV, "CreateConstantBufferView");
        let (hook_srv, _): (CreateSrvFn, CreateSrvFn) = slot_pair(raw_dev, SLOT_CREATE_SRV, "CreateShaderResourceView");
        let (hook_smp, _): (CreateSamplerFn, CreateSamplerFn) = slot_pair(raw_dev, SLOT_CREATE_SAMPLER, "CreateSampler");
        let (hook_copy_r, _): (CopyRangesFn, CopyRangesFn) = slot_pair(raw_dev, SLOT_COPY_DESCRIPTORS, "CopyDescriptors");
        let (hook_copy_s, _): (CopySimpleFn, CopySimpleFn) =
            slot_pair(raw_dev, SLOT_COPY_DESCRIPTORS_SIMPLE, "CopyDescriptorsSimple");

        let (_src_heap, src_cpu, _) = console_heap_pair(raw_dev, XBOX_EXTRA_CBV_SRV_UAV, 6, false);
        let (_dst_heap, dst_cpu, _) = console_heap_pair(raw_dev, XBOX_EXTRA_CBV_SRV_UAV, 6, true);
        let (_s_src_heap, s_src_cpu, _) = console_heap_pair(raw_dev, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32, 4, false);
        let (_s_dst_heap, s_dst_cpu, _) = console_heap_pair(raw_dev, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32, 4, true);

        let src = |k: u64| src_cpu + k * console_heap::SLOT as u64;
        let dst = |k: u64| dst_cpu + k * console_heap::SLOT as u64;
        let s_src = |k: u64| s_src_cpu + k * console_heap::SAMPLER_SLOT as u64;
        let s_dst = |k: u64| s_dst_cpu + k * console_heap::SAMPLER_SLOT as u64;

        let tex = host_committed(dev, &tex2d_desc(32, 16, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON);
        let cube = host_committed(dev, &tex2d_desc(16, 16, 6, 1, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON);
        let uma_va = uma.base() as u64 + 0x2000;

        title_create_srv(hook_srv, raw_dev, tex.as_raw(), std::ptr::null(), src(0));
        title_create_cbv(hook_cbv, raw_dev, &D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: uma_va, SizeInBytes: 128 }, src(1));
        let bvh_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                RaytracingAccelerationStructure: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV { Location: uma_va },
            },
        };
        title_create_srv(hook_srv, raw_dev, null_mut(), &bvh_srv, src(2));
        let cube_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBE,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                TextureCube: D3D12_TEXCUBE_SRV { MostDetailedMip: 0, MipLevels: 1, ResourceMinLODClamp: 0.0 },
            },
        };
        title_create_srv(hook_srv, raw_dev, cube.as_raw(), &cube_srv, src(3));

        let src_t0 = console_heap::words_at(rel32(uma, src(0))).unwrap();
        let src_v1 = console_heap::words_at(rel32(uma, src(1))).unwrap();
        let src_bvh2 = console_heap::words_at(rel32(uma, src(2))).unwrap();
        let src_cube3 = console_heap::words_at(rel32(uma, src(3))).unwrap();

        // CopyDescriptorsSimple over a 2-slot span (T# image + CBV buffer) using XBOX_EXTRA_CBV_SRV_UAV.
        title_copy_simple(hook_copy_s, raw_dev, 2, dst(2), src(0), XBOX_EXTRA_CBV_SRV_UAV);
        let dst2_slot = console_heap::slot(dst(2)).unwrap();
        let dst2 = console_heap::words_at(rel32(uma, dst(2))).unwrap();
        assert_eq!(dst2[0], dst2_slot.index, "copied T# must hold destination PC SRV index");
        assert_eq!(dst2[1..], src_t0[1..], "copied T# metadata words must stay intact");
        assert_eq!(console_heap::words_at(rel32(uma, dst(3))).unwrap(), src_v1, "copied CBV V# must stay verbatim");

        // CopyDescriptors with multiple ranges and explicit sizes (BVH + TextureCube).
        let dst_ranges = [dst(0), dst(4)];
        let src_ranges = [src(2), src(3)];
        let sizes = [1u32, 1u32];
        title_copy_ranges(
            hook_copy_r,
            raw_dev,
            2,
            dst_ranges.as_ptr(),
            sizes.as_ptr(),
            2,
            src_ranges.as_ptr(),
            sizes.as_ptr(),
            XBOX_EXTRA_CBV_SRV_UAV,
        );
        assert_eq!(console_heap::words_at(rel32(uma, dst(0))).unwrap(), src_bvh2, "copied BVH SRSRC must stay verbatim");
        let dst4_slot = console_heap::slot(dst(4)).unwrap();
        let dst4 = console_heap::words_at(rel32(uma, dst(4))).unwrap();
        assert_eq!(dst4[0], dst4_slot.index, "copied cube T# must hold destination PC SRV index");
        assert_eq!(dst4[1..], src_cube3[1..]);

        // CopyDescriptors with NULL range-size pointers (each range defaults to 1 descriptor).
        title_copy_ranges(
            hook_copy_r,
            raw_dev,
            1,
            &dst(5),
            std::ptr::null(),
            1,
            &src(0),
            std::ptr::null(),
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32,
        );
        let dst5_slot = console_heap::slot(dst(5)).unwrap();
        let dst5 = console_heap::words_at(rel32(uma, dst(5))).unwrap();
        assert_eq!(dst5[0], dst5_slot.index);
        assert_eq!(dst5[1..], src_t0[1..]);

        // Sampler CopyDescriptorsSimple and CopyDescriptors: S# word 3 [11:0] renumbered to destination slot.
        let smp_desc = D3D12_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
            ..Default::default()
        };
        title_create_sampler(hook_smp, raw_dev, &smp_desc, s_src(0));
        title_create_sampler(hook_smp, raw_dev, &smp_desc, s_src(1));
        title_copy_simple(hook_copy_s, raw_dev, 2, s_dst(1), s_src(0), D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32);
        assert_eq!(console_heap::words_at(rel32(uma, s_dst(1))).unwrap()[..4], [0, 0, 0, 1]);
        assert_eq!(console_heap::words_at(rel32(uma, s_dst(2))).unwrap()[..4], [0, 0, 0, 2]);
        title_copy_ranges(
            hook_copy_r,
            raw_dev,
            1,
            &s_dst(3),
            std::ptr::null(),
            1,
            &s_src(0),
            std::ptr::null(),
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER.0 as u32,
        );
        assert_eq!(console_heap::words_at(rel32(uma, s_dst(3))).unwrap()[..4], [0, 0, 0, 3]);

        // Mixed console/PC copy must not crash or overwrite the destination console descriptor.
        let pc_non_sv = pc_heap(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
        let pc_h = pc_non_sv.GetCPUDescriptorHandleForHeapStart().ptr as u64;
        title_copy_simple(hook_copy_s, raw_dev, 1, dst(2), pc_h, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32);
        title_copy_simple(hook_copy_s, raw_dev, 1, pc_h, src(0), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32);
        title_copy_ranges(
            hook_copy_r,
            raw_dev,
            1,
            &pc_h,
            std::ptr::null(),
            1,
            &src(0),
            std::ptr::null(),
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0 as u32,
        );
        assert_eq!(console_heap::words_at(rel32(uma, dst(2))).unwrap(), dst2);
        dev.GetDeviceRemovedReason().expect("mixed descriptor copy must not remove device");
    }
}
