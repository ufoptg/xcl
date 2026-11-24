//! Host-driver parity, live GPU, and contract tests for `ID3D12Device` pipeline hooks
//! (`CreateGraphicsPipelineState` #10, `CreateComputePipelineState` #11/#62,
//! `ID3D12Device2::CreatePipelineState` #47/#86) and Xbox console device extensions
//! (`CreateGraphicsPipelineStateX` #60, `CreateDerivedPipelineStateX` #55,
//! `SerializePipelineStateX` #56/#66, `Deserialize{Graphics,Compute}PipelineStateX` #57/#67,
//! `SetDebugFlagsX` #52, `RegisterPagePoolX` #74, `UnregisterPagePoolX` #75) without game files
//! (Part of #480).

use std::ffi::c_void;
use std::mem::{offset_of, size_of};
use std::ptr::{null, null_mut};
use std::sync::{Arc, Once};

use windows::core::{Interface, PCSTR};
use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows_sys::core::GUID as SysGuid;
use windows_sys::Win32::System::Diagnostics::Debug::RtlLookupFunctionEntry;

use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::{self, layout, real};
use xcl::modules::{self, MappedModule};
use xcl::util::{E_INVALIDARG, E_NOINTERFACE, HRESULT, S_OK};

const E_ACCESSDENIED: HRESULT = 0x8007_0005u32 as i32;
const E_NOTIMPL: HRESULT = 0x8000_4001u32 as i32;
const E_FAIL: HRESULT = 0x8000_4005u32 as i32;
const CONSOLE_UNKNOWN_IID: SysGuid = SysGuid::from_u128(0x33445566_7788_99aa_bbcc_ddeeff001122);
const PATCH_STRIDE: usize = 0x14C;

type Call2 = unsafe extern "system" fn(*mut c_void, usize) -> HRESULT;
type Call4 = unsafe extern "system" fn(*mut c_void, usize, usize, usize) -> HRESULT;
type Call6 = unsafe extern "system" fn(*mut c_void, usize, usize, usize, usize, usize) -> HRESULT;

#[inline(never)]
unsafe fn title_call2(f: Call2, dev: *mut c_void, a1: usize) -> HRESULT {
    let tag = std::hint::black_box(0x5449_5402u32);
    let r = f(dev, a1);
    std::hint::black_box((r, tag)).0
}

#[inline(never)]
unsafe fn title_call4(f: Call4, dev: *mut c_void, a1: usize, a2: usize, a3: usize) -> HRESULT {
    let tag = std::hint::black_box(0x5449_5404u32);
    let r = f(dev, a1, a2, a3);
    std::hint::black_box((r, tag)).0
}

#[inline(never)]
unsafe fn title_call6(f: Call6, dev: *mut c_void, a1: usize, a2: usize, a3: usize, a4: usize, a5: usize) -> HRESULT {
    let tag = std::hint::black_box(0x5449_5406u32);
    let r = f(dev, a1, a2, a3, a4, a5);
    std::hint::black_box((r, tag)).0
}

#[inline(never)]
unsafe fn pc_call4(f: Call4, dev: *mut c_void, a1: usize, a2: usize, a3: usize) -> HRESULT {
    let tag = std::hint::black_box(0x5043_5F04u32);
    let r = f(dev, a1, a2, a3);
    std::hint::black_box((r, tag)).0
}

fn fn_range(addr: u64) -> (usize, usize) {
    let mut base = 0u64;
    let entry = unsafe { RtlLookupFunctionEntry(addr, &mut base, null_mut()).as_ref().expect("RUNTIME_FUNCTION") };
    ((base + u64::from(entry.BeginAddress)) as usize, (entry.EndAddress - entry.BeginAddress) as usize)
}

/// Registers synthetic mapped-module ranges for `title_call*` so `guard::make_split` sends
/// their calls to console slots while `pc_call4` keeps taking the PC path.
fn ensure_title_callers() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let pc = fn_range(pc_call4 as *const () as u64);
        for (i, addr) in [title_call2 as *const () as u64, title_call4 as *const () as u64, title_call6 as *const () as u64]
            .into_iter()
            .enumerate()
        {
            let (base, size) = fn_range(addr);
            assert_ne!(base, pc.0, "title trampoline must stay distinct from pc_call4");
            modules::register_mapped(Arc::new(MappedModule {
                name: format!("d3d12-pso-fixture-{i}"),
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
    let iface = layout::find(iface_name).unwrap_or_else(|| panic!("missing {iface_name}"));
    layout::slot(iface, method).unwrap_or_else(|| panic!("missing {iface_name}::{method}"))
}

unsafe fn vtable_fn<F: Copy>(obj: *mut c_void, slot: usize) -> F {
    assert_eq!(size_of::<F>(), size_of::<usize>());
    let addr = *(*(obj as *const *const usize)).add(slot);
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

fn compile_dxbc(src: &str, target: &str) -> Vec<u8> {
    let t = format!("{target}\0");
    let mut blob = None;
    let mut err = None;
    unsafe {
        let r = D3DCompile(src.as_ptr() as *const c_void, src.len(), PCSTR::null(), None, None, PCSTR(b"main\0".as_ptr()), PCSTR(t.as_ptr()), 0, 0, &mut blob, Some(&mut err));
        if let Err(e) = r {
            let msg = err.map(|b| String::from_utf8_lossy(std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize())).into_owned()).unwrap_or_default();
            panic!("D3DCompile({target}) failed: {e}: {msg}");
        }
        let b = blob.expect("compiled shader blob");
        std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize()).to_vec()
    }
}

fn root_sig(dev: &ID3D12Device, with_uav0: bool, flags: D3D12_ROOT_SIGNATURE_FLAGS) -> ID3D12RootSignature {
    let param = D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
        Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: if with_uav0 { 1 } else { 0 },
        pParameters: if with_uav0 { &param } else { null() },
        Flags: flags,
        ..Default::default()
    };
    let mut ser = None;
    unsafe {
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).expect("serialize root sig");
        let b = ser.unwrap();
        dev.CreateRootSignature(0, std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize())).expect("CreateRootSignature")
    }
}

fn bc(bytes: &[u8]) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr() as *const c_void, BytecodeLength: bytes.len() }
}

fn base_gfx_desc(rs: &ID3D12RootSignature, vs: &[u8], ps: &[u8], dsv: DXGI_FORMAT) -> D3D12_GRAPHICS_PIPELINE_STATE_DESC {
    let mut d = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: unsafe { std::mem::transmute_copy(rs) },
        VS: bc(vs),
        PS: bc(ps),
        SampleMask: u32::MAX,
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        DSVFormat: dsv,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    d.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    d.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
    d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    d.RasterizerState.DepthClipEnable = true.into();
    d
}

#[repr(C, align(8))]
struct Sub<T> {
    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
    v: T,
}

#[repr(C)]
struct GfxStream {
    rs: Sub<*mut c_void>,
    vs: Sub<D3D12_SHADER_BYTECODE>,
    ps: Sub<D3D12_SHADER_BYTECODE>,
    blend: Sub<D3D12_BLEND_DESC>,
    mask: Sub<u32>,
    rast: Sub<D3D12_RASTERIZER_DESC>,
    depth: Sub<D3D12_DEPTH_STENCIL_DESC>,
    topo: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
    rtv: Sub<D3D12_RT_FORMAT_ARRAY>,
    samples: Sub<DXGI_SAMPLE_DESC>,
}

#[repr(C)]
struct ComputeStream {
    rs: Sub<*mut c_void>,
    cs: Sub<D3D12_SHADER_BYTECODE>,
    node: Sub<u32>,
}

fn assert_cached_blob(pso: &ID3D12PipelineState, label: &str) {
    let blob = unsafe { pso.GetCachedBlob() }.unwrap_or_else(|e| panic!("{label}: GetCachedBlob: {e}"));
    assert!(unsafe { blob.GetBufferSize() } > 0, "{label}: empty cached blob");
}

unsafe fn wait_fence(queue: &ID3D12CommandQueue, fence: &ID3D12Fence, val: &mut u64) {
    *val += 1;
    let target = *val;
    queue.Signal(fence, target).expect("Signal");
    if fence.GetCompletedValue() < target {
        let ev = CreateEventW(None, false, false, None).expect("CreateEventW");
        let r = fence.SetEventOnCompletion(target, ev);
        let wait = WaitForSingleObject(ev, 10_000);
        let _ = CloseHandle(ev);
        r.expect("SetEventOnCompletion");
        assert_eq!(wait, WAIT_OBJECT_0, "GPU fence timeout");
    }
}

unsafe fn committed_buf(dev: &ID3D12Device, heap: D3D12_HEAP_TYPE, bytes: u64, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    let mut r = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).expect("buffer");
    r.unwrap()
}

fn transition(res: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(res) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

unsafe fn run_compute_u32(dev: &ID3D12Device, rs: &ID3D12RootSignature, pso: &ID3D12PipelineState) -> u32 {
    let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, pso).unwrap();
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
    let out = committed_buf(dev, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    let rb = committed_buf(dev, D3D12_HEAP_TYPE_READBACK, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

    list.ResourceBarrier(&[transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
    list.SetComputeRootSignature(rs);
    list.SetComputeRootUnorderedAccessView(0, out.GetGPUVirtualAddress());
    list.Dispatch(1, 1, 1);
    list.ResourceBarrier(&[transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
    list.CopyBufferRegion(&rb, 0, &out, 0, 4);
    list.Close().unwrap();
    queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
    let mut fv = 0;
    wait_fence(&queue, &fence, &mut fv);

    let mut p = null_mut();
    rb.Map(0, None, Some(&mut p)).unwrap();
    let v = *(p as *const u32);
    rb.Unmap(0, None);
    v
}

struct DrawTarget {
    queue: ID3D12CommandQueue,
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    fv: u64,
    rt: ID3D12Resource,
    rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
    _heaps: [ID3D12DescriptorHeap; 2],
    _ds: ID3D12Resource,
    rb: ID3D12Resource,
    fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
}

impl DrawTarget {
    unsafe fn new(dev: &ID3D12Device) -> Self {
        let rt_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 4,
            Height: 4,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let ds_desc = D3D12_RESOURCE_DESC { Format: DXGI_FORMAT_D32_FLOAT, Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, ..rt_desc };
        let hp = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let rt_cv = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_R32G32B32A32_FLOAT, Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0; 4] } };
        let ds_cv = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_D32_FLOAT, Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 } } };
        let (mut rt, mut ds) = (None, None);
        dev.CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rt_desc, D3D12_RESOURCE_STATE_COMMON, Some(&rt_cv), &mut rt).unwrap();
        dev.CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &ds_desc, D3D12_RESOURCE_STATE_DEPTH_WRITE, Some(&ds_cv), &mut ds).unwrap();
        let (rt, ds) = (rt.unwrap(), ds.unwrap());
        let rtv_h: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() }).unwrap();
        let dsv_h: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 1, ..Default::default() }).unwrap();
        let (rtv, dsv) = (rtv_h.GetCPUDescriptorHandleForHeapStart(), dsv_h.GetCPUDescriptorHandleForHeapStart());
        dev.CreateRenderTargetView(&rt, None, rtv);
        dev.CreateDepthStencilView(&ds, None, dsv);
        let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0u64;
        dev.GetCopyableFootprints(&rt_desc, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
        let rb = committed_buf(dev, D3D12_HEAP_TYPE_READBACK, total, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
        list.Close().unwrap();
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        Self { queue, alloc, list, fence, fv: 0, rt, rtv, dsv, _heaps: [rtv_h, dsv_h], _ds: ds, rb, fp }
    }

    unsafe fn draw_pixel(&mut self, rs: &ID3D12RootSignature, pso: &ID3D12PipelineState) -> [f32; 4] {
        self.alloc.Reset().unwrap();
        self.list.Reset(&self.alloc, pso).unwrap();
        self.list.ResourceBarrier(&[transition(&self.rt, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET)]);
        self.list.ClearRenderTargetView(self.rtv, &[0.0; 4], None);
        self.list.ClearDepthStencilView(self.dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
        self.list.OMSetRenderTargets(1, Some(&self.rtv), false, Some(&self.dsv));
        self.list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: 4.0, Height: 4.0, MinDepth: 0.0, MaxDepth: 1.0 }]);
        self.list.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: 4, bottom: 4 }]);
        self.list.SetGraphicsRootSignature(rs);
        self.list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        self.list.DrawInstanced(3, 1, 0, 0);
        self.list.ResourceBarrier(&[transition(&self.rt, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&self.rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: self.fp } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&self.rt), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        self.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        self.list.ResourceBarrier(&[transition(&self.rt, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
        self.list.Close().unwrap();
        self.queue.ExecuteCommandLists(&[Some(self.list.cast().unwrap())]);
        wait_fence(&self.queue, &self.fence, &mut self.fv);

        let mut p = null_mut();
        self.rb.Map(0, None, Some(&mut p)).unwrap();
        let px = *(p as *const [f32; 4]);
        self.rb.Unmap(0, None);
        px
    }
}

fn patch_bytes<T>(part: u32, value: &T) -> [u8; PATCH_STRIDE] {
    let mut buf = [0u8; PATCH_STRIDE];
    buf[0..4].copy_from_slice(&part.to_le_bytes());
    unsafe { std::ptr::copy_nonoverlapping(value as *const T as *const u8, buf[4..].as_mut_ptr(), size_of::<T>()) };
    buf
}

fn fullscreen_shaders() -> (Vec<u8>, Vec<u8>) {
    let vs = compile_dxbc("float4 main(uint id : SV_VertexID) : SV_Position { float2 p = float2((id << 1) & 2, id & 2); return float4(p * float2(2, -2) + float2(-1, 1), 0.5, 1.0); }", "vs_5_0");
    let ps = compile_dxbc("float4 main() : SV_Target { return float4(1.0, 0.5, 0.25, 1.0); }", "ps_5_0");
    (vs, ps)
}

#[test]
fn graphics_and_compute_pso_match_host_driver_and_fallbacks() {
    let Some(dev) = init_device() else { return };
    unsafe {
        let raw_dev = dev.as_raw();
        let iid_pso = sys_guid(ID3D12PipelineState::IID);
        let rs_gfx = root_sig(&dev, false, D3D12_ROOT_SIGNATURE_FLAG_NONE);
        let rs_cs = root_sig(&dev, true, D3D12_ROOT_SIGNATURE_FLAG_NONE);
        let (vs, ps) = fullscreen_shaders();
        let cs = compile_dxbc("RWByteAddressBuffer Out : register(u0);\n[numthreads(1, 1, 1)] void main() { Out.Store(0, 0x12345678u); }", "cs_5_0");
        let gfx = base_gfx_desc(&rs_gfx, &vs, &ps, DXGI_FORMAT_UNKNOWN);
        let cmp = D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: std::mem::transmute_copy(&rs_cs), CS: bc(&cs), ..Default::default() };

        let slot_gfx = slot_of("ID3D12Device", "CreateGraphicsPipelineState");
        let slot_cmp = slot_of("ID3D12Device", "CreateComputePipelineState");
        let slot_stream = slot_of("ID3D12Device2", "CreatePipelineState");
        let (hook_gfx, orig_gfx): (Call4, Call4) = (vtable_fn(raw_dev, slot_gfx), orig_slot_fn(raw_dev, slot_gfx));
        let (hook_cmp, orig_cmp): (Call4, Call4) = (vtable_fn(raw_dev, slot_cmp), orig_slot_fn(raw_dev, slot_cmp));
        let (hook_stream, orig_stream): (Call4, Call4) = (vtable_fn(raw_dev, slot_stream), orig_slot_fn(raw_dev, slot_stream));
        let gfx_x: Call6 = vtable_fn(raw_dev, 60);
        let cmp_x: Call6 = vtable_fn(raw_dev, 62);
        let stream_x: Call4 = vtable_fn(raw_dev, 86);

        // Graphics PSO: orig vs PC slot 10 vs title slot 10 vs console slot 60
        let mut p_orig = null_mut();
        let mut p_pc = null_mut();
        let mut p_title = null_mut();
        let mut p_x0 = null_mut();
        let mut p_x1 = null_mut();
        assert_eq!(orig_gfx(raw_dev, &gfx as *const _ as usize, &iid_pso as *const _ as usize, &mut p_orig as *mut _ as usize), S_OK);
        assert_eq!(pc_call4(hook_gfx, raw_dev, &gfx as *const _ as usize, &iid_pso as *const _ as usize, &mut p_pc as *mut _ as usize), S_OK);
        assert_eq!(title_call4(hook_gfx, raw_dev, &gfx as *const _ as usize, &iid_pso as *const _ as usize, &mut p_title as *mut _ as usize), S_OK);
        assert_eq!(title_call6(gfx_x, raw_dev, &gfx as *const _ as usize, 0, 0, &iid_pso as *const _ as usize, &mut p_x0 as *mut _ as usize), S_OK);
        assert_eq!(title_call6(gfx_x, raw_dev, &gfx as *const _ as usize, 2, 0, &iid_pso as *const _ as usize, &mut p_x1 as *mut _ as usize), S_OK);
        for (p, label) in [(p_orig, "orig gfx"), (p_pc, "pc gfx #10"), (p_title, "title gfx #10"), (p_x0, "gfx_x #60"), (p_x1, "gfx_x #60 ext")] {
            let pso = ID3D12PipelineState::from_raw(p);
            assert_cached_blob(&pso, label);
        }

        // Invalid-descriptor parity on CreateGraphicsPipelineState and CreateComputePipelineState
        // (both drivers return E_INVALIDARG without triggering fallback when root sig & shaders are missing)
        let mut dummy_out: *mut c_void = null_mut();
        let bad_gfx = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        let bad_cmp = D3D12_COMPUTE_PIPELINE_STATE_DESC::default();
        for (orig, hook, bad_ptr, name) in [
            (orig_gfx, hook_gfx, &bad_gfx as *const _ as usize, "gfx"),
            (orig_cmp, hook_cmp, &bad_cmp as *const _ as usize, "cmp"),
        ] {
            let o1 = orig(raw_dev, bad_ptr, &iid_pso as *const _ as usize, &mut dummy_out as *mut _ as usize);
            assert_eq!(pc_call4(hook, raw_dev, bad_ptr, &iid_pso as *const _ as usize, &mut dummy_out as *mut _ as usize), o1, "{name} empty desc");
        }

        // Compute PSO: orig vs PC slot 11 vs console slot 62, executed on GPU
        let (mut c_orig, mut c_hook, mut c_x0, mut c_x1) = (null_mut(), null_mut(), null_mut(), null_mut());
        assert_eq!(orig_cmp(raw_dev, &cmp as *const _ as usize, &iid_pso as *const _ as usize, &mut c_orig as *mut _ as usize), S_OK);
        assert_eq!(pc_call4(hook_cmp, raw_dev, &cmp as *const _ as usize, &iid_pso as *const _ as usize, &mut c_hook as *mut _ as usize), S_OK);
        assert_eq!(title_call6(cmp_x, raw_dev, &cmp as *const _ as usize, 0, 0, &iid_pso as *const _ as usize, &mut c_x0 as *mut _ as usize), S_OK);
        assert_eq!(title_call6(cmp_x, raw_dev, &cmp as *const _ as usize, 7, 9, &iid_pso as *const _ as usize, &mut c_x1 as *mut _ as usize), S_OK);
        for (p, label) in [(c_orig, "orig cs"), (c_hook, "hook cs #11"), (c_x0, "cmp_x #62"), (c_x1, "cmp_x #62 nonzero")] {
            let pso = ID3D12PipelineState::from_raw(p);
            assert_cached_blob(&pso, label);
            assert_eq!(run_compute_u32(&dev, &rs_cs, &pso), 0x1234_5678, "{label} dispatch");
        }

        // CreatePipelineState streams (ID3D12Device2 #47 + console #86)
        let mut rtv = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: 1, ..Default::default() };
        rtv.RTFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
        let mut gs_body = GfxStream {
            rs: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, v: rs_gfx.as_raw() },
            vs: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, v: bc(&vs) },
            ps: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, v: bc(&ps) },
            blend: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, v: gfx.BlendState },
            mask: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, v: u32::MAX },
            rast: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, v: gfx.RasterizerState },
            depth: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, v: D3D12_DEPTH_STENCIL_DESC::default() },
            topo: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, v: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE },
            rtv: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, v: rtv },
            samples: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, v: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 } },
        };
        let gs_desc = D3D12_PIPELINE_STATE_STREAM_DESC { SizeInBytes: size_of::<GfxStream>(), pPipelineStateSubobjectStream: &mut gs_body as *mut _ as *mut c_void };
        let mut cs_body = ComputeStream {
            rs: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, v: rs_cs.as_raw() },
            cs: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS, v: bc(&cs) },
            node: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK, v: 0 },
        };
        let cs_desc = D3D12_PIPELINE_STATE_STREAM_DESC { SizeInBytes: size_of::<ComputeStream>(), pPipelineStateSubobjectStream: &mut cs_body as *mut _ as *mut c_void };
        let mut bad_node = Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK, v: 0u32 };
        let bad_stream = D3D12_PIPELINE_STATE_STREAM_DESC { SizeInBytes: size_of::<Sub<u32>>(), pPipelineStateSubobjectStream: &mut bad_node as *mut _ as *mut c_void };

        let (mut sg_orig, mut sg_pc, mut sg_x, mut sc_orig, mut sc_x) = (null_mut(), null_mut(), null_mut(), null_mut(), null_mut());
        assert_eq!(orig_stream(raw_dev, &gs_desc as *const _ as usize, &iid_pso as *const _ as usize, &mut sg_orig as *mut _ as usize), S_OK);
        assert_eq!(pc_call4(hook_stream, raw_dev, &gs_desc as *const _ as usize, &iid_pso as *const _ as usize, &mut sg_pc as *mut _ as usize), S_OK);
        assert_eq!(title_call4(stream_x, raw_dev, &gs_desc as *const _ as usize, &iid_pso as *const _ as usize, &mut sg_x as *mut _ as usize), S_OK);
        assert_eq!(orig_stream(raw_dev, &cs_desc as *const _ as usize, &iid_pso as *const _ as usize, &mut sc_orig as *mut _ as usize), S_OK);
        assert_eq!(title_call4(stream_x, raw_dev, &cs_desc as *const _ as usize, &iid_pso as *const _ as usize, &mut sc_x as *mut _ as usize), S_OK);
        for (p, label) in [(sg_orig, "orig stream gfx"), (sg_pc, "hook stream gfx"), (sg_x, "stream_x #86 gfx")] {
            assert_cached_blob(&ID3D12PipelineState::from_raw(p), label);
        }
        for (p, label) in [(sc_orig, "orig stream cs"), (sc_x, "stream_x #86 cs")] {
            let pso = ID3D12PipelineState::from_raw(p);
            assert_eq!(run_compute_u32(&dev, &rs_cs, &pso), 0x1234_5678, "{label} dispatch");
        }
        assert_eq!(pc_call4(hook_stream, raw_dev, &bad_stream as *const _ as usize, &iid_pso as *const _ as usize, &mut dummy_out as *mut _ as usize), orig_stream(raw_dev, &bad_stream as *const _ as usize, &iid_pso as *const _ as usize, &mut dummy_out as *mut _ as usize));

        // Unknown IID fallback (skipped when debug device is active because SDKLayers fast-fails)
        if dev.cast::<ID3D12DebugDevice>().is_err() {
            let mut fb = [null_mut(); 3];
            assert_eq!(orig_gfx(raw_dev, &gfx as *const _ as usize, &CONSOLE_UNKNOWN_IID as *const _ as usize, &mut fb[0] as *mut _ as usize), E_NOINTERFACE);
            assert_eq!(pc_call4(hook_gfx, raw_dev, &gfx as *const _ as usize, &CONSOLE_UNKNOWN_IID as *const _ as usize, &mut fb[0] as *mut _ as usize), S_OK);
            assert_eq!(orig_cmp(raw_dev, &cmp as *const _ as usize, &CONSOLE_UNKNOWN_IID as *const _ as usize, &mut fb[1] as *mut _ as usize), E_NOINTERFACE);
            assert_eq!(pc_call4(hook_cmp, raw_dev, &cmp as *const _ as usize, &CONSOLE_UNKNOWN_IID as *const _ as usize, &mut fb[1] as *mut _ as usize), S_OK);
            assert_eq!(orig_stream(raw_dev, &cs_desc as *const _ as usize, &CONSOLE_UNKNOWN_IID as *const _ as usize, &mut fb[2] as *mut _ as usize), E_NOINTERFACE);
            assert_eq!(title_call4(stream_x, raw_dev, &cs_desc as *const _ as usize, &CONSOLE_UNKNOWN_IID as *const _ as usize, &mut fb[2] as *mut _ as usize), S_OK);
            for p in fb {
                drop(ID3D12PipelineState::from_raw(p));
            }
        }
    }
}

#[test]
fn derived_pipeline_state_patches_render_state_on_gpu_and_validates_errors() {
    type G = D3D12_GRAPHICS_PIPELINE_STATE_DESC;
    assert_eq!((offset_of!(G, BlendState), size_of::<D3D12_BLEND_DESC>()), (0x78, 0x148));
    assert_eq!((offset_of!(G, SampleMask), size_of::<u32>()), (0x1C0, 4));
    assert_eq!((offset_of!(G, RasterizerState), size_of::<D3D12_RASTERIZER_DESC>()), (0x1C4, 0x2C));
    assert_eq!((offset_of!(G, DepthStencilState), size_of::<D3D12_DEPTH_STENCIL_DESC>()), (0x1F0, 0x34));

    let Some(dev) = init_device() else { return };
    unsafe {
        let raw_dev = dev.as_raw();
        let iid_pso = sys_guid(ID3D12PipelineState::IID);
        let rs = root_sig(&dev, false, D3D12_ROOT_SIGNATURE_FLAG_NONE);
        let rs_so = root_sig(&dev, false, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_STREAM_OUTPUT);
        let (vs, ps) = fullscreen_shaders();
        let gs = compile_dxbc("struct V { float4 p : SV_Position; float4 d : TEXCOORD0; }; [maxvertexcount(3)] void main(triangle float4 i[3] : SV_Position, inout TriangleStream<V> s) { for (int k = 0; k < 3; ++k) { V o; o.p = i[k]; o.d = i[k]; s.Append(o); } }", "gs_5_0");
        let base_desc = base_gfx_desc(&rs, &vs, &ps, DXGI_FORMAT_D32_FLOAT);

        let hook_gfx: Call4 = vtable_fn(raw_dev, slot_of("ID3D12Device", "CreateGraphicsPipelineState"));
        let derive_x: Call6 = vtable_fn(raw_dev, 55);
        let mut base_raw = null_mut();
        assert_eq!(title_call4(hook_gfx, raw_dev, &base_desc as *const _ as usize, &iid_pso as *const _ as usize, &mut base_raw as *mut _ as usize), S_OK);
        let base_pso = ID3D12PipelineState::from_raw(base_raw);

        // Error and boundary contracts
        let mut out = base_raw; // sentinel non-null to confirm zeroing
        assert_eq!(title_call6(derive_x, raw_dev, base_raw as usize, 0, 0, &iid_pso as *const _ as usize, 0), E_INVALIDARG);
        assert_eq!(title_call6(derive_x, raw_dev, 0, 0, 0, &iid_pso as *const _ as usize, &mut out as *mut _ as usize), E_FAIL);
        assert!(out.is_null());
        out = base_raw;
        assert_eq!(title_call6(derive_x, raw_dev, base_raw as usize, 65, 0, &iid_pso as *const _ as usize, &mut out as *mut _ as usize), E_INVALIDARG);
        assert!(out.is_null());
        assert_eq!(title_call6(derive_x, raw_dev, base_raw as usize, 1, 0, &iid_pso as *const _ as usize, &mut out as *mut _ as usize), E_INVALIDARG);
        let bad_part = patch_bytes(4, &0u32);
        assert_eq!(title_call6(derive_x, raw_dev, base_raw as usize, 1, bad_part.as_ptr() as usize, &iid_pso as *const _ as usize, &mut out as *mut _ as usize), E_FAIL);

        // Stream-output base is intentionally excluded from BASES -> E_FAIL on derive
        let so_entry = D3D12_SO_DECLARATION_ENTRY { Stream: 0, SemanticName: windows::core::s!("TEXCOORD"), SemanticIndex: 0, StartComponent: 0, ComponentCount: 4, OutputSlot: 0 };
        let stride = 16u32;
        let mut so_desc = base_gfx_desc(&rs_so, &vs, &ps, DXGI_FORMAT_UNKNOWN);
        so_desc.GS = bc(&gs);
        so_desc.PS = D3D12_SHADER_BYTECODE::default();
        so_desc.NumRenderTargets = 0;
        so_desc.RTVFormats[0] = DXGI_FORMAT_UNKNOWN;
        so_desc.StreamOutput = D3D12_STREAM_OUTPUT_DESC { pSODeclaration: &so_entry, NumEntries: 1, pBufferStrides: &stride, NumStrides: 1, RasterizedStream: D3D12_SO_NO_RASTERIZED_STREAM };
        let mut so_raw = null_mut();
        assert_eq!(title_call4(hook_gfx, raw_dev, &so_desc as *const _ as usize, &iid_pso as *const _ as usize, &mut so_raw as *mut _ as usize), S_OK);
        let _so_pso = ID3D12PipelineState::from_raw(so_raw);
        assert_eq!(title_call6(derive_x, raw_dev, so_raw as usize, 0, 0, &iid_pso as *const _ as usize, &mut out as *mut _ as usize), E_FAIL);

        // Live GPU verification of base + parts 2 (blend), 1 (rasterizer), 3 (sample mask), 0 (depth-stencil), 5 (ignored), and chaining
        let mut target = DrawTarget::new(&dev);
        assert_eq!(target.draw_pixel(&rs, &base_pso), [1.0, 0.5, 0.25, 1.0]);

        let mut blend = base_desc.BlendState;
        blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_GREEN.0 as u8;
        let p_blend_and_ign = [patch_bytes(2, &blend), patch_bytes(5, &[0u8; 0x30])];
        let mut d_blend_raw = null_mut();
        assert_eq!(title_call6(derive_x, raw_dev, base_raw as usize, 2, p_blend_and_ign.as_ptr() as usize, &iid_pso as *const _ as usize, &mut d_blend_raw as *mut _ as usize), S_OK);
        let d_blend = ID3D12PipelineState::from_raw(d_blend_raw);
        assert_eq!(target.draw_pixel(&rs, &d_blend), [0.0, 0.5, 0.0, 0.0]);

        // Chain from d_blend: front-face cull rejects our CW fullscreen triangle
        let mut rast = base_desc.RasterizerState;
        rast.CullMode = D3D12_CULL_MODE_FRONT;
        let p_rast = patch_bytes(1, &rast);
        let mut d_rast_raw = null_mut();
        assert_eq!(title_call6(derive_x, raw_dev, d_blend_raw as usize, 1, p_rast.as_ptr() as usize, &iid_pso as *const _ as usize, &mut d_rast_raw as *mut _ as usize), S_OK);
        let d_rast = ID3D12PipelineState::from_raw(d_rast_raw);
        assert_eq!(target.draw_pixel(&rs, &d_rast), [0.0, 0.0, 0.0, 0.0]);

        // SampleMask = 0 discards all samples
        let p_mask = patch_bytes(3, &0u32);
        let mut d_mask_raw = null_mut();
        assert_eq!(title_call6(derive_x, raw_dev, base_raw as usize, 1, p_mask.as_ptr() as usize, &iid_pso as *const _ as usize, &mut d_mask_raw as *mut _ as usize), S_OK);
        let d_mask = ID3D12PipelineState::from_raw(d_mask_raw);
        assert_eq!(target.draw_pixel(&rs, &d_mask), [0.0, 0.0, 0.0, 0.0]);

        // DepthEnable = true, DepthFunc = GREATER rejects z=0.5 against DSV cleared to 1.0
        let ds_state = D3D12_DEPTH_STENCIL_DESC { DepthEnable: true.into(), DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO, DepthFunc: D3D12_COMPARISON_FUNC_GREATER, ..Default::default() };
        let p_depth = patch_bytes(0, &ds_state);
        let mut d_depth_raw = null_mut();
        assert_eq!(title_call6(derive_x, raw_dev, base_raw as usize, 1, p_depth.as_ptr() as usize, &iid_pso as *const _ as usize, &mut d_depth_raw as *mut _ as usize), S_OK);
        let d_depth = ID3D12PipelineState::from_raw(d_depth_raw);
        assert_eq!(target.draw_pixel(&rs, &d_depth), [0.0, 0.0, 0.0, 0.0]);

        // Releasing a base PSO forgets it via pso_release -> forget_base
        drop(d_blend);
        assert_eq!(title_call6(derive_x, raw_dev, d_blend_raw as usize, 0, 0, &iid_pso as *const _ as usize, &mut out as *mut _ as usize), E_FAIL);
    }
}

#[test]
fn console_pipeline_serialization_and_device_extensions_follow_contract() {
    let Some(dev) = init_device() else { return };
    unsafe {
        let raw_dev = dev.as_raw();
        let iid_pso = sys_guid(ID3D12PipelineState::IID);

        let set_debug_flags: Call2 = vtable_fn(raw_dev, 52);
        let ser_gfx: Call2 = vtable_fn(raw_dev, 56);
        let deser_gfx: Call4 = vtable_fn(raw_dev, 57);
        let ser_cmp: Call2 = vtable_fn(raw_dev, 66);
        let deser_cmp: Call4 = vtable_fn(raw_dev, 67);
        let reg_pool: Call4 = vtable_fn(raw_dev, 74);
        let unreg_pool: Call2 = vtable_fn(raw_dev, 75);

        assert_eq!(title_call2(set_debug_flags, raw_dev, 0), S_OK);
        assert_eq!(title_call2(set_debug_flags, raw_dev, 0xFFFF_FFFF), S_OK);

        assert_eq!(title_call2(ser_gfx, raw_dev, 0), E_ACCESSDENIED);
        assert_eq!(title_call2(ser_cmp, raw_dev, 0), E_ACCESSDENIED);

        for deser in [deser_gfx, deser_cmp] {
            let mut out = 0xDEAD_BEEFusize as *mut c_void;
            assert_eq!(title_call4(deser, raw_dev, 0, &iid_pso as *const _ as usize, &mut out as *mut _ as usize), E_NOTIMPL);
            assert!(out.is_null(), "Deserialize*PipelineStateX must clear *ppv");
            assert_eq!(title_call4(deser, raw_dev, 0, &iid_pso as *const _ as usize, 0), E_NOTIMPL);
        }

        let (mut idx0, mut idx1, mut idx2) = (u32::MAX, u32::MAX, u32::MAX);
        assert_eq!(title_call4(reg_pool, raw_dev, 0, 4, &mut idx0 as *mut _ as usize), E_INVALIDARG);
        assert_eq!(title_call4(reg_pool, raw_dev, 0x2_0000_0000, 0, &mut idx0 as *mut _ as usize), E_INVALIDARG);
        assert_eq!(title_call4(reg_pool, raw_dev, 0x2_0000_0000, 16, &mut idx0 as *mut _ as usize), S_OK);
        assert_eq!(title_call4(reg_pool, raw_dev, 0x2_0000_0000, 16, &mut idx1 as *mut _ as usize), S_OK);
        assert_eq!(idx0, idx1, "identical (va, pages) registration must return same index");
        assert_eq!(title_call4(reg_pool, raw_dev, 0x2_0000_0000, 16, 0), S_OK);
        assert_eq!(title_call4(reg_pool, raw_dev, 0x2_0010_0000, 32, &mut idx2 as *mut _ as usize), S_OK);
        assert_ne!(idx2, idx0);
        assert_eq!(title_call2(unreg_pool, raw_dev, idx0 as usize), S_OK);
        assert_eq!(title_call2(unreg_pool, raw_dev, idx0 as usize), E_FAIL);
        assert_eq!(title_call2(unreg_pool, raw_dev, idx2 as usize), S_OK);
        assert_eq!(title_call2(unreg_pool, raw_dev, usize::MAX), E_FAIL);
    }
}
