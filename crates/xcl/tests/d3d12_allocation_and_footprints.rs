//! Host-driver parity and contract tests for XCL's D3D12.X memory-layout and
//! copy-footprint query hooks without console game files (Part of #480):
//! `GetResourceAllocationInfo` (#25), `GetResourceAllocationInfo2` (console #108 vs
//! PC `ID3D12Device8` #68), `GetCustomHeapProperties` (#26), and `GetCopyableFootprints` (#38).

use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::null_mut;
use std::sync::{Arc, Once};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows_sys::Win32::System::Diagnostics::Debug::RtlLookupFunctionEntry;

use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::{self, device_x, layout, real};
use xcl::hle::xg;
use xcl::modules::{self, MappedModule};

const SLOT_ALLOC_INFO: usize = 25;
const SLOT_HEAP_PROPS: usize = 26;
const SLOT_COPY_FOOTPRINTS: usize = 38;
const SLOT_ALLOC_INFO2_CONSOLE: usize = 108;

/// Console `D3D12_RESOURCE_DESC1` stride expected by device slot 108 (`device_x`).
const CONSOLE_DESC1_BYTES: usize = 0x48;
/// Console `D3D12_TEXTURE_LAYOUT` value rejected by PC drivers (triggers console fallback).
const CONSOLE_LAYOUT_SWIZZLE: i32 = 0x101;

type AllocInfoCppFn = unsafe extern "system" fn(*mut c_void, *mut [u64; 2], u32, u32, *const [u8; 0x38]) -> *mut [u64; 2];
type AllocInfoCFn = unsafe extern "system" fn(*mut [u64; 2], *mut c_void, u32, u32, *const [u8; 0x38]) -> *mut [u64; 2];

type AllocInfo2PcFn = unsafe extern "system" fn(*mut c_void, *mut [u64; 2], u32, u32, *const D3D12_RESOURCE_DESC1, *mut [u64; 3]) -> *mut [u64; 2];
type AllocInfo2ConsoleCppFn = unsafe extern "system" fn(*mut c_void, *mut [u64; 2], u32, u32, *const [u8; CONSOLE_DESC1_BYTES], *mut [u64; 3]) -> *mut [u64; 2];
type AllocInfo2ConsoleCFn = unsafe extern "system" fn(*mut [u64; 2], *mut c_void, u32, u32, *const [u8; CONSOLE_DESC1_BYTES], *mut [u64; 3]) -> *mut [u64; 2];

type HeapPropsCppFn = unsafe extern "system" fn(*mut c_void, *mut [u32; 5], u32, i32) -> *mut [u32; 5];
type HeapPropsCFn = unsafe extern "system" fn(*mut [u32; 5], *mut c_void, u32, i32) -> *mut [u32; 5];

type CopyFootprintsFn = unsafe extern "system" fn(
    *mut c_void,
    *const [u8; 0x38],
    u32,
    u32,
    u64,
    *mut D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    *mut u32,
    *mut u64,
    *mut u64,
);

#[inline(never)]
unsafe fn title_alloc_info_cpp(f: AllocInfoCppFn, dev: *mut c_void, out: *mut [u64; 2], mask: u32, n: u32, descs: *const [u8; 0x38]) -> *mut [u64; 2] {
    let r = f(dev, out, mask, n, descs);
    std::hint::black_box(r)
}

#[inline(never)]
unsafe fn title_alloc_info_c(f: AllocInfoCFn, out: *mut [u64; 2], dev: *mut c_void, mask: u32, n: u32, descs: *const [u8; 0x38]) -> *mut [u64; 2] {
    let r = f(out, dev, mask, n, descs);
    std::hint::black_box(r)
}

#[inline(never)]
unsafe fn title_alloc_info2_cpp(
    f: AllocInfo2ConsoleCppFn,
    dev: *mut c_void,
    out: *mut [u64; 2],
    mask: u32,
    n: u32,
    descs: *const [u8; CONSOLE_DESC1_BYTES],
    infos: *mut [u64; 3],
) -> *mut [u64; 2] {
    let r = f(dev, out, mask, n, descs, infos);
    std::hint::black_box(r)
}

#[inline(never)]
unsafe fn title_alloc_info2_c(
    f: AllocInfo2ConsoleCFn,
    out: *mut [u64; 2],
    dev: *mut c_void,
    mask: u32,
    n: u32,
    descs: *const [u8; CONSOLE_DESC1_BYTES],
    infos: *mut [u64; 3],
) -> *mut [u64; 2] {
    let r = f(out, dev, mask, n, descs, infos);
    std::hint::black_box(r)
}

#[inline(never)]
unsafe fn title_heap_props_cpp(f: HeapPropsCppFn, dev: *mut c_void, out: *mut [u32; 5], mask: u32, ty: i32) -> *mut [u32; 5] {
    let r = f(dev, out, mask, ty);
    std::hint::black_box(r)
}

#[inline(never)]
unsafe fn title_heap_props_c(f: HeapPropsCFn, out: *mut [u32; 5], dev: *mut c_void, mask: u32, ty: i32) -> *mut [u32; 5] {
    let r = f(out, dev, mask, ty);
    std::hint::black_box(r)
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn title_copy_footprints(
    f: CopyFootprintsFn,
    dev: *mut c_void,
    desc: *const [u8; 0x38],
    first: u32,
    num: u32,
    base: u64,
    layouts: *mut D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    rows: *mut u32,
    row_sizes: *mut u64,
    total: *mut u64,
) {
    f(dev, desc, first, num, base, layouts, rows, row_sizes, total);
    std::hint::black_box(0u8);
}

#[inline(never)]
unsafe fn pc_alloc_info(f: AllocInfoCppFn, dev: *mut c_void, out: *mut [u64; 2], mask: u32, n: u32, descs: *const [u8; 0x38]) -> *mut [u64; 2] {
    let r = f(dev, out, mask, n, descs);
    std::hint::black_box(r)
}

#[inline(never)]
unsafe fn pc_heap_props(f: HeapPropsCppFn, dev: *mut c_void, out: *mut [u32; 5], mask: u32, ty: i32) -> *mut [u32; 5] {
    let r = f(dev, out, mask, ty);
    std::hint::black_box(r)
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn pc_copy_footprints(
    f: CopyFootprintsFn,
    dev: *mut c_void,
    desc: *const [u8; 0x38],
    first: u32,
    num: u32,
    base: u64,
    layouts: *mut D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    rows: *mut u32,
    row_sizes: *mut u64,
    total: *mut u64,
) {
    f(dev, desc, first, num, base, layouts, rows, row_sizes, total);
    std::hint::black_box(0u8);
}

/// Register the `title_*` wrapper functions as a synthetic mapped module so `guard::make_split`
/// routes calls originating from them to XCL's console device slots while `pc_*` wrappers stay PC.
fn ensure_title_callers() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let callers = [
            title_alloc_info_cpp as *const () as u64,
            title_alloc_info_c as *const () as u64,
            title_alloc_info2_cpp as *const () as u64,
            title_alloc_info2_c as *const () as u64,
            title_heap_props_cpp as *const () as u64,
            title_heap_props_c as *const () as u64,
            title_copy_footprints as *const () as u64,
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
                name: format!("d3d12-alloc-fixture-{i}"),
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
    let iface = layout::find(iface_name).unwrap_or_else(|| panic!("missing {iface_name} in layout"));
    layout::slot(iface, method).unwrap_or_else(|| panic!("missing {iface_name}::{method} in layout"))
}

unsafe fn vtable_fn<F: Copy>(dev: *mut c_void, slot: usize) -> F {
    assert_eq!(size_of::<F>(), size_of::<usize>());
    let vt = *(dev as *const *const usize);
    let addr = *vt.add(slot);
    assert_ne!(addr, 0, "null vtable entry at slot {slot}");
    std::mem::transmute_copy(&addr)
}

unsafe fn orig_slot_fn<F: Copy>(dev: *mut c_void, slot: usize) -> F {
    assert_eq!(size_of::<F>(), size_of::<usize>());
    let addr = real::orig_fn(dev, slot);
    assert_ne!(addr, 0, "null original vtable entry at slot {slot}");
    std::mem::transmute_copy(&addr)
}

/// 8-byte-aligned `[u32; 5]` buffer for `GetCustomHeapProperties` (`D3D12_HEAP_PROPERTIES` is 4-byte
/// aligned, which hits `real::is_adopted`'s unaligned `*const usize` dereference in #1130).
#[repr(C, align(8))]
#[derive(Clone, Copy, Default)]
struct AlignedHeapProps([u32; 5]);

fn desc_bytes(desc: &D3D12_RESOURCE_DESC) -> [u8; 0x38] {
    assert_eq!(size_of::<D3D12_RESOURCE_DESC>(), 0x38);
    let mut out = [0u8; 0x38];
    unsafe {
        std::ptr::copy_nonoverlapping(desc as *const _ as *const u8, out.as_mut_ptr(), 0x38);
    }
    out
}

fn desc1_from_desc(desc: &D3D12_RESOURCE_DESC) -> D3D12_RESOURCE_DESC1 {
    assert_eq!(size_of::<D3D12_RESOURCE_DESC1>(), 0x40);
    D3D12_RESOURCE_DESC1 {
        Dimension: desc.Dimension,
        Alignment: desc.Alignment,
        Width: desc.Width,
        Height: desc.Height,
        DepthOrArraySize: desc.DepthOrArraySize,
        MipLevels: desc.MipLevels,
        Format: desc.Format,
        SampleDesc: desc.SampleDesc,
        Layout: desc.Layout,
        Flags: desc.Flags,
        SamplerFeedbackMipRegion: D3D12_MIP_REGION::default(),
    }
}

fn console_desc1_bytes(desc: &D3D12_RESOURCE_DESC) -> [u8; CONSOLE_DESC1_BYTES] {
    let mut out = [0u8; CONSOLE_DESC1_BYTES];
    out[..0x38].copy_from_slice(&desc_bytes(desc));
    out
}

fn buf_desc(bytes: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    }
}

fn tex2d_desc(
    format: DXGI_FORMAT,
    width: u64,
    height: u32,
    array: u16,
    mips: u16,
    samples: u32,
    layout: D3D12_TEXTURE_LAYOUT,
    flags: D3D12_RESOURCE_FLAGS,
) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: width,
        Height: height,
        DepthOrArraySize: array,
        MipLevels: mips,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: samples, Quality: 0 },
        Layout: layout,
        Flags: flags,
    }
}

fn tex3d_desc(format: DXGI_FORMAT, width: u64, height: u32, depth: u16, mips: u16, layout: D3D12_TEXTURE_LAYOUT) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
        Alignment: 0,
        Width: width,
        Height: height,
        DepthOrArraySize: depth,
        MipLevels: mips,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: layout,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    }
}

/// Pack `(size, align)` pairs identically to D3D12's multi-resource `GetResourceAllocationInfo` rule.
fn pack_allocations(parts: &[(u64, u64)]) -> ([u64; 2], Vec<[u64; 3]>) {
    let mut size = 0u64;
    let mut align = 0u64;
    let mut per = Vec::with_capacity(parts.len());
    for &(s, a) in parts {
        if s == u64::MAX || a == 0 {
            return ([u64::MAX, align], per);
        }
        let offset = size.next_multiple_of(a);
        per.push([offset, a, s]);
        size = offset + s;
        align = align.max(a);
    }
    if parts.len() > 1 {
        size = size.next_multiple_of(align);
    }
    ([size, align], per)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FootprintSnapshot {
    layouts: Vec<(u64, i32, u32, u32, u32, u32)>,
    rows: Vec<u32>,
    row_sizes: Vec<u64>,
    total: u64,
}

fn capture_footprints(
    call: impl FnOnce(*mut D3D12_PLACED_SUBRESOURCE_FOOTPRINT, *mut u32, *mut u64, *mut u64),
    n: usize,
) -> FootprintSnapshot {
    let mut layouts = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n.max(1)];
    let mut rows = vec![0xDEAD_BEEFu32; n.max(1)];
    let mut row_sizes = vec![0xDEAD_BEEF_DEAD_BEEFu64; n.max(1)];
    let mut total = 0xDEAD_BEEF_DEAD_BEEFu64;
    call(
        if n == 0 { null_mut() } else { layouts.as_mut_ptr() },
        if n == 0 { null_mut() } else { rows.as_mut_ptr() },
        if n == 0 { null_mut() } else { row_sizes.as_mut_ptr() },
        &mut total,
    );
    layouts.truncate(n);
    rows.truncate(n);
    row_sizes.truncate(n);
    FootprintSnapshot {
        layouts: layouts
            .into_iter()
            .map(|l| (l.Offset, l.Footprint.Format.0, l.Footprint.Width, l.Footprint.Height, l.Footprint.Depth, l.Footprint.RowPitch))
            .collect(),
        rows,
        row_sizes,
        total,
    }
}

#[test]
fn get_resource_allocation_info_matches_host_driver_and_console_tiling() {
    let Some(dev) = init_device() else { return };
    let raw = dev.as_raw();
    assert_eq!(slot_of("ID3D12Device", "GetResourceAllocationInfo"), SLOT_ALLOC_INFO);

    let hooked_cpp: AllocInfoCppFn = unsafe { vtable_fn(raw, SLOT_ALLOC_INFO) };
    let hooked_c: AllocInfoCFn = unsafe { vtable_fn(raw, SLOT_ALLOC_INFO) };
    let orig_cpp: AllocInfoCppFn = unsafe { orig_slot_fn(raw, SLOT_ALLOC_INFO) };
    assert_ne!(hooked_cpp as usize, orig_cpp as usize, "slot 25 must be split-hooked");

    // Non-title PC callers go straight to the host driver, even on a console-layout texture.
    let console_tex = desc_bytes(&tex2d_desc(
        DXGI_FORMAT_R8G8B8A8_UNORM,
        256,
        256,
        1,
        1,
        1,
        D3D12_TEXTURE_LAYOUT(CONSOLE_LAYOUT_SWIZZLE),
        D3D12_RESOURCE_FLAG_NONE,
    ));
    let mut pc_out = [0u64; 2];
    let mut host_out = [0u64; 2];
    unsafe {
        pc_alloc_info(hooked_cpp, raw, &mut pc_out, 0, 1, &console_tex);
        orig_cpp(raw, &mut host_out, 0, 1, &console_tex);
    }
    assert_eq!(pc_out, host_out, "PC caller on slot 25 must match host driver directly");

    // Console callers on buffers and MSAA textures normalize via pc_resource_desc and match host.
    let mut console_buf = desc_bytes(&buf_desc(1234));
    console_buf[8..16].copy_from_slice(&0x40u64.to_le_bytes());
    console_buf[0x30..0x34].copy_from_slice(&0x8000_0004u32.to_le_bytes());
    let norm_buf = device_x::pc_resource_desc(&console_buf);

    let mut invalid_flags_buf = desc_bytes(&buf_desc(0x1_0000));
    invalid_flags_buf[0x30..0x34].copy_from_slice(&(D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS.0 as u32).to_le_bytes());

    let msaa_rt = desc_bytes(&tex2d_desc(
        DXGI_FORMAT_R8G8B8A8_UNORM,
        256,
        256,
        1,
        1,
        4,
        D3D12_TEXTURE_LAYOUT_UNKNOWN,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
    ));
    let std_buf = desc_bytes(&buf_desc(0x1_0000));
    let zero_buf = desc_bytes(&buf_desc(0));

    let single_cases: [(&str, [u8; 0x38], [u8; 0x38], bool); 6] = [
        ("buf 1 B", desc_bytes(&buf_desc(1)), desc_bytes(&buf_desc(1)), false),
        ("buf 64 KiB", std_buf, std_buf, false),
        ("console buf alignment+flags", console_buf, norm_buf, false),
        ("msaa 4x render target", msaa_rt, msaa_rt, false),
        ("zero-width buf", zero_buf, zero_buf, true),
        ("invalid simultaneous-access buf", invalid_flags_buf, invalid_flags_buf, true),
    ];

    for (label, xcl_d, host_d, expect_err) in &single_cases {
        let mut host = [0u64; 2];
        let mut xcl_cpp = [0u64; 2];
        let mut xcl_c = [0u64; 2];
        // Seed the first qword with the device's extended vtable pointer to verify `this_first`.
        let mut xcl_stale = [unsafe { *(raw as *const u64) }, 0u64];
        unsafe {
            orig_cpp(raw, &mut host, 0, 1, host_d);
            let r_cpp = title_alloc_info_cpp(hooked_cpp, raw, &mut xcl_cpp, 0, 1, xcl_d);
            assert_eq!(r_cpp, &mut xcl_cpp as *mut _);
            let r_c = title_alloc_info_c(hooked_c, &mut xcl_c, raw, 0, 1, xcl_d);
            assert_eq!(r_c, &mut xcl_c as *mut _);
            let r_stale = title_alloc_info_c(hooked_c, &mut xcl_stale, raw, 0, 1, xcl_d);
            assert_eq!(r_stale, &mut xcl_stale as *mut _);
        }
        if *expect_err {
            assert_eq!(host[0], u64::MAX, "{label}: expected host SizeInBytes == u64::MAX");
            assert_eq!(xcl_cpp[0], u64::MAX, "{label}: C++ ABI expected SizeInBytes == u64::MAX");
            assert_eq!(xcl_c[0], u64::MAX, "{label}: C ABI expected SizeInBytes == u64::MAX");
            assert_eq!(xcl_stale[0], u64::MAX, "{label}: stale-vtable C ABI expected SizeInBytes == u64::MAX");
        } else {
            assert_eq!(xcl_cpp, host, "{label}: C++ ABI mismatch vs host");
            assert_eq!(xcl_c, host, "{label}: C ABI mismatch vs host");
            assert_eq!(xcl_stale, host, "{label}: C ABI with stale vtable copy mismatch vs host");
        }
    }

    // Multi-resource alignment packing: 64 KiB buffer + MSAA 4x RT + 64 KiB buffer.
    let multi_pc = [std_buf, msaa_rt, std_buf];
    let mut host_multi = [0u64; 2];
    let mut xcl_multi = [0u64; 2];
    unsafe {
        orig_cpp(raw, &mut host_multi, 0, multi_pc.len() as u32, multi_pc.as_ptr());
        title_alloc_info_cpp(hooked_cpp, raw, &mut xcl_multi, 0, multi_pc.len() as u32, multi_pc.as_ptr());
    }
    assert_eq!(xcl_multi, host_multi, "multi-resource buffer+MSAA packing must match host");
    assert_ne!(host_multi[0], u64::MAX);

    // Invalid entry anywhere in the array stops packing with SizeInBytes == u64::MAX.
    let multi_err = [std_buf, zero_buf];
    let mut host_err = [0u64; 2];
    let mut xcl_err = [0u64; 2];
    unsafe {
        orig_cpp(raw, &mut host_err, 0, multi_err.len() as u32, multi_err.as_ptr());
        title_alloc_info_cpp(hooked_cpp, raw, &mut xcl_err, 0, multi_err.len() as u32, multi_err.as_ptr());
    }
    assert_eq!(host_err[0], u64::MAX);
    assert_eq!(xcl_err[0], u64::MAX);

    // Zero count and NULL descriptor array handling.
    let mut zero_count = [123u64; 2];
    unsafe {
        title_alloc_info_cpp(hooked_cpp, raw, &mut zero_count, 0, 0, std::ptr::null());
    }
    assert_eq!(zero_count, [0, 0]);
    let mut null_descs = [0u64; 2];
    unsafe {
        title_alloc_info_cpp(hooked_cpp, raw, &mut null_descs, 0, 1, std::ptr::null());
    }
    assert_eq!(null_descs[0], u64::MAX);

    // Non-MSAA textures use XG's console layout size/alignment and pack via allocation_of.
    let tex_cases = [
        desc_bytes(&tex2d_desc(
            DXGI_FORMAT_R8G8B8A8_UNORM,
            1937,
            333,
            1,
            1,
            1,
            D3D12_TEXTURE_LAYOUT_UNKNOWN,
            D3D12_RESOURCE_FLAG_NONE,
        )),
        desc_bytes(&tex2d_desc(
            DXGI_FORMAT_BC1_UNORM,
            256,
            256,
            6,
            3,
            1,
            D3D12_TEXTURE_LAYOUT(CONSOLE_LAYOUT_SWIZZLE),
            D3D12_RESOURCE_FLAG_NONE,
        )),
        desc_bytes(&tex2d_desc(
            DXGI_FORMAT(118),
            256,
            256,
            1,
            1,
            1,
            D3D12_TEXTURE_LAYOUT_UNKNOWN,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        )),
        desc_bytes(&tex3d_desc(
            DXGI_FORMAT_R8G8B8A8_UNORM,
            32,
            24,
            16,
            3,
            D3D12_TEXTURE_LAYOUT_UNKNOWN,
        )),
    ];
    let mut expected_parts = Vec::with_capacity(tex_cases.len() + 1);
    for d in &tex_cases {
        let xg_tex = xg::Texture::new(d).expect("xg::Texture must model console texture");
        let want = [xg_tex.size(), xg_tex.alignment()];
        let mut got = [0u64; 2];
        unsafe {
            title_alloc_info_cpp(hooked_cpp, raw, &mut got, 0, 1, d);
        }
        assert_eq!(got, want, "console texture GetResourceAllocationInfo must match xg::Texture");
        expected_parts.push((want[0], want[1]));
    }
    let mut host_std_buf = [0u64; 2];
    unsafe { orig_cpp(raw, &mut host_std_buf, 0, 1, &std_buf) };
    expected_parts.push((host_std_buf[0], host_std_buf[1]));

    let mut mixed = tex_cases.to_vec();
    mixed.push(std_buf);
    let mut got_mixed = [0u64; 2];
    unsafe {
        title_alloc_info_cpp(hooked_cpp, raw, &mut got_mixed, 0, mixed.len() as u32, mixed.as_ptr());
    }
    let (want_mixed, _) = pack_allocations(&expected_parts);
    assert_eq!(got_mixed, want_mixed, "multi-resource console texture + buffer packing mismatch");
}

#[test]
fn get_resource_allocation_info2_matches_host_driver_and_console_offsets() {
    let Some(dev) = init_device() else { return };
    let raw = dev.as_raw();
    let Ok(dev8) = dev.cast::<ID3D12Device8>() else {
        runner::skip_test("no ID3D12Device8");
        return;
    };

    let pc_slot = slot_of("ID3D12Device8", "GetResourceAllocationInfo2");
    let orig_pc: AllocInfo2PcFn = unsafe { orig_slot_fn(dev8.as_raw(), pc_slot) };
    let hooked_cpp: AllocInfo2ConsoleCppFn = unsafe { vtable_fn(raw, SLOT_ALLOC_INFO2_CONSOLE) };
    let hooked_c: AllocInfo2ConsoleCFn = unsafe { vtable_fn(raw, SLOT_ALLOC_INFO2_CONSOLE) };

    let buf_64k = buf_desc(0x1_0000);
    let buf_256k = buf_desc(0x4_0000);
    let msaa_rt = tex2d_desc(
        DXGI_FORMAT_R8G8B8A8_UNORM,
        256,
        256,
        1,
        1,
        4,
        D3D12_TEXTURE_LAYOUT_UNKNOWN,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
    );
    let zero_buf = buf_desc(0);

    // PC GetResourceAllocationInfo2 takes 0x40-byte D3D12_RESOURCE_DESC1; console slot 108 takes 0x48-byte entries.
    let pc_descs = [desc1_from_desc(&buf_64k), desc1_from_desc(&msaa_rt), desc1_from_desc(&buf_256k)];
    let xcl_descs = [console_desc1_bytes(&buf_64k), console_desc1_bytes(&msaa_rt), console_desc1_bytes(&buf_256k)];

    let mut host_total = [0u64; 2];
    let mut host_infos = [[0u64; 3]; 3];
    let mut xcl_cpp_total = [0u64; 2];
    let mut xcl_cpp_infos = [[0u64; 3]; 3];
    let mut xcl_c_total = [0u64; 2];
    let mut xcl_c_infos = [[0u64; 3]; 3];
    let mut xcl_stale_total = [unsafe { *(raw as *const u64) }, 0];
    let mut xcl_null_infos_total = [0u64; 2];

    unsafe {
        orig_pc(dev8.as_raw(), &mut host_total, 0, pc_descs.len() as u32, pc_descs.as_ptr(), host_infos.as_mut_ptr());
        let r_cpp = title_alloc_info2_cpp(
            hooked_cpp,
            raw,
            &mut xcl_cpp_total,
            0,
            xcl_descs.len() as u32,
            xcl_descs.as_ptr(),
            xcl_cpp_infos.as_mut_ptr(),
        );
        assert_eq!(r_cpp, &mut xcl_cpp_total as *mut _);
        let r_c = title_alloc_info2_c(
            hooked_c,
            &mut xcl_c_total,
            raw,
            0,
            xcl_descs.len() as u32,
            xcl_descs.as_ptr(),
            xcl_c_infos.as_mut_ptr(),
        );
        assert_eq!(r_c, &mut xcl_c_total as *mut _);
        title_alloc_info2_c(
            hooked_c,
            &mut xcl_stale_total,
            raw,
            0,
            xcl_descs.len() as u32,
            xcl_descs.as_ptr(),
            null_mut(),
        );
        title_alloc_info2_cpp(
            hooked_cpp,
            raw,
            &mut xcl_null_infos_total,
            0,
            xcl_descs.len() as u32,
            xcl_descs.as_ptr(),
            null_mut(),
        );
    }

    assert_eq!(xcl_cpp_total, host_total, "GetResourceAllocationInfo2 total mismatch vs host");
    assert_eq!(xcl_cpp_infos, host_infos, "GetResourceAllocationInfo2 per-resource infos mismatch vs host");
    assert_eq!(xcl_c_total, host_total);
    assert_eq!(xcl_c_infos, host_infos);
    assert_eq!(xcl_stale_total, host_total);
    assert_eq!(xcl_null_infos_total, host_total);

    // Console textures in D3D12_RESOURCE_DESC1 compute XG sizes/alignments and pack offsets cleanly.
    let c_tex_a = tex2d_desc(
        DXGI_FORMAT_R8G8B8A8_UNORM,
        256,
        256,
        1,
        4,
        1,
        D3D12_TEXTURE_LAYOUT(CONSOLE_LAYOUT_SWIZZLE),
        D3D12_RESOURCE_FLAG_NONE,
    );
    let c_tex_b = tex3d_desc(
        DXGI_FORMAT_BC1_UNORM,
        32,
        32,
        16,
        2,
        D3D12_TEXTURE_LAYOUT_UNKNOWN,
    );
    let xg_a = xg::Texture::new(&desc_bytes(&c_tex_a)).unwrap();
    let xg_b = xg::Texture::new(&desc_bytes(&c_tex_b)).unwrap();
    let (want_tex_total, want_tex_infos) = pack_allocations(&[(xg_a.size(), xg_a.alignment()), (xg_b.size(), xg_b.alignment())]);

    let xcl_tex_descs = [console_desc1_bytes(&c_tex_a), console_desc1_bytes(&c_tex_b)];
    let mut got_tex_total = [0u64; 2];
    let mut got_tex_infos = [[0u64; 3]; 2];
    unsafe {
        title_alloc_info2_cpp(
            hooked_cpp,
            raw,
            &mut got_tex_total,
            0,
            xcl_tex_descs.len() as u32,
            xcl_tex_descs.as_ptr(),
            got_tex_infos.as_mut_ptr(),
        );
    }
    assert_eq!(got_tex_total, want_tex_total);
    assert_eq!(got_tex_infos, want_tex_infos.as_slice());

    // Failure on any entry propagates u64::MAX identically to the host driver.
    let pc_err = [desc1_from_desc(&buf_64k), desc1_from_desc(&zero_buf)];
    let xcl_err = [console_desc1_bytes(&buf_64k), console_desc1_bytes(&zero_buf)];
    let mut host_err_total = [0u64; 2];
    let mut host_err_infos = [[0u64; 3]; 2];
    let mut xcl_err_total = [0u64; 2];
    let mut xcl_err_infos = [[0u64; 3]; 2];
    unsafe {
        orig_pc(dev8.as_raw(), &mut host_err_total, 0, pc_err.len() as u32, pc_err.as_ptr(), host_err_infos.as_mut_ptr());
        title_alloc_info2_cpp(
            hooked_cpp,
            raw,
            &mut xcl_err_total,
            0,
            xcl_err.len() as u32,
            xcl_err.as_ptr(),
            xcl_err_infos.as_mut_ptr(),
        );
    }
    assert_eq!(host_err_total[0], u64::MAX);
    assert_eq!(xcl_err_total[0], u64::MAX);
}

#[test]
fn get_custom_heap_properties_matches_host_driver_across_types_and_abis() {
    let Some(dev) = init_device() else { return };
    let raw = dev.as_raw();
    assert_eq!(slot_of("ID3D12Device", "GetCustomHeapProperties"), SLOT_HEAP_PROPS);

    let hooked_cpp: HeapPropsCppFn = unsafe { vtable_fn(raw, SLOT_HEAP_PROPS) };
    let hooked_c: HeapPropsCFn = unsafe { vtable_fn(raw, SLOT_HEAP_PROPS) };
    let orig_cpp: HeapPropsCppFn = unsafe { orig_slot_fn(raw, SLOT_HEAP_PROPS) };
    assert_ne!(hooked_cpp as usize, orig_cpp as usize, "slot 26 must be split-hooked");

    // PC caller goes straight to the host driver.
    let mut pc_props = AlignedHeapProps::default();
    let mut host_props = AlignedHeapProps::default();
    unsafe {
        pc_heap_props(hooked_cpp, raw, pc_props.0.as_mut_ptr() as *mut [u32; 5], 0, D3D12_HEAP_TYPE_UPLOAD.0);
        orig_cpp(raw, host_props.0.as_mut_ptr() as *mut [u32; 5], 0, D3D12_HEAP_TYPE_UPLOAD.0);
    }
    assert_eq!(pc_props.0, host_props.0);

    let heap_types = [
        D3D12_HEAP_TYPE_DEFAULT.0,
        D3D12_HEAP_TYPE_UPLOAD.0,
        D3D12_HEAP_TYPE_READBACK.0,
        D3D12_HEAP_TYPE_CUSTOM.0,
        99, // invalid heap type
    ];
    for mask in [0u32, 1u32] {
        for ty in heap_types {
            let mut host = AlignedHeapProps::default();
            let mut xcl_cpp = AlignedHeapProps::default();
            let mut xcl_c = AlignedHeapProps::default();
            let stale_lo = unsafe { *(raw as *const u64) };
            let mut xcl_stale = AlignedHeapProps([stale_lo as u32, (stale_lo >> 32) as u32, 0, 0, 0]);
            unsafe {
                orig_cpp(raw, host.0.as_mut_ptr() as *mut [u32; 5], mask, ty);
                let r_cpp = title_heap_props_cpp(hooked_cpp, raw, xcl_cpp.0.as_mut_ptr() as *mut [u32; 5], mask, ty);
                assert_eq!(r_cpp, xcl_cpp.0.as_mut_ptr() as *mut [u32; 5]);
                let r_c = title_heap_props_c(hooked_c, xcl_c.0.as_mut_ptr() as *mut [u32; 5], raw, mask, ty);
                assert_eq!(r_c, xcl_c.0.as_mut_ptr() as *mut [u32; 5]);
                let r_stale = title_heap_props_c(hooked_c, xcl_stale.0.as_mut_ptr() as *mut [u32; 5], raw, mask, ty);
                assert_eq!(r_stale, xcl_stale.0.as_mut_ptr() as *mut [u32; 5]);
            }
            assert_eq!(xcl_cpp.0, host.0, "heap type {ty} mask {mask}: C++ ABI mismatch vs host");
            assert_eq!(xcl_c.0, host.0, "heap type {ty} mask {mask}: C ABI mismatch vs host");
            assert_eq!(xcl_stale.0, host.0, "heap type {ty} mask {mask}: C ABI with stale vtable copy mismatch vs host");

            if ty == D3D12_HEAP_TYPE_UPLOAD.0 {
                assert_eq!(host.0[0], D3D12_HEAP_TYPE_CUSTOM.0 as u32);
                assert!(
                    host.0[1] == D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE.0 as u32
                        || host.0[1] == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK.0 as u32,
                    "unexpected UPLOAD CPU page property: {}",
                    host.0[1]
                );
            } else if ty == D3D12_HEAP_TYPE_READBACK.0 {
                assert_eq!(host.0[0], D3D12_HEAP_TYPE_CUSTOM.0 as u32);
                assert_eq!(host.0[1], D3D12_CPU_PAGE_PROPERTY_WRITE_BACK.0 as u32);
            }
        }
    }
}

#[test]
fn get_copyable_footprints_matches_host_driver_and_console_fallbacks() {
    let Some(dev) = init_device() else { return };
    let raw = dev.as_raw();
    assert_eq!(slot_of("ID3D12Device", "GetCopyableFootprints"), SLOT_COPY_FOOTPRINTS);

    let hooked: CopyFootprintsFn = unsafe { vtable_fn(raw, SLOT_COPY_FOOTPRINTS) };
    let orig: CopyFootprintsFn = unsafe { orig_slot_fn(raw, SLOT_COPY_FOOTPRINTS) };
    assert_ne!(hooked as usize, orig as usize, "slot 38 must be split-hooked");

    // PC callers route directly to host GetCopyableFootprints (no console descriptor translation).
    let swizzled = desc_bytes(&tex2d_desc(
        DXGI_FORMAT_R8G8B8A8_UNORM,
        256,
        256,
        1,
        3,
        1,
        D3D12_TEXTURE_LAYOUT(CONSOLE_LAYOUT_SWIZZLE),
        D3D12_RESOURCE_FLAG_NONE,
    ));
    let mut pc_total = 0u64;
    let mut host_total = 0u64;
    unsafe {
        pc_copy_footprints(hooked, raw, &swizzled, 0, 1, 0, null_mut(), null_mut(), null_mut(), &mut pc_total);
        orig(raw, &swizzled, 0, 1, 0, null_mut(), null_mut(), null_mut(), &mut host_total);
    }
    assert_eq!(host_total, u64::MAX, "host must reject console texture layout {CONSOLE_LAYOUT_SWIZZLE:#x}");
    assert_eq!(pc_total, host_total, "PC caller on slot 38 must pass straight through to host");

    // Standard PC-valid descriptors match host outputs across every subresource field.
    let cases = [
        ("buf 1024", desc_bytes(&buf_desc(1024)), 0, 1, 0),
        ("buf 300 at offset 512", desc_bytes(&buf_desc(300)), 0, 1, 512),
        (
            "tex2d rgba8 4 mips",
            desc_bytes(&tex2d_desc(
                DXGI_FORMAT_R8G8B8A8_UNORM,
                256,
                256,
                1,
                4,
                1,
                D3D12_TEXTURE_LAYOUT_UNKNOWN,
                D3D12_RESOURCE_FLAG_NONE,
            )),
            0,
            4,
            0,
        ),
        (
            "tex2d bc1 slice",
            desc_bytes(&tex2d_desc(
                DXGI_FORMAT_BC1_UNORM,
                255,
                129,
                2,
                3,
                1,
                D3D12_TEXTURE_LAYOUT_UNKNOWN,
                D3D12_RESOURCE_FLAG_NONE,
            )),
            1,
            3,
            512,
        ),
        (
            "tex3d rgba16f",
            desc_bytes(&tex3d_desc(
                DXGI_FORMAT_R16G16B16A16_FLOAT,
                16,
                12,
                8,
                2,
                D3D12_TEXTURE_LAYOUT_UNKNOWN,
            )),
            0,
            2,
            0,
        ),
    ];

    for (label, d, first, num, base) in &cases {
        let n = *num as usize;
        let host = capture_footprints(
            |l, r, rs, t| unsafe { orig(raw, d, *first, *num, *base, l, r, rs, t) },
            n,
        );
        let xcl = capture_footprints(
            |l, r, rs, t| unsafe { title_copy_footprints(hooked, raw, d, *first, *num, *base, l, r, rs, t) },
            n,
        );
        assert_eq!(xcl, host, "{label}: GetCopyableFootprints mismatch vs host");
        assert_ne!(host.total, u64::MAX, "{label}: expected valid host footprints");
    }

    // Console descriptors rejected by raw host driver fall back to `device_x::pc_resource_desc`.
    let console_cases = [
        ("console layout 0x101", swizzled, 0u32, 3u32),
        (
            "console format 116 (R10G10B10_7E3_A2_FLOAT)",
            desc_bytes(&tex2d_desc(
                DXGI_FORMAT(116),
                64,
                32,
                1,
                2,
                1,
                D3D12_TEXTURE_LAYOUT_UNKNOWN,
                D3D12_RESOURCE_FLAG_NONE,
            )),
            0,
            2,
        ),
        (
            "console depth format 118 (D16_UNORM_S8_UINT)",
            desc_bytes(&tex2d_desc(
                DXGI_FORMAT(118),
                128,
                128,
                1,
                1,
                1,
                D3D12_TEXTURE_LAYOUT_UNKNOWN,
                D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            )),
            0,
            2,
        ),
        (
            "console depth+UAV flags (0x6) D32_FLOAT",
            desc_bytes(&tex2d_desc(
                DXGI_FORMAT_D32_FLOAT,
                128,
                64,
                1,
                1,
                1,
                D3D12_TEXTURE_LAYOUT_UNKNOWN,
                D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            )),
            0,
            1,
        ),
    ];

    for (label, raw_d, first, num) in &console_cases {
        let mut raw_total = 0u64;
        unsafe { orig(raw, raw_d, *first, *num, 0, null_mut(), null_mut(), null_mut(), &mut raw_total) };
        assert_eq!(raw_total, u64::MAX, "{label}: expected raw host rejection before pc_resource_desc");

        let pc_d = device_x::pc_resource_desc(raw_d);
        let n = *num as usize;
        let expected = capture_footprints(
            |l, r, rs, t| unsafe { orig(raw, &pc_d, *first, *num, 0, l, r, rs, t) },
            n,
        );
        let got = capture_footprints(
            |l, r, rs, t| unsafe { title_copy_footprints(hooked, raw, raw_d, *first, *num, 0, l, r, rs, t) },
            n,
        );
        assert_eq!(got, expected, "{label}: fallback footprints mismatch vs host on pc_resource_desc");
        assert_ne!(expected.total, u64::MAX, "{label}: expected translated PC desc to succeed");
    }

    // Optional NULL output pointers: verify probe catches fallback even when `pTotalBytes` is NULL.
    let pc_swizzled = device_x::pc_resource_desc(&swizzled);
    let mut want_layouts = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 3];
    let mut want_rows = vec![0u32; 3];
    let mut want_row_sizes = vec![0u64; 3];
    let mut got_layouts = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 3];
    let mut got_rows = vec![0u32; 3];
    let mut got_row_sizes = vec![0u64; 3];
    unsafe {
        orig(raw, &pc_swizzled, 0, 3, 512, want_layouts.as_mut_ptr(), want_rows.as_mut_ptr(), want_row_sizes.as_mut_ptr(), null_mut());
        title_copy_footprints(
            hooked,
            raw,
            &swizzled,
            0,
            3,
            512,
            got_layouts.as_mut_ptr(),
            got_rows.as_mut_ptr(),
            got_row_sizes.as_mut_ptr(),
            null_mut(),
        );
        // All optional outputs NULL must also be tolerated cleanly.
        title_copy_footprints(hooked, raw, &swizzled, 0, 3, 0, null_mut(), null_mut(), null_mut(), null_mut());
    }
    for i in 0..3 {
        assert_eq!(got_layouts[i].Offset, want_layouts[i].Offset, "mip {i} offset mismatch with NULL pTotalBytes");
        assert_eq!(got_layouts[i].Footprint.Format, want_layouts[i].Footprint.Format);
        assert_eq!(got_layouts[i].Footprint.Width, want_layouts[i].Footprint.Width);
        assert_eq!(got_layouts[i].Footprint.Height, want_layouts[i].Footprint.Height);
        assert_eq!(got_layouts[i].Footprint.Depth, want_layouts[i].Footprint.Depth);
        assert_eq!(got_layouts[i].Footprint.RowPitch, want_layouts[i].Footprint.RowPitch);
        assert_eq!(got_rows[i], want_rows[i]);
        assert_eq!(got_row_sizes[i], want_row_sizes[i]);
    }

    // Invalid descriptor stays u64::MAX after fallback too.
    let invalid = desc_bytes(&tex2d_desc(
        DXGI_FORMAT_UNKNOWN,
        0,
        0,
        1,
        1,
        1,
        D3D12_TEXTURE_LAYOUT_UNKNOWN,
        D3D12_RESOURCE_FLAG_NONE,
    ));
    let mut xcl_invalid = 0u64;
    let mut host_invalid = 0u64;
    unsafe {
        orig(raw, &invalid, 0, 1, 0, null_mut(), null_mut(), null_mut(), &mut host_invalid);
        title_copy_footprints(hooked, raw, &invalid, 0, 1, 0, null_mut(), null_mut(), null_mut(), &mut xcl_invalid);
    }
    assert_eq!(host_invalid, u64::MAX);
    assert_eq!(xcl_invalid, u64::MAX);
}
