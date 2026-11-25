//! Host-driver parity and contract tests for `ID3D12Device::CheckFeatureSupport` (#13),
//! `CreateReservedResource` (#30), `CreateReservedResource1` (#55), and `GetResourceTiling`
//! (#42) without console game files (Part of #480).

use std::ffi::c_void;
use std::mem::{size_of, size_of_val};
use std::sync::OnceLock;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows_sys::core::GUID as SysGuid;

use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::{layout, real, vram};
use xcl::util::HRESULT;

const E_INVALIDARG: HRESULT = 0x8007_0057u32 as i32;
const E_NOINTERFACE: HRESULT = 0x8000_4002u32 as i32;

/// Unknown IID used to exercise `creator!` fallback to `ID3D12Resource`.
const CONSOLE_UNKNOWN_IID: SysGuid = SysGuid::from_u128(0x11223344_5566_7788_99aa_bbccddeeff01);

type CheckFeatureSupportFn = unsafe extern "system" fn(*mut c_void, i32, *mut u8, u32) -> HRESULT;
type CreateReservedResourceFn =
    unsafe extern "system" fn(*mut c_void, *const D3D12_RESOURCE_DESC, D3D12_RESOURCE_STATES, *const D3D12_CLEAR_VALUE, *const SysGuid, *mut *mut c_void) -> HRESULT;
type CreateReservedResource1Fn = unsafe extern "system" fn(
    *mut c_void,
    *const D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_STATES,
    *const D3D12_CLEAR_VALUE,
    *mut c_void,
    *const SysGuid,
    *mut *mut c_void,
) -> HRESULT;
type GetResourceTilingFn = unsafe extern "system" fn(
    *mut c_void,
    *mut c_void,
    *mut u32,
    *mut D3D12_PACKED_MIP_INFO,
    *mut D3D12_TILE_SHAPE,
    *mut u32,
    u32,
    *mut D3D12_SUBRESOURCE_TILING,
);

fn init_device() -> Option<ID3D12Device> {
    static INIT: OnceLock<bool> = OnceLock::new();
    if !*INIT.get_or_init(|| real::init().is_ok()) {
        runner::skip_test("no GPU");
        return None;
    }
    let mut raw = std::ptr::null_mut();
    let hr = unsafe { real::create_device(&ID3D12Device::IID as *const _ as *const c_void, &mut raw) };
    assert!(hr >= 0 && !raw.is_null(), "real::create_device failed: {hr:#x}");
    Some(unsafe { ID3D12Device::from_raw(raw) })
}

fn slot_of(iface_name: &str, method: &str) -> usize {
    let iface = layout::find(iface_name).unwrap_or_else(|| panic!("missing {iface_name} in layout"));
    layout::slot(iface, method).unwrap_or_else(|| panic!("missing {iface_name}::{method} in layout"))
}

fn raw_bytes<T: Copy>(v: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, size_of::<T>()) }
}

fn sys_iid_resource() -> SysGuid {
    let g = ID3D12Resource::IID;
    SysGuid { data1: g.data1, data2: g.data2, data3: g.data3, data4: g.data4 }
}

fn tiled_tier(dev: &ID3D12Device) -> i32 {
    let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    unsafe {
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut opts as *mut _ as *mut c_void, size_of_val(&opts) as u32)
            .expect("D3D12_OPTIONS");
    }
    opts.TiledResourcesTier.0
}

fn resource_heap_tier(dev: &ID3D12Device) -> i32 {
    let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    unsafe {
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut opts as *mut _ as *mut c_void, size_of_val(&opts) as u32)
            .expect("D3D12_OPTIONS");
    }
    opts.ResourceHeapTier.0
}

fn tex2d_reserved_desc(format: DXGI_FORMAT, width: u64, height: u32, mips: u16, array: u16, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: width,
        Height: height,
        DepthOrArraySize: array,
        MipLevels: mips,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
        Flags: flags,
    }
}

fn tex3d_reserved_desc(format: DXGI_FORMAT, width: u64, height: u32, depth: u16, mips: u16) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
        Alignment: 0,
        Width: width,
        Height: height,
        DepthOrArraySize: depth,
        MipLevels: mips,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    }
}

fn buffer_reserved_desc(bytes: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
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
        Flags: flags,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TilingSnapshot {
    total_tiles: u32,
    num_standard_mips: u8,
    num_packed_mips: u8,
    num_tiles_for_packed_mips: u32,
    packed_start_tile: u32,
    tile_shape: (u32, u32, u32),
    subresource_tilings: Vec<(u32, u16, u16, u32)>,
}

fn query_tiling_xcl(dev: &ID3D12Device, res: &ID3D12Resource, subresource_cap: usize, first: u32) -> TilingSnapshot {
    let mut total = u32::MAX;
    let mut packed = D3D12_PACKED_MIP_INFO::default();
    let mut shape = D3D12_TILE_SHAPE::default();
    let mut count = subresource_cap as u32;
    let mut subs = vec![D3D12_SUBRESOURCE_TILING::default(); subresource_cap.max(1)];
    unsafe {
        dev.GetResourceTiling(
            res,
            Some(&mut total),
            Some(&mut packed),
            Some(&mut shape),
            Some(&mut count),
            first,
            if subresource_cap == 0 { std::ptr::null_mut() } else { subs.as_mut_ptr() },
        );
    }
    subs.truncate(count as usize);
    TilingSnapshot {
        total_tiles: total,
        num_standard_mips: packed.NumStandardMips,
        num_packed_mips: packed.NumPackedMips,
        num_tiles_for_packed_mips: packed.NumTilesForPackedMips,
        packed_start_tile: packed.StartTileIndexInOverallResource,
        tile_shape: (shape.WidthInTexels, shape.HeightInTexels, shape.DepthInTexels),
        subresource_tilings: subs
            .into_iter()
            .map(|s| (s.WidthInTiles, s.HeightInTiles, s.DepthInTiles, s.StartTileIndexInOverallResource))
            .collect(),
    }
}

fn query_tiling_orig(orig: GetResourceTilingFn, dev_raw: *mut c_void, res: &ID3D12Resource, subresource_cap: usize, first: u32) -> TilingSnapshot {
    let mut total = u32::MAX;
    let mut packed = D3D12_PACKED_MIP_INFO::default();
    let mut shape = D3D12_TILE_SHAPE::default();
    let mut count = subresource_cap as u32;
    let mut subs = vec![D3D12_SUBRESOURCE_TILING::default(); subresource_cap.max(1)];
    unsafe {
        orig(
            dev_raw,
            res.as_raw(),
            &mut total,
            &mut packed,
            &mut shape,
            &mut count,
            first,
            if subresource_cap == 0 { std::ptr::null_mut() } else { subs.as_mut_ptr() },
        );
    }
    subs.truncate(count as usize);
    TilingSnapshot {
        total_tiles: total,
        num_standard_mips: packed.NumStandardMips,
        num_packed_mips: packed.NumPackedMips,
        num_tiles_for_packed_mips: packed.NumTilesForPackedMips,
        packed_start_tile: packed.StartTileIndexInOverallResource,
        tile_shape: (shape.WidthInTexels, shape.HeightInTexels, shape.DepthInTexels),
        subresource_tilings: subs
            .into_iter()
            .map(|s| (s.WidthInTiles, s.HeightInTiles, s.DepthInTiles, s.StartTileIndexInOverallResource))
            .collect(),
    }
}

fn assert_feature_parity<T: Copy>(dev: &ID3D12Device, orig: CheckFeatureSupportFn, feature: D3D12_FEATURE, init: T, label: &str) {
    let mut xcl_val = init;
    let mut orig_val = init;
    let size = size_of::<T>() as u32;
    let xcl_hr = unsafe { dev.CheckFeatureSupport(feature, &mut xcl_val as *mut _ as *mut c_void, size) }.err().map_or(0, |e| e.code().0);
    let orig_hr = unsafe { orig(dev.as_raw(), feature.0, &mut orig_val as *mut _ as *mut u8, size) };
    assert_eq!(xcl_hr, orig_hr, "{label}: CheckFeatureSupport HRESULT mismatch");
    assert_eq!(raw_bytes(&xcl_val), raw_bytes(&orig_val), "{label}: CheckFeatureSupport payload mismatch");
}

fn assert_oversized_feature_fallback<T: Copy + Default>(dev: &ID3D12Device, orig: CheckFeatureSupportFn, feature: D3D12_FEATURE, init: T, label: &str) {
    let pc_size = size_of::<T>();
    let mut pc_expected = init;
    let orig_pc_hr = unsafe { orig(dev.as_raw(), feature.0, &mut pc_expected as *mut _ as *mut u8, pc_size as u32) };
    assert!(orig_pc_hr >= 0, "{label}: host CheckFeatureSupport failed at PC size {pc_size}: {orig_pc_hr:#x}");

    let mut orig_over = vec![0xAAu8; pc_size + 4];
    orig_over[..pc_size].copy_from_slice(raw_bytes(&init));
    let orig_over_hr = unsafe { orig(dev.as_raw(), feature.0, orig_over.as_mut_ptr(), orig_over.len() as u32) };
    assert!(orig_over_hr < 0, "{label}: expected host CheckFeatureSupport to reject oversized buffer ({} bytes)", orig_over.len());

    let mut xcl_over = vec![0xAAu8; pc_size + 4];
    xcl_over[..pc_size].copy_from_slice(raw_bytes(&init));
    let xcl_res = unsafe { dev.CheckFeatureSupport(feature, xcl_over.as_mut_ptr() as *mut c_void, xcl_over.len() as u32) };
    assert!(xcl_res.is_ok(), "{label}: XCL CheckFeatureSupport oversized fallback failed: {xcl_res:?}");
    assert_eq!(&xcl_over[..pc_size], raw_bytes(&pc_expected), "{label}: oversized fallback prefix mismatch");
    assert_eq!(&xcl_over[pc_size..], &[0u8; 4], "{label}: oversized fallback must zero trailing console bytes");
}

#[test]
fn check_feature_support_matches_host_driver_and_console_extensions() {
    let Some(dev) = init_device() else { return };
    let slot = slot_of("ID3D12Device", "CheckFeatureSupport");
    assert_eq!(slot, 13);
    let orig: CheckFeatureSupportFn = unsafe { std::mem::transmute(real::orig_fn(dev.as_raw(), slot)) };

    assert_feature_parity(&dev, orig, D3D12_FEATURE_D3D12_OPTIONS, D3D12_FEATURE_DATA_D3D12_OPTIONS::default(), "D3D12_OPTIONS");
    assert_feature_parity(&dev, orig, D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1::default(), "D3D12_OPTIONS1");
    assert_feature_parity(&dev, orig, D3D12_FEATURE_D3D12_OPTIONS2, D3D12_FEATURE_DATA_D3D12_OPTIONS2::default(), "D3D12_OPTIONS2");
    assert_feature_parity(&dev, orig, D3D12_FEATURE_D3D12_OPTIONS3, D3D12_FEATURE_DATA_D3D12_OPTIONS3::default(), "D3D12_OPTIONS3");
    assert_feature_parity(&dev, orig, D3D12_FEATURE_D3D12_OPTIONS4, D3D12_FEATURE_DATA_D3D12_OPTIONS4::default(), "D3D12_OPTIONS4");
    assert_feature_parity(&dev, orig, D3D12_FEATURE_D3D12_OPTIONS6, D3D12_FEATURE_DATA_D3D12_OPTIONS6::default(), "D3D12_OPTIONS6");
    assert_feature_parity(&dev, orig, D3D12_FEATURE_D3D12_OPTIONS7, D3D12_FEATURE_DATA_D3D12_OPTIONS7::default(), "D3D12_OPTIONS7");
    assert_feature_parity(&dev, orig, D3D12_FEATURE_ARCHITECTURE, D3D12_FEATURE_DATA_ARCHITECTURE::default(), "ARCHITECTURE");
    assert_feature_parity(&dev, orig, D3D12_FEATURE_ARCHITECTURE1, D3D12_FEATURE_DATA_ARCHITECTURE1::default(), "ARCHITECTURE1");
    assert_feature_parity(
        &dev,
        orig,
        D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT,
        D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT::default(),
        "GPU_VIRTUAL_ADDRESS_SUPPORT",
    );

    // D3D12_OPTIONS5 respects `XCL_RAYTRACING` override if configured, otherwise exact parity.
    let mut xcl_o5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
    let mut orig_o5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
    let xcl_o5_hr = unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &mut xcl_o5 as *mut _ as *mut c_void, size_of_val(&xcl_o5) as u32) };
    let orig_o5_hr = unsafe { orig(dev.as_raw(), D3D12_FEATURE_D3D12_OPTIONS5.0, &mut orig_o5 as *mut _ as *mut u8, size_of_val(&orig_o5) as u32) };
    assert_eq!(xcl_o5_hr.is_ok(), orig_o5_hr >= 0, "D3D12_OPTIONS5 status parity");
    if xcl_o5_hr.is_ok() {
        assert_eq!(xcl_o5.SRVOnlyTiledResourceTier3, orig_o5.SRVOnlyTiledResourceTier3);
        assert_eq!(xcl_o5.RenderPassesTier, orig_o5.RenderPassesTier);
        if let Some(tier) = xcl::state::config().raytracing_tier_override() {
            assert_eq!(xcl_o5.RaytracingTier.0, tier as i32);
        } else {
            assert_eq!(xcl_o5.RaytracingTier, orig_o5.RaytracingTier);
        }
    }

    for sm in [D3D_SHADER_MODEL_6_6, D3D_SHADER_MODEL_6_0] {
        assert_feature_parity(
            &dev,
            orig,
            D3D12_FEATURE_SHADER_MODEL,
            D3D12_FEATURE_DATA_SHADER_MODEL { HighestShaderModel: sm },
            "SHADER_MODEL",
        );
    }
    assert_feature_parity(
        &dev,
        orig,
        D3D12_FEATURE_ROOT_SIGNATURE,
        D3D12_FEATURE_DATA_ROOT_SIGNATURE { HighestVersion: D3D_ROOT_SIGNATURE_VERSION_1_1 },
        "ROOT_SIGNATURE",
    );

    let levels = [D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_12_0, D3D_FEATURE_LEVEL_12_1];
    let mut xcl_fl = D3D12_FEATURE_DATA_FEATURE_LEVELS {
        NumFeatureLevels: levels.len() as u32,
        pFeatureLevelsRequested: levels.as_ptr(),
        MaxSupportedFeatureLevel: D3D_FEATURE_LEVEL(0),
    };
    let mut orig_fl = xcl_fl;
    let xcl_fl_hr = unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &mut xcl_fl as *mut _ as *mut c_void, size_of_val(&xcl_fl) as u32) }
        .err()
        .map_or(0, |e| e.code().0);
    let orig_fl_hr = unsafe { orig(dev.as_raw(), D3D12_FEATURE_FEATURE_LEVELS.0, &mut orig_fl as *mut _ as *mut u8, size_of_val(&orig_fl) as u32) };
    assert_eq!(xcl_fl_hr, orig_fl_hr, "FEATURE_LEVELS HRESULT mismatch");
    assert_eq!(xcl_fl.MaxSupportedFeatureLevel, orig_fl.MaxSupportedFeatureLevel, "FEATURE_LEVELS MaxSupportedFeatureLevel mismatch");

    for fmt in [
        DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_BC1_UNORM,
        DXGI_FORMAT_D32_FLOAT,
        DXGI_FORMAT_R32_TYPELESS,
        DXGI_FORMAT_D24_UNORM_S8_UINT,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
    ] {
        assert_feature_parity(
            &dev,
            orig,
            D3D12_FEATURE_FORMAT_SUPPORT,
            D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: fmt, ..Default::default() },
            "FORMAT_SUPPORT",
        );
        let mut xcl_fi = D3D12_FEATURE_DATA_FORMAT_INFO { Format: fmt, PlaneCount: 0 };
        let mut orig_fi = D3D12_FEATURE_DATA_FORMAT_INFO { Format: fmt, PlaneCount: 0 };
        let xcl_fi_hr = unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &mut xcl_fi as *mut _ as *mut c_void, size_of_val(&xcl_fi) as u32) }
            .err()
            .map_or(0, |e| e.code().0);
        let orig_fi_hr = unsafe { orig(dev.as_raw(), D3D12_FEATURE_FORMAT_INFO.0, &mut orig_fi as *mut _ as *mut u8, size_of_val(&orig_fi) as u32) };
        assert_eq!(xcl_fi_hr, orig_fi_hr, "FORMAT_INFO HRESULT mismatch");
        assert_eq!((xcl_fi.Format, xcl_fi.PlaneCount), (orig_fi.Format, orig_fi.PlaneCount), "FORMAT_INFO field mismatch");
    }
    for (fmt, samples) in [(DXGI_FORMAT_R8G8B8A8_UNORM, 1), (DXGI_FORMAT_R8G8B8A8_UNORM, 4), (DXGI_FORMAT_D32_FLOAT, 4)] {
        assert_feature_parity(
            &dev,
            orig,
            D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS { Format: fmt, SampleCount: samples, ..Default::default() },
            "MULTISAMPLE_QUALITY_LEVELS",
        );
    }
    assert_feature_parity(
        &dev,
        orig,
        D3D12_FEATURE_PLACED_RESOURCE_SUPPORT_INFO,
        D3D12_FEATURE_DATA_PLACED_RESOURCE_SUPPORT_INFO {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            DestHeapProperties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Supported: Default::default(),
        },
        "PLACED_RESOURCE_SUPPORT_INFO",
    );

    // Console oversized structure fallback (`0x40` bytes for `D3D12_OPTIONS` vs PC `0x3C`, etc.).
    assert_oversized_feature_fallback(&dev, orig, D3D12_FEATURE_D3D12_OPTIONS, D3D12_FEATURE_DATA_D3D12_OPTIONS::default(), "D3D12_OPTIONS +4");
    assert_oversized_feature_fallback(&dev, orig, D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1::default(), "D3D12_OPTIONS1 +4");
    assert_oversized_feature_fallback(
        &dev,
        orig,
        D3D12_FEATURE_FORMAT_SUPPORT,
        D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: DXGI_FORMAT_R8G8B8A8_UNORM, ..Default::default() },
        "FORMAT_SUPPORT +4",
    );

    // Undersized buffer retry (`pc_size - 4`) truncates cleanly to caller buffer size.
    let pc_opt_size = size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>();
    let mut pc_opt = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    assert!(unsafe { orig(dev.as_raw(), D3D12_FEATURE_D3D12_OPTIONS.0, &mut pc_opt as *mut _ as *mut u8, pc_opt_size as u32) } >= 0);
    let mut short_buf = vec![0x55u8; pc_opt_size - 4];
    let short_res = unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, short_buf.as_mut_ptr() as *mut c_void, short_buf.len() as u32) };
    assert!(short_res.is_ok(), "undersized D3D12_OPTIONS fallback failed: {short_res:?}");
    assert_eq!(short_buf.as_slice(), &raw_bytes(&pc_opt)[..pc_opt_size - 4]);

    // Console-only formats (116..=120) map to their PC equivalents and restore the input format.
    for (console_fmt, pc_fmt) in [(118u32, 45u32), (119, 46), (120, 47), (116, 10), (117, 10)] {
        let mut xcl_fs = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: DXGI_FORMAT(console_fmt as i32), ..Default::default() };
        let mut orig_fs = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: DXGI_FORMAT(pc_fmt as i32), ..Default::default() };
        unsafe {
            dev.CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &mut xcl_fs as *mut _ as *mut c_void, size_of_val(&xcl_fs) as u32)
                .expect("console FORMAT_SUPPORT");
            assert!(orig(dev.as_raw(), D3D12_FEATURE_FORMAT_SUPPORT.0, &mut orig_fs as *mut _ as *mut u8, size_of_val(&orig_fs) as u32) >= 0);
        }
        assert_eq!(xcl_fs.Format.0, console_fmt as i32, "console FORMAT_SUPPORT must restore format");
        assert_eq!((xcl_fs.Support1, xcl_fs.Support2), (orig_fs.Support1, orig_fs.Support2));

        let mut xcl_ms = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS { Format: DXGI_FORMAT(console_fmt as i32), SampleCount: 4, ..Default::default() };
        let mut orig_ms = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS { Format: DXGI_FORMAT(pc_fmt as i32), SampleCount: 4, ..Default::default() };
        unsafe {
            dev.CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &mut xcl_ms as *mut _ as *mut c_void, size_of_val(&xcl_ms) as u32)
                .expect("console MULTISAMPLE_QUALITY_LEVELS");
            assert!(orig(dev.as_raw(), D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS.0, &mut orig_ms as *mut _ as *mut u8, size_of_val(&orig_ms) as u32) >= 0);
        }
        assert_eq!(xcl_ms.Format.0, console_fmt as i32);
        assert_eq!((xcl_ms.Flags, xcl_ms.NumQualityLevels), (orig_ms.Flags, orig_ms.NumQualityLevels));

        let mut xcl_fi = D3D12_FEATURE_DATA_FORMAT_INFO { Format: DXGI_FORMAT(console_fmt as i32), PlaneCount: 0 };
        let mut orig_fi = D3D12_FEATURE_DATA_FORMAT_INFO { Format: DXGI_FORMAT(pc_fmt as i32), PlaneCount: 0 };
        unsafe {
            dev.CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &mut xcl_fi as *mut _ as *mut c_void, size_of_val(&xcl_fi) as u32)
                .expect("console FORMAT_INFO");
            assert!(orig(dev.as_raw(), D3D12_FEATURE_FORMAT_INFO.0, &mut orig_fi as *mut _ as *mut u8, size_of_val(&orig_fi) as u32) >= 0);
        }
        assert_eq!(xcl_fi.Format.0, console_fmt as i32);
        assert_eq!(xcl_fi.PlaneCount, orig_fi.PlaneCount);
    }

    // Null data pointer matches host failure.
    let xcl_null = unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, std::ptr::null_mut(), 0) };
    let orig_null = unsafe { orig(dev.as_raw(), D3D12_FEATURE_D3D12_OPTIONS.0, std::ptr::null_mut(), 0) };
    assert_eq!(xcl_null.err().map_or(0, |e| e.code().0), orig_null);
}

#[test]
fn create_reserved_resource_and_v1_match_host_driver() {
    let Some(dev) = init_device() else { return };
    let slot0 = slot_of("ID3D12Device", "CreateReservedResource");
    let slot1 = slot_of("ID3D12Device5", "CreateReservedResource1");
    assert_eq!(slot0, 30);
    assert_eq!(slot1, 55);
    let orig0: CreateReservedResourceFn = unsafe { std::mem::transmute(real::orig_fn(dev.as_raw(), slot0)) };
    let orig1: CreateReservedResource1Fn = unsafe { std::mem::transmute(real::orig_fn(dev.as_raw(), slot1)) };
    let dev5 = dev.cast::<ID3D12Device5>().ok();
    let tier = tiled_tier(&dev);

    // 1. Buffer + textures (when tiled resources are supported).
    let mut shapes: Vec<(&str, D3D12_RESOURCE_DESC, D3D12_RESOURCE_STATES)> = vec![(
        "buffer 128 KiB",
        buffer_reserved_desc(128 * 1024, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_COMMON,
    )];
    if tier >= D3D12_TILED_RESOURCES_TIER_1.0 {
        shapes.extend([
            ("rgba8 8x8", tex2d_reserved_desc(DXGI_FORMAT_R8G8B8A8_UNORM, 8, 8, 1, 1, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON),
            ("rgba8 256x256", tex2d_reserved_desc(DXGI_FORMAT_R8G8B8A8_UNORM, 256, 256, 1, 1, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON),
            ("rgba8 256x256 9mips", tex2d_reserved_desc(DXGI_FORMAT_R8G8B8A8_UNORM, 256, 256, 9, 1, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON),
            ("bc1 256x256", tex2d_reserved_desc(DXGI_FORMAT_BC1_UNORM, 256, 256, 1, 1, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON),
            (
                "d32 8x8",
                tex2d_reserved_desc(DXGI_FORMAT_D32_FLOAT, 8, 8, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
            ),
        ]);
    }
    if tier >= D3D12_TILED_RESOURCES_TIER_3.0 {
        shapes.push(("rgba8 64x64x64 3d", tex3d_reserved_desc(DXGI_FORMAT_R8G8B8A8_UNORM, 64, 64, 64, 1), D3D12_RESOURCE_STATE_COMMON));
    }

    let iid = sys_iid_resource();
    for (name, desc, state) in shapes {
        let mut xcl_res: Option<ID3D12Resource> = None;
        let xcl_hr = unsafe { dev.CreateReservedResource(&desc, state, None, &mut xcl_res) }.err().map_or(0, |e| e.code().0);
        let mut orig_raw = std::ptr::null_mut();
        let orig_hr = unsafe { orig0(dev.as_raw(), &desc, state, std::ptr::null(), &iid, &mut orig_raw) };
        assert_eq!(xcl_hr, orig_hr, "{name}: CreateReservedResource HRESULT mismatch");
        assert_eq!(xcl_res.is_some(), !orig_raw.is_null(), "{name}: CreateReservedResource presence mismatch");
        if let Some(xr) = xcl_res {
            let or = unsafe { ID3D12Resource::from_raw(orig_raw) };
            assert_eq!(unsafe { xr.GetDesc() }, unsafe { or.GetDesc() }, "{name}: CreateReservedResource GetDesc mismatch");
            // `note_reserved` keeps `res_map` from invoking native `Map`/`GetHeapProperties`.
            let mut ptr = std::ptr::null_mut();
            let map_hr = unsafe { xr.Map(0, None, Some(&mut ptr)) }.err().map_or(0, |e| e.code().0);
            assert_eq!(map_hr, E_INVALIDARG, "{name}: reserved resource Map must return E_INVALIDARG");
        }

        if let Some(dev5) = &dev5 {
            let mut xcl_res1: Option<ID3D12Resource> = None;
            let xcl_hr1 = unsafe { dev5.CreateReservedResource1(&desc, state, None, None, &mut xcl_res1) }
                .err()
                .map_or(0, |e| e.code().0);
            let mut orig_raw1 = std::ptr::null_mut();
            let orig_hr1 = unsafe { orig1(dev.as_raw(), &desc, state, std::ptr::null(), std::ptr::null_mut(), &iid, &mut orig_raw1) };
            assert_eq!(xcl_hr1, orig_hr1, "{name}: CreateReservedResource1 HRESULT mismatch");
            if let Some(xr1) = xcl_res1 {
                let or1 = unsafe { ID3D12Resource::from_raw(orig_raw1) };
                assert_eq!(unsafe { xr1.GetDesc() }, unsafe { or1.GetDesc() }, "{name}: CreateReservedResource1 GetDesc mismatch");
                let mut ptr = std::ptr::null_mut();
                let map_hr = unsafe { xr1.Map(0, None, Some(&mut ptr)) }.err().map_or(0, |e| e.code().0);
                assert_eq!(map_hr, E_INVALIDARG, "{name}: CreateReservedResource1 Map must return E_INVALIDARG");
            }
        }
    }

    // 2. Unknown console IID fallback (`E_NOINTERFACE` from host -> `S_OK` + adopted `ID3D12Resource`).
    // Skip when the D3D12 debug layer is active: `D3D12SDKLayers.dll` fast-fails (`0xc0000409`)
    // inside `CreateReservedResource` when passed an unsupported `riid`.
    let debug_layer = dev.cast::<ID3D12DebugDevice>().is_ok();
    let fb_desc = buffer_reserved_desc(64 * 1024, D3D12_RESOURCE_FLAG_NONE);
    let xcl_slot0: CreateReservedResourceFn = unsafe { std::mem::transmute(*(*(dev.as_raw() as *const *const usize)).add(slot0)) };
    if !debug_layer {
        let mut orig_fb_raw = std::ptr::null_mut();
        let orig_fb_hr = unsafe { orig0(dev.as_raw(), &fb_desc, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), &iid, &mut orig_fb_raw) };
        assert!(orig_fb_hr >= 0 && !orig_fb_raw.is_null(), "host fallback baseline failed: {orig_fb_hr:#x}");
        let orig_fb_desc = unsafe { ID3D12Resource::from_raw(orig_fb_raw).GetDesc() };

        let mut orig_unk = std::ptr::null_mut();
        let orig_unk_hr = unsafe { orig0(dev.as_raw(), &fb_desc, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), &CONSOLE_UNKNOWN_IID, &mut orig_unk) };
        assert_eq!(orig_unk_hr, E_NOINTERFACE);
        assert!(orig_unk.is_null());

        let mut xcl_unk = std::ptr::null_mut();
        let xcl_unk_hr = unsafe { xcl_slot0(dev.as_raw(), &fb_desc, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), &CONSOLE_UNKNOWN_IID, &mut xcl_unk) };
        assert!(xcl_unk_hr >= 0 && !xcl_unk.is_null(), "CreateReservedResource unknown IID fallback failed: {xcl_unk_hr:#x}");
        let xcl_unk_res = unsafe { ID3D12Resource::from_raw(xcl_unk) };
        assert_eq!(unsafe { xcl_unk_res.GetDesc() }, orig_fb_desc);
        let mut ptr = std::ptr::null_mut();
        assert_eq!(unsafe { xcl_unk_res.Map(0, None, Some(&mut ptr)) }.err().map_or(0, |e| e.code().0), E_INVALIDARG);

        if dev5.is_some() {
            let xcl_slot1: CreateReservedResource1Fn = unsafe { std::mem::transmute(*(*(dev.as_raw() as *const *const usize)).add(slot1)) };
            let mut orig_unk1 = std::ptr::null_mut();
            let orig_unk1_hr = unsafe {
                orig1(dev.as_raw(), &fb_desc, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), std::ptr::null_mut(), &CONSOLE_UNKNOWN_IID, &mut orig_unk1)
            };
            assert_eq!(orig_unk1_hr, E_NOINTERFACE);

            let mut xcl_unk1 = std::ptr::null_mut();
            let xcl_unk1_hr = unsafe {
                xcl_slot1(dev.as_raw(), &fb_desc, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), std::ptr::null_mut(), &CONSOLE_UNKNOWN_IID, &mut xcl_unk1)
            };
            assert!(xcl_unk1_hr >= 0 && !xcl_unk1.is_null(), "CreateReservedResource1 unknown IID fallback failed: {xcl_unk1_hr:#x}");
            let xcl_unk1_res = unsafe { ID3D12Resource::from_raw(xcl_unk1) };
            assert_eq!(unsafe { xcl_unk1_res.GetDesc() }, orig_fb_desc);
            let mut ptr1 = std::ptr::null_mut();
            assert_eq!(unsafe { xcl_unk1_res.Map(0, None, Some(&mut ptr1)) }.err().map_or(0, |e| e.code().0), E_INVALIDARG);
        }
    }

    // 3. Invalid descriptors match host driver failure HRESULT.
    let mut bad_layout = tex2d_reserved_desc(DXGI_FORMAT_R8G8B8A8_UNORM, 64, 64, 1, 1, D3D12_RESOURCE_FLAG_NONE);
    bad_layout.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    let bad_format = tex2d_reserved_desc(DXGI_FORMAT_UNKNOWN, 64, 64, 1, 1, D3D12_RESOURCE_FLAG_NONE);
    for (label, bad) in [("TEXTURE_LAYOUT_UNKNOWN", bad_layout), ("DXGI_FORMAT_UNKNOWN", bad_format)] {
        let mut xr: Option<ID3D12Resource> = None;
        let x_hr = unsafe { dev.CreateReservedResource(&bad, D3D12_RESOURCE_STATE_COMMON, None, &mut xr) }.err().map_or(0, |e| e.code().0);
        let mut or = std::ptr::null_mut();
        let o_hr = unsafe { orig0(dev.as_raw(), &bad, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), &iid, &mut or) };
        assert!(x_hr < 0 && o_hr < 0, "{label}: expected failure, got xcl={x_hr:#x}, orig={o_hr:#x}");
        assert_eq!(x_hr, o_hr, "{label}: invalid CreateReservedResource HRESULT mismatch");

        if let Some(dev5) = &dev5 {
            let mut xr1: Option<ID3D12Resource> = None;
            let x_hr1 = unsafe { dev5.CreateReservedResource1(&bad, D3D12_RESOURCE_STATE_COMMON, None, None, &mut xr1) }
                .err()
                .map_or(0, |e| e.code().0);
            let mut or1 = std::ptr::null_mut();
            let o_hr1 = unsafe { orig1(dev.as_raw(), &bad, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), std::ptr::null_mut(), &iid, &mut or1) };
            assert_eq!(x_hr1, o_hr1, "{label}: invalid CreateReservedResource1 HRESULT mismatch");
        }
    }

    // Null `ppv` validation check matches host (`S_FALSE` on valid descriptor).
    let x_null_ppv = unsafe { xcl_slot0(dev.as_raw(), &fb_desc, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), &iid, std::ptr::null_mut()) };
    let o_null_ppv = unsafe { orig0(dev.as_raw(), &fb_desc, D3D12_RESOURCE_STATE_COMMON, std::ptr::null(), &iid, std::ptr::null_mut()) };
    assert_eq!(x_null_ppv, o_null_ppv, "CreateReservedResource(ppv=null) HRESULT mismatch");
}

#[test]
fn get_resource_tiling_matches_host_driver_and_d3d12_contract() {
    let Some(dev) = init_device() else { return };
    let slot = slot_of("ID3D12Device", "GetResourceTiling");
    assert_eq!(slot, 42);
    let orig: GetResourceTilingFn = unsafe { std::mem::transmute(real::orig_fn(dev.as_raw(), slot)) };
    let tier = tiled_tier(&dev);

    // 1. Reserved buffer tiling parity.
    let buf_desc = buffer_reserved_desc(128 * 1024, D3D12_RESOURCE_FLAG_NONE);
    let mut buf_res: Option<ID3D12Resource> = None;
    unsafe { dev.CreateReservedResource(&buf_desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buf_res) }.expect("reserved buffer");
    let buf_res = buf_res.unwrap();
    let xcl_buf = query_tiling_xcl(&dev, &buf_res, 1, 0);
    let orig_buf = query_tiling_orig(orig, dev.as_raw(), &buf_res, 1, 0);
    assert_eq!(xcl_buf, orig_buf, "reserved buffer GetResourceTiling mismatch");
    assert_eq!(xcl_buf.total_tiles, 2, "128 KiB reserved buffer must occupy 2 64 KiB tiles");

    // 2. Non-reserved committed texture returns 0 tiles on both XCL and host.
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let mut committed_desc = tex2d_reserved_desc(DXGI_FORMAT_R8G8B8A8_UNORM, 64, 64, 1, 1, D3D12_RESOURCE_FLAG_NONE);
    committed_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    let mut committed: Option<ID3D12Resource> = None;
    unsafe { dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &committed_desc, D3D12_RESOURCE_STATE_COMMON, None, &mut committed) }
        .expect("committed texture");
    let committed = committed.unwrap();
    let xcl_comm = query_tiling_xcl(&dev, &committed, 1, 0);
    let orig_comm = query_tiling_orig(orig, dev.as_raw(), &committed, 1, 0);
    assert_eq!(xcl_comm, orig_comm, "committed texture GetResourceTiling mismatch");
    assert_eq!(xcl_comm.total_tiles, 0);

    if tier < D3D12_TILED_RESOURCES_TIER_1.0 {
        return;
    }

    // 3. Reserved textures: host-driver parity + vendor-independent 64 KiB tiling invariants.
    struct TexCase {
        name: &'static str,
        desc: D3D12_RESOURCE_DESC,
        state: D3D12_RESOURCE_STATES,
        texel_or_block_bytes: u32,
        block_dim: u32,
    }
    let mut cases = vec![
        TexCase {
            name: "rgba8 8x8",
            desc: tex2d_reserved_desc(DXGI_FORMAT_R8G8B8A8_UNORM, 8, 8, 1, 1, D3D12_RESOURCE_FLAG_NONE),
            state: D3D12_RESOURCE_STATE_COMMON,
            texel_or_block_bytes: 4,
            block_dim: 1,
        },
        TexCase {
            name: "rgba8 256x256",
            desc: tex2d_reserved_desc(DXGI_FORMAT_R8G8B8A8_UNORM, 256, 256, 1, 1, D3D12_RESOURCE_FLAG_NONE),
            state: D3D12_RESOURCE_STATE_COMMON,
            texel_or_block_bytes: 4,
            block_dim: 1,
        },
        TexCase {
            name: "rgba8 256x256 9mips",
            desc: tex2d_reserved_desc(DXGI_FORMAT_R8G8B8A8_UNORM, 256, 256, 9, 1, D3D12_RESOURCE_FLAG_NONE),
            state: D3D12_RESOURCE_STATE_COMMON,
            texel_or_block_bytes: 4,
            block_dim: 1,
        },
        TexCase {
            name: "rgba8 256x256 array2",
            desc: tex2d_reserved_desc(DXGI_FORMAT_R8G8B8A8_UNORM, 256, 256, 1, 2, D3D12_RESOURCE_FLAG_NONE),
            state: D3D12_RESOURCE_STATE_COMMON,
            texel_or_block_bytes: 4,
            block_dim: 1,
        },
        TexCase {
            name: "bc1 256x256",
            desc: tex2d_reserved_desc(DXGI_FORMAT_BC1_UNORM, 256, 256, 1, 1, D3D12_RESOURCE_FLAG_NONE),
            state: D3D12_RESOURCE_STATE_COMMON,
            texel_or_block_bytes: 8,
            block_dim: 4,
        },
        TexCase {
            name: "d32 8x8",
            desc: tex2d_reserved_desc(DXGI_FORMAT_D32_FLOAT, 8, 8, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
            state: D3D12_RESOURCE_STATE_DEPTH_WRITE,
            texel_or_block_bytes: 4,
            block_dim: 1,
        },
    ];
    if tier >= D3D12_TILED_RESOURCES_TIER_3.0 {
        cases.push(TexCase {
            name: "rgba8 64x64x64 3d",
            desc: tex3d_reserved_desc(DXGI_FORMAT_R8G8B8A8_UNORM, 64, 64, 64, 1),
            state: D3D12_RESOURCE_STATE_COMMON,
            texel_or_block_bytes: 4,
            block_dim: 1,
        });
    }

    for c in &cases {
        let mut res: Option<ID3D12Resource> = None;
        unsafe { dev.CreateReservedResource(&c.desc, c.state, None, &mut res) }.unwrap_or_else(|e| panic!("{}: CreateReservedResource: {e}", c.name));
        let res = res.unwrap();
        let subresources = if c.desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            c.desc.MipLevels as usize
        } else {
            (c.desc.MipLevels as usize) * (c.desc.DepthOrArraySize as usize)
        };
        let xcl = query_tiling_xcl(&dev, &res, subresources, 0);
        let host = query_tiling_orig(orig, dev.as_raw(), &res, subresources, 0);
        assert_eq!(xcl, host, "{}: XCL GetResourceTiling != host driver", c.name);

        // Total-tiles-only query with optional out-pointers omitted.
        let mut total_only_xcl = 0u32;
        let mut total_only_orig = 0u32;
        unsafe {
            dev.GetResourceTiling(&res, Some(&mut total_only_xcl), None, None, None, 0, std::ptr::null_mut());
            orig(dev.as_raw(), res.as_raw(), &mut total_only_orig, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), 0, std::ptr::null_mut());
        }
        assert_eq!(total_only_xcl, total_only_orig, "{}: total-only query mismatch", c.name);
        assert_eq!(total_only_xcl, xcl.total_tiles);

        // Vendor-independent D3D12 tiling invariants (hold across AMD, NVIDIA, Intel, WARP).
        assert!(xcl.total_tiles >= 1, "{}: expected at least 1 tile", c.name);
        let mips_sum = xcl.num_standard_mips as u16 + xcl.num_packed_mips as u16;
        assert_eq!(mips_sum, c.desc.MipLevels, "{}: standard + packed mips != MipLevels", c.name);
        assert_eq!(xcl.subresource_tilings.len(), subresources, "{}: subresource tiling count mismatch", c.name);

        let mut standard_tiles = 0u32;
        for (idx, &(w, h, d, start)) in xcl.subresource_tilings.iter().enumerate() {
            if start == D3D12_PACKED_TILE {
                assert_eq!((w, h, d), (0, 0, 0), "{}: subresource {idx} packed tile must have 0x0x0 dimensions", c.name);
            } else {
                let count = w * h as u32 * d as u32;
                assert!(count >= 1, "{}: subresource {idx} non-packed tile count is 0", c.name);
                standard_tiles += count;
            }
        }
        assert_eq!(
            standard_tiles + xcl.num_tiles_for_packed_mips,
            xcl.total_tiles,
            "{}: standard tiles ({standard_tiles}) + packed ({}) != total ({})",
            c.name,
            xcl.num_tiles_for_packed_mips,
            xcl.total_tiles
        );

        let (tw, th, td) = xcl.tile_shape;
        assert_eq!(tw % c.block_dim, 0);
        assert_eq!(th % c.block_dim, 0);
        let tile_bytes = (tw / c.block_dim) * (th / c.block_dim) * td * c.texel_or_block_bytes;
        assert_eq!(tile_bytes, 65536, "{}: unexpected tile shape {:?} ({tile_bytes} B)", c.name, xcl.tile_shape);

        if c.desc.MipLevels > 2 {
            let xcl_slice = query_tiling_xcl(&dev, &res, 3, 2);
            let orig_slice = query_tiling_orig(orig, dev.as_raw(), &res, 3, 2);
            assert_eq!(xcl_slice, orig_slice, "{}: partial FirstSubresourceTilingToGet mismatch", c.name);
            assert_eq!(xcl_slice.subresource_tilings, xcl.subresource_tilings[2..5]);
        }
    }

    // 4. Console `vram::place` reserved texture path agrees with `orig_fn` `GetResourceTiling`.
    if resource_heap_tier(&dev) >= 2 && std::env::var_os("XCL_VRAM_ARENAS").is_none() {
        let mut rt_desc = tex2d_reserved_desc(DXGI_FORMAT_R8G8B8A8_UNORM, 256, 256, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        rt_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        let iid = sys_iid_resource();
        let mut placed_raw = std::ptr::null_mut();
        if let Some(hr) = unsafe { vram::place(dev.as_raw(), 0x1_0000_0000, &rt_desc, D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), Some(256 * 256 * 4), &iid, &mut placed_raw) } {
            assert!(hr >= 0 && !placed_raw.is_null(), "vram::place failed: {hr:#x}");
            let placed = unsafe { ID3D12Resource::from_raw(placed_raw) };
            let xcl_p = query_tiling_xcl(&dev, &placed, 1, 0);
            let orig_p = query_tiling_orig(orig, dev.as_raw(), &placed, 1, 0);
            assert_eq!(xcl_p, orig_p, "vram::place reserved resource GetResourceTiling mismatch");
            assert!(xcl_p.total_tiles >= 1);
            let mut ptr = std::ptr::null_mut();
            assert_eq!(unsafe { placed.Map(0, None, Some(&mut ptr)) }.err().map_or(0, |e| e.code().0), E_INVALIDARG);
            // Synchronize any pending `vram::Pages` `UpdateTileMappings` before releasing `placed`
            // so the D3D12 debug layer never sees a resource destroyed while tile binding is in flight.
            unsafe {
                let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
                let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
                q.ExecuteCommandLists(&[]);
                q.Signal(&fence, 1).expect("Signal");
                fence.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).expect("wait");
            }
            drop(placed);
            real::drain_gpu();
        }
    }
}
