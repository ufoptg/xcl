//! Contract test for `xcl::gpu::adapter::name`: it must return exactly the UTF-16
//! prefix of `DXGI_ADAPTER_DESC1::Description` up to the first NUL, capped at the
//! fixed array length. No device, GPU or DXGI factory involved.

use windows::Win32::Graphics::Dxgi::DXGI_ADAPTER_DESC1;

use xcl::gpu::adapter;

/// A `DXGI_ADAPTER_DESC1` whose `Description` is exactly `s` followed by NUL padding
/// (truncated to the array length, so `s` longer than 128 UTF-16 code units is
/// kept untruncated at the boundary – no trailing NUL is guaranteed).
fn desc_with_name(s: &str) -> DXGI_ADAPTER_DESC1 {
    let mut desc = DXGI_ADAPTER_DESC1::default();
    let wide: Vec<u16> = s.encode_utf16().collect();
    let n = wide.len().min(desc.Description.len());
    desc.Description[..n].copy_from_slice(&wide[..n]);
    desc
}

#[test]
fn name_with_trailing_nul_padding_returns_the_prefix() {
    let desc = desc_with_name("AMD Radeon™ Graphics");
    assert_eq!(adapter::name(&desc), "AMD Radeon™ Graphics");
}

#[test]
fn name_truncated_at_array_boundary_returns_the_full_array() {
    // 128 UTF-16 code units, no NUL anywhere: the whole array is the name.
    let s = "a".repeat(128);
    let desc = desc_with_name(&s);
    assert_eq!(adapter::name(&desc), s);
    // One code unit short of a NUL still yields exactly the prefix.
    let s = "b".repeat(127);
    assert_eq!(adapter::name(&desc_with_name(&s)), s);
}

#[test]
fn name_with_nul_in_the_middle_returns_the_prefix_only() {
    let desc = desc_with_name("AMD Radeon\0not-really");
    assert_eq!(adapter::name(&desc), "AMD Radeon");
}

#[test]
fn name_with_all_zero_array_returns_empty() {
    let desc = DXGI_ADAPTER_DESC1::default();
    assert_eq!(adapter::name(&desc), "");
    assert_eq!(adapter::name(&desc_with_name("")), "");
    // An array that is all NUL but not freshly zero-initialised is the same thing.
    let mut desc = desc_with_name("AMD Radeon");
    for c in desc.Description.iter_mut() {
        *c = 0;
    }
    assert_eq!(adapter::name(&desc), "");
}

#[test]
fn name_with_nul_terminated_empty_string_returns_empty() {
    let mut desc = DXGI_ADAPTER_DESC1::default();
    desc.Description[0] = 0;
    desc.Description[1] = 'x' as u16;
    assert_eq!(adapter::name(&desc), "");
}

#[test]
fn name_with_non_bmp_code_units_round_trips() {
    // BMP code units beyond ASCII (e.g. the ™ used by AMD descriptions) must come
    // back exactly, not be lost or mangled.
    let desc = desc_with_name("Intel® Arc™ A770");
    assert_eq!(adapter::name(&desc), "Intel® Arc™ A770");
}
