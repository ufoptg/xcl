//! Cross-process E2E verification for persistent two-pass (`MS` stream and `VS+GS`) driver PSO caching.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::{dxc, runner, IO_GS_PRIMS};
use xcl::hle::d3d12x::{pso_async, real};

const V: u64 = 256;

fn le32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn container(parts: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    let header = 32 + parts.len() * 4;
    let mut v = vec![0u8; header];
    v[0..4].copy_from_slice(b"DXBC");
    v[20..24].copy_from_slice(&1u32.to_le_bytes());
    v[28..32].copy_from_slice(&(parts.len() as u32).to_le_bytes());
    for (i, (cc, data)) in parts.iter().enumerate() {
        let off = v.len() as u32;
        v[32 + i * 4..36 + i * 4].copy_from_slice(&off.to_le_bytes());
        v.extend_from_slice(*cc);
        v.extend_from_slice(&(data.len() as u32).to_le_bytes());
        v.extend_from_slice(data);
    }
    let total = v.len() as u32;
    v[24..28].copy_from_slice(&total.to_le_bytes());
    v
}

fn fssm(code: &[u8], stages: &[(u8, u8)]) -> Vec<u8> {
    let code_at = 0x308usize;
    let mut v = vec![0u8; code_at];
    v[0..4].copy_from_slice(b"FSSM");
    v[4..8].copy_from_slice(&0x101u32.to_le_bytes());
    for q in (0x18..0xC0).step_by(8) {
        v[q..q + 8].copy_from_slice(&0xDEAD_BEEF_DEAD_BEEFu64.to_le_bytes());
    }
    v[0xC0..0xC8].copy_from_slice(&((code_at - 0x10) as u64).to_le_bytes());
    v[0xC8..0xD0].copy_from_slice(&(code.len() as u64 + 8).to_le_bytes());
    for (i, &(hw, api)) in stages.iter().enumerate() {
        let o = 0x100 + i * 0x88;
        let q = 0x0000_00C0_0002_0000u64 | ((api as u64) << 8) | hw as u64;
        v[o..o + 8].copy_from_slice(&q.to_le_bytes());
    }
    v.extend_from_slice(code);
    v.extend_from_slice(&[0x00, 0x00, 0x9f, 0xbf, 0x00, 0x00, 0x9f, 0xbf]);
    v.extend_from_slice(b"\x03\0\0\0metadata\0\0\0\0");
    let size = (v.len() - 0x10) as u32;
    v[8..12].copy_from_slice(&size.to_le_bytes());
    v
}

fn xbch(fssm_bytes: &[u8]) -> Vec<u8> {
    let z = miniz_oxide::deflate::compress_to_vec_zlib(fssm_bytes, 6);
    let mut v = b"XBCH".to_vec();
    le32(&mut v, 3);
    le32(&mut v, z.len() as u32);
    le32(&mut v, 0);
    le32(&mut v, fssm_bytes.len() as u32);
    le32(&mut v, 0);
    v.extend_from_slice(&z);
    v
}

fn program_header(kind: u32, size_dw: u32, bitcode_off: u32) -> Vec<u8> {
    let mut v = Vec::new();
    le32(&mut v, (kind << 16) | 0x66);
    le32(&mut v, size_dw);
    v.extend_from_slice(b"DXIL");
    le32(&mut v, 0x106);
    le32(&mut v, bitcode_off);
    le32(&mut v, 4);
    v
}

fn dxil_part(kind: u32, fssms: &[Vec<u8>]) -> Vec<u8> {
    let mut blobs: Vec<(u32, Vec<u8>)> = vec![(0x12, (0u8..16).collect())];
    for f in fssms {
        blobs.push((0x0D, xbch(f)));
    }
    let mut nested = program_header(kind, 0x8000_0007, 0x10);
    nested.extend_from_slice(b"NOBC");
    blobs.push((0x07, nested));
    let table = 0x18 + 12;
    let mut data_at = table + blobs.len() * 12;
    let mut xdat = b"XDAT".to_vec();
    le32(&mut xdat, blobs.len() as u32);
    le32(&mut xdat, table as u32);
    for (id, b) in &blobs {
        le32(&mut xdat, *id);
        le32(&mut xdat, data_at as u32);
        le32(&mut xdat, b.len() as u32);
        data_at += b.len();
    }
    for (_, b) in &blobs {
        xdat.extend_from_slice(b);
    }
    while (0x18 + xdat.len()) % 4 != 0 {
        xdat.push(0);
    }
    let nobc_at = 0x18 + xdat.len();
    let mut v = program_header(kind, ((nobc_at + 4) / 4) as u32, (nobc_at - 8) as u32);
    v.extend_from_slice(&xdat);
    v.extend_from_slice(b"NOBC");
    v
}

fn sig_part(entries: &[(&str, u32, u32, u32, u8)]) -> Vec<u8> {
    let mut d = Vec::new();
    le32(&mut d, entries.len() as u32);
    le32(&mut d, 8);
    let mut names = Vec::new();
    let base = 8 + entries.len() * 32;
    for &(name, sv, ct, reg, mask) in entries {
        let off = (base + names.len()) as u32;
        names.extend_from_slice(name.as_bytes());
        names.push(0);
        le32(&mut d, 0);
        le32(&mut d, off);
        le32(&mut d, 0);
        le32(&mut d, sv);
        le32(&mut d, ct);
        le32(&mut d, reg);
        d.extend_from_slice(&[mask, mask, 0, 0]);
        le32(&mut d, 0);
    }
    d.extend_from_slice(&names);
    d
}

fn mesh_psv(max_v: u16, max_p: u16, threads: [u32; 3]) -> Vec<u8> {
    let mut r = vec![0u8; 48];
    r[12..14].copy_from_slice(&max_v.to_le_bytes());
    r[14..16].copy_from_slice(&max_p.to_le_bytes());
    r[16..20].copy_from_slice(&64u32.to_le_bytes());
    r[20..24].copy_from_slice(&64u32.to_le_bytes());
    r[24] = 13; // MS
    r[27] = 2; // triangles
    r[36..40].copy_from_slice(&threads[0].to_le_bytes());
    r[40..44].copy_from_slice(&threads[1].to_le_bytes());
    r[44..48].copy_from_slice(&threads[2].to_le_bytes());
    let mut p = 48u32.to_le_bytes().to_vec();
    p.extend_from_slice(&r);
    le32(&mut p, 0);
    p
}

fn gs_psv(max_v: u16) -> Vec<u8> {
    let mut r = vec![0u8; 48];
    r[0..4].copy_from_slice(&3u32.to_le_bytes()); // input primitive: triangles
    r[4..8].copy_from_slice(&5u32.to_le_bytes()); // output topology: triangle strip
    r[16..20].copy_from_slice(&64u32.to_le_bytes());
    r[20..24].copy_from_slice(&64u32.to_le_bytes());
    r[24] = 2; // GS
    r[26..28].copy_from_slice(&max_v.to_le_bytes());
    let mut p = 48u32.to_le_bytes().to_vec();
    p.extend_from_slice(&r);
    le32(&mut p, 0);
    p
}

fn ms_code(alt: bool) -> Vec<u8> {
    let exp = |tgt: u64, en: u64, v: [u64; 4], done: u64| -> (&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>) {
        ("EXP", "ENC_EXP", vec![("TGT", tgt), ("EN", en), ("VSRC0", v[0]), ("VSRC1", v[1]), ("VSRC2", v[2]), ("VSRC3", v[3]), ("DONE", done)], None)
    };
    let mut list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = vec![
        ("S_BFE_U32", "ENC_SOP2", vec![("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)], Some(0x40018)),
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 193), ("SRC1", 128)], None),
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 193), ("SRC1", V)], None),
        ("V_LSHL_ADD_U32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 0), ("SRC1", 134), ("SRC2", V)], None),
        ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 124), ("SSRC0", 255)], Some((1 << 12) | 3)),
        ("S_SENDMSG", "ENC_SOPP", vec![("SIMM16", 9)], None),
        ("S_BARRIER", "ENC_SOPP", vec![], None),
        // Keep threads 0..3 active; build full-viewport triangle (-1,-1), (4,-1), (-1,4).
        ("V_CMPX_GT_U32", "ENC_VOPC", vec![("SRC0", 131), ("VSRC1", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 10), ("SRC0", 243)], None), // -1.0
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 11), ("SRC0", 243)], None), // -1.0
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 12), ("SRC0", 128)], None), // 0.0
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 13), ("SRC0", 242)], None), // 1.0
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 15), ("SRC0", 246)], None), // 4.0
        ("V_CMP_EQ_U32", "ENC_VOPC", vec![("SRC0", 129), ("VSRC1", 0)], None),
        ("V_CNDMASK_B32", "ENC_VOP2", vec![("VDST", 10), ("SRC0", V + 10), ("VSRC1", 15)], None),
        ("V_CMP_EQ_U32", "ENC_VOPC", vec![("SRC0", 130), ("VSRC1", 0)], None),
        ("V_CNDMASK_B32", "ENC_VOP2", vec![("VDST", 11), ("SRC0", V + 11), ("VSRC1", 15)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 14), ("SRC0", if alt { 240 } else { 242 })], None),
    ];
    list.push(exp(12, 0xF, [10, 11, 12, 13], 1));
    list.push(exp(32, 0xF, [14, 12, 12, 13], 0));
    list.push(("V_CMPX_EQ_U32", "ENC_VOPC", vec![("SRC0", 128), ("VSRC1", 0)], None));
    list.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 21), ("SRC0", 255)], Some((1 << 10) | (2 << 20))));
    list.push(exp(20, 1, [21, 0, 0, 0], 1));
    list.push(("S_ENDPGM", "ENC_SOPP", vec![], None));
    list.iter().flat_map(|(n, e, f, l)| encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}"))).collect()
}

fn synthetic_ms_container(alt: bool) -> Vec<u8> {
    let osg1 = sig_part(&[("SV_Position", 1, 3, 0, 0xF), ("TEXCOORD", 0, 3, 1, 0xF)]);
    let psv0 = mesh_psv(128, 128, [128, 1, 1]);
    let dxil = dxil_part(13, &[fssm(&ms_code(alt), &[(7, 6)])]);
    container(&[(b"SFI0", &[0; 8]), (b"OSG1", &osg1), (b"PSV0", &psv0), (b"HASH", &[0; 20]), (b"DXIL", &dxil)])
}

fn synthetic_vs_gs_containers() -> (Vec<u8>, Vec<u8>) {
    let e = |n: &str, enc: &str, f: &[(&str, u64)]| encode(n, enc, f, None).unwrap();
    let mut es = Vec::new();
    es.extend(e("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 6)]));
    // GS emits 1 full-viewport green triangle (0, 1, 0, 1) from lane 0 at LDS base s2.
    let mut gs = Vec::new();
    gs.extend(e("S_BFE_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)]));
    // Re-encode S_BFE_U32 with literal 0x80008 for wave primitive count in s3[15:8].
    gs.truncate(gs.len() - 4);
    gs.extend(encode("S_BFE_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)], Some(0x80008)).unwrap());
    gs.extend(e("S_BFM_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 0), ("SSRC1", 128)]));
    gs.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 50), ("SRC0", 2)])); // s2 LDS base
    gs.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 40), ("SRC0", 243)])); // -1.0
    gs.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 41), ("SRC0", 246)])); // 4.0
    gs.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 42), ("SRC0", 128)])); // 0.0
    gs.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 43), ("SRC0", 242)])); // 1.0
    let verts = [(40u64, 40u64), (41, 40), (40, 41)];
    for (k, (vx, vy)) in verts.into_iter().enumerate() {
        let o = 8 * k as u64;
        gs.extend(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", vx), ("DATA1", vy), ("OFFSET0", o), ("OFFSET1", o + 1)]));
        gs.extend(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", 42), ("DATA1", 43), ("OFFSET0", o + 2), ("OFFSET1", o + 3)]));
        gs.extend(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", 42), ("DATA1", 43), ("OFFSET0", o + 4), ("OFFSET1", o + 5)]));
        gs.extend(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", 42), ("DATA1", 43), ("OFFSET0", o + 6), ("OFFSET1", o + 7)]));
        gs.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x22)]));
    }
    gs.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x12)]));
    gs.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x3)]));
    gs.extend(e("S_ENDPGM", "ENC_SOPP", &[]));
    let vs_osg1 = sig_part(&[("SV_Position", 1, 3, 0, 0xF)]);
    let gs_osg1 = sig_part(&[("SV_Position", 1, 3, 0, 0xF), ("TEXCOORD", 0, 3, 1, 0xF)]);
    let vs = container(&[(b"SFI0", &[0; 8]), (b"OSG1", &vs_osg1), (b"DXIL", &dxil_part(1, &[fssm(&es, &[(2, 0)])]))]);
    let gs = container(&[(b"SFI0", &[0; 8]), (b"OSG1", &gs_osg1), (b"PSV0", &gs_psv(3)), (b"DXIL", &dxil_part(2, &[fssm(&gs, &[(3, 2), (4, 2)])]))]);
    (vs, gs)
}

#[repr(C, align(8))]
struct Sub<T> {
    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
    val: T,
}

#[repr(C)]
struct MeshStream {
    rs: Sub<*mut std::ffi::c_void>,
    ms: Sub<D3D12_SHADER_BYTECODE>,
    ps: Sub<D3D12_SHADER_BYTECODE>,
    blend: Sub<D3D12_BLEND_DESC>,
    mask: Sub<u32>,
    rast: Sub<D3D12_RASTERIZER_DESC>,
    depth: Sub<D3D12_DEPTH_STENCIL_DESC>,
    rtv: Sub<D3D12_RT_FORMAT_ARRAY>,
    samples: Sub<DXGI_SAMPLE_DESC>,
}

fn empty_rs(dev: &ID3D12Device) -> ID3D12RootSignature {
    unsafe {
        let mut ser = None;
        D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).unwrap();
        let ser = ser.unwrap();
        dev.CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize())).unwrap()
    }
}

fn build_ms_pso(dev: &ID3D12Device, rs: &ID3D12RootSignature, ms: &[u8], ps: &[u8]) -> ID3D12PipelineState {
    let mut blend = D3D12_BLEND_DESC::default();
    blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
    let rast = D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() };
    let mut rtv_formats = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: 1, ..Default::default() };
    rtv_formats.RTFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    let mut stream = MeshStream {
        rs: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, val: rs.as_raw() },
        ms: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, val: D3D12_SHADER_BYTECODE { pShaderBytecode: ms.as_ptr() as _, BytecodeLength: ms.len() } },
        ps: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, val: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr() as _, BytecodeLength: ps.len() } },
        blend: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, val: blend },
        mask: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, val: u32::MAX },
        rast: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, val: rast },
        depth: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, val: D3D12_DEPTH_STENCIL_DESC::default() },
        rtv: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, val: rtv_formats },
        samples: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, val: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 } },
    };
    let dev2: ID3D12Device2 = dev.cast().unwrap();
    let pso: ID3D12PipelineState = unsafe {
        dev2.CreatePipelineState(&D3D12_PIPELINE_STATE_STREAM_DESC {
            SizeInBytes: std::mem::size_of::<MeshStream>(),
            pPipelineStateSubobjectStream: &mut stream as *mut MeshStream as *mut std::ffi::c_void,
        })
    }
    .expect("CreatePipelineState two-pass MS");
    let raw = pso_async::resolve_wait(pso.as_raw() as usize);
    let tp = real::two_pass_of(raw).expect("two-pass MS entry");
    assert_eq!(tp.kind, "MS");
    pso
}

fn build_gs_pso(dev: &ID3D12Device, rs: &ID3D12RootSignature, vs: &[u8], gs: &[u8], ps: &[u8]) -> ID3D12PipelineState {
    let mut blend = D3D12_BLEND_DESC::default();
    blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
    let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
        VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr() as _, BytecodeLength: vs.len() },
        GS: D3D12_SHADER_BYTECODE { pShaderBytecode: gs.as_ptr() as _, BytecodeLength: gs.len() },
        PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr() as _, BytecodeLength: ps.len() },
        BlendState: blend,
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    desc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    let pso: ID3D12PipelineState = unsafe { dev.CreateGraphicsPipelineState(&desc) }.expect("CreateGraphicsPipelineState two-pass GS");
    unsafe { std::mem::ManuallyDrop::drop(&mut desc.pRootSignature) };
    let raw = pso_async::resolve_wait(pso.as_raw() as usize);
    let tp = real::two_pass_of(raw).expect("two-pass GS entry");
    assert_eq!(tp.kind, "GS");
    pso
}

fn buf_res(dev: &ID3D12Device, heap_ty: D3D12_HEAP_TYPE, size: u64, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
    let mut out: Option<ID3D12Resource> = None;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    let hp = D3D12_HEAP_PROPERTIES { Type: heap_ty, ..Default::default() };
    unsafe { dev.CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out) }.unwrap();
    out.unwrap()
}

fn info_queue_errors(dev: &ID3D12Device) -> (u64, u64) {
    let Ok(iq) = dev.cast::<ID3D12InfoQueue>() else { return (0, 0) };
    let n = unsafe { iq.GetNumStoredMessagesAllowedByRetrievalFilter() };
    let (mut other, mut corrupt_blob) = (0u64, 0u64);
    for i in 0..n {
        let mut sz = 0usize;
        if unsafe { iq.GetMessage(i, None, &mut sz) }.is_ok() && sz > 0 {
            let mut raw = vec![0u8; sz];
            let msg = raw.as_mut_ptr() as *mut D3D12_MESSAGE;
            if unsafe { iq.GetMessage(i, Some(msg), &mut sz) }.is_ok() {
                let (sev, id) = unsafe { ((*msg).Severity, (*msg).ID.0) };
                if sev == D3D12_MESSAGE_SEVERITY_ERROR || sev == D3D12_MESSAGE_SEVERITY_CORRUPTION {
                    // Intentional corrupt-blob tests feed unparseable garbage to trigger ID 884.
                    if id == D3D12_MESSAGE_ID_CREATEPIPELINESTATE_INVALIDCACHEDBLOB.0 {
                        corrupt_blob += 1;
                    } else {
                        other += 1;
                    }
                }
            }
        }
    }
    (other, corrupt_blob)
}

/// Execute both passes of `pso` (`compute` -> UAV barrier -> `DispatchMesh`) and return
/// the center pixel `[f32; 4]` (verifying uniform 8x8 RTV fill and zeroed-`res` ordering when `run_compute == false`).
fn draw_two_pass_and_read(dev: &ID3D12Device, pso: &ID3D12PipelineState, gs_prims: u32, run_compute: bool) -> [f32; 4] {
    const DIM: u32 = 8;
    const RES_SIZE: u64 = 128 << 10;
    let clear = [0.25f32, 0.25, 0.25, 0.25];
    let raw = pso_async::resolve_wait(pso.as_raw() as usize);
    let tp = real::two_pass_of(raw).expect("two-pass pipeline entry");
    let cpso = unsafe { ID3D12PipelineState::from_raw_borrowed(&(tp.compute as *mut std::ffi::c_void)) }.unwrap().clone();
    let crs = unsafe { ID3D12RootSignature::from_raw_borrowed(&(tp.compute_rs as *mut std::ffi::c_void)) }.unwrap().clone();
    let mrs = unsafe { ID3D12RootSignature::from_raw_borrowed(&(tp.mesh_rs as *mut std::ffi::c_void)) }.unwrap().clone();

    unsafe {
        let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList6 = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();

        let io_up = buf_res(dev, D3D12_HEAP_TYPE_UPLOAD, 4096, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        let io_gpu = buf_res(dev, D3D12_HEAP_TYPE_DEFAULT, 4096, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        let zero_up = buf_res(dev, D3D12_HEAP_TYPE_UPLOAD, RES_SIZE, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        let res_gpu = buf_res(dev, D3D12_HEAP_TYPE_DEFAULT, RES_SIZE, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        let mem_gpu = buf_res(dev, D3D12_HEAP_TYPE_DEFAULT, 4096, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        let cbv_heap: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 1, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, ..Default::default() })
            .unwrap();
        let samp_heap: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, NumDescriptors: 1, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, ..Default::default() })
            .unwrap();

        let mut p_io: *mut std::ffi::c_void = std::ptr::null_mut();
        io_up.Map(0, None, Some(&mut p_io)).unwrap();
        let io_words = std::slice::from_raw_parts_mut(p_io as *mut u32, 1024);
        io_words.fill(0);
        if gs_prims != 0 {
            io_words[IO_GS_PRIMS] = gs_prims;
        }
        io_up.Unmap(0, None);

        let mut p_z: *mut std::ffi::c_void = std::ptr::null_mut();
        zero_up.Map(0, None, Some(&mut p_z)).unwrap();
        std::ptr::write_bytes(p_z, 0, RES_SIZE as usize);
        zero_up.Unmap(0, None);

        let tex_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: DIM as u64,
            Height: DIM,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let cv = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_R32G32B32A32_FLOAT, Anonymous: D3D12_CLEAR_VALUE_0 { Color: clear } };
        let mut rtv_tex: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &tex_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, Some(&cv), &mut rtv_tex).unwrap();
        let rtv_tex = rtv_tex.unwrap();
        let rtv_heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() }).unwrap();
        let rtv_cpu = rtv_heap.GetCPUDescriptorHandleForHeapStart();
        dev.CreateRenderTargetView(&rtv_tex, None, rtv_cpu);

        let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total_bytes = 0u64;
        dev.GetCopyableFootprints(&tex_desc, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total_bytes));
        let rb = buf_res(dev, D3D12_HEAP_TYPE_READBACK, total_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

        let trans = |res: &ID3D12Resource, from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::ManuallyDrop::new(Some(res.clone())), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: from, StateAfter: to }),
            },
        };
        let uav_b = |res: &ID3D12Resource| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(Some(res.clone())) }),
            },
        };

        list.CopyBufferRegion(&io_gpu, 0, &io_up, 0, 4096);
        list.CopyBufferRegion(&res_gpu, 0, &zero_up, 0, RES_SIZE);
        let init_bars = [
            trans(&io_gpu, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            trans(&res_gpu, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            trans(&mem_gpu, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ];
        list.ResourceBarrier(&init_bars);
        for b in init_bars {
            let transition = std::mem::ManuallyDrop::into_inner(b.Anonymous.Transition);
            drop(std::mem::ManuallyDrop::into_inner(transition.pResource));
        }
        list.SetDescriptorHeaps(&[Some(cbv_heap.clone()), Some(samp_heap.clone())]);

        if run_compute {
            list.SetPipelineState(&cpso);
            list.SetComputeRootSignature(&crs);
            list.SetComputeRootUnorderedAccessView(0, io_gpu.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(1, res_gpu.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, mem_gpu.GetGPUVirtualAddress());
            list.Dispatch(1, 1, 1);
            let ub = [uav_b(&res_gpu)];
            list.ResourceBarrier(&ub);
            for b in ub {
                let barrier = std::mem::ManuallyDrop::into_inner(b.Anonymous.UAV);
                drop(std::mem::ManuallyDrop::into_inner(barrier.pResource));
            }
        }

        list.ClearRenderTargetView(rtv_cpu, &clear, None);
        list.OMSetRenderTargets(1, Some(&rtv_cpu), false, None);
        list.RSSetViewports(&[D3D12_VIEWPORT { Width: DIM as f32, Height: DIM as f32, MaxDepth: 1.0, ..Default::default() }]);
        list.RSSetScissorRects(&[windows::Win32::Foundation::RECT { right: DIM as i32, bottom: DIM as i32, ..Default::default() }]);
        list.SetPipelineState(pso);
        list.SetGraphicsRootSignature(&mrs);
        list.SetGraphicsRootUnorderedAccessView(0, io_gpu.GetGPUVirtualAddress());
        list.SetGraphicsRootUnorderedAccessView(1, res_gpu.GetGPUVirtualAddress());
        list.SetGraphicsRootUnorderedAccessView(2, mem_gpu.GetGPUVirtualAddress());
        list.DispatchMesh(tp.mesh_groups_per_record, 1, 1);

        let end_bars = [trans(&rtv_tex, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)];
        list.ResourceBarrier(&end_bars);
        for b in end_bars {
            let transition = std::mem::ManuallyDrop::into_inner(b.Anonymous.Transition);
            drop(std::mem::ManuallyDrop::into_inner(transition.pResource));
        }
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::ManuallyDrop::new(Some(rtv_tex.clone())), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::ManuallyDrop::new(Some(rb.clone())), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        std::mem::ManuallyDrop::into_inner(src.pResource);
        std::mem::ManuallyDrop::into_inner(dst.pResource);

        list.Close().unwrap();
        let cls = [Some(list.cast::<ID3D12CommandList>().unwrap())];
        q.ExecuteCommandLists(&cls);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let ev = windows::Win32::System::Threading::CreateEventW(None, false, false, None).unwrap();
        q.Signal(&fence, 1).unwrap();
        fence.SetEventOnCompletion(1, ev).unwrap();
        let wait = windows::Win32::System::Threading::WaitForSingleObject(ev, 5000);
        windows::Win32::Foundation::CloseHandle(ev).unwrap();
        assert_eq!(wait, windows::Win32::Foundation::WAIT_OBJECT_0);

        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut ptr)).unwrap();
        let row_pitch = fp.Footprint.RowPitch as usize;
        let pixel_at = |x: usize, y: usize| -> [f32; 4] {
            let p = (ptr as *const u8).add(y * row_pitch + x * 16) as *const [f32; 4];
            std::ptr::read_unaligned(p)
        };
        let center = pixel_at((DIM / 2) as usize, (DIM / 2) as usize);
        for y in 0..DIM as usize {
            for x in 0..DIM as usize {
                assert_eq!(pixel_at(x, y), center, "non-uniform pixel at ({x},{y})");
            }
        }
        rb.Unmap(0, None);
        center
    }
}

fn mesh_ready(dev: &ID3D12Device) -> bool {
    let mut o7 = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
    unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &mut o7 as *mut _ as _, std::mem::size_of_val(&o7) as u32) }.is_ok()
        && o7.MeshShaderTier.0 >= D3D12_MESH_SHADER_TIER_1.0
}

const CHILD_CASE: &str = "XCL_TEST_TWO_PASS_CACHE_CASE";
const TEST: &str = "two_pass_driver_cache_reuses_both_passes_and_recovers_corrupt_blobs";

#[test]
fn two_pass_driver_cache_reuses_both_passes_and_recovers_corrupt_blobs() {
    if let Ok(case) = std::env::var(CHILD_CASE) {
        return run_case(&case);
    }
    struct TempDir(PathBuf);
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let root = TempDir(std::env::temp_dir().join(format!("xcl-two-pass-{}-{nonce}", std::process::id())));
    let exe = std::env::current_exe().unwrap();
    let run = |case: &str, disk: bool, seed: bool| {
        let mut cmd = Command::new(&exe);
        cmd.args(["--exact", TEST, "--nocapture"]).env(CHILD_CASE, case)
            .env("XCL_DXIL_CACHE", &root.0)
            .env("XCL_PSO_DISK_CACHE", if disk { "1" } else { "0" })
            .env("XCL_ASYNC_PSO", "1").env("XCL_D3D12_DEBUG", "1")
            .env_remove("XCL_PSO_DISK_CACHE_MIN_MS");
        if seed {
            cmd.env("XCL_PSO_DISK_CACHE_MIN_MS", "0");
        }
        let out = cmd.output().unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{case}: {}
{stderr}", String::from_utf8_lossy(&out.stdout));
        let prefix = format!("SKIPPED {TEST}: ");
        stderr.lines().find_map(|line| line.strip_prefix(&prefix)).map(str::to_owned)
    };

    if let Some(reason) = run("disabled", false, true) {
        return runner::skip_test(&reason);
    }
    let driver = root.0.join("driver");
    assert!(!driver.exists(), "disabled cache must not create the driver directory");
    assert_eq!(run("seed", true, true), None);
    assert_eq!(run("warm", true, false), None);
    let files = fs::read_dir(&driver).unwrap().map(|e| e.unwrap().path()).collect::<Vec<_>>();
    let by_prefix = |prefix: &str| files.iter()
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with(prefix)).count();
    assert_eq!((by_prefix("tp-cs-"), by_prefix("tp-ms-")), (3, 2));
    for prefix in ["tp-cs-", "tp-ms-"] {
        for path in files.iter().filter(|p| p.file_name().unwrap().to_string_lossy().starts_with(prefix)) {
            fs::write(path, b"invalid driver blob").unwrap();
        }
        // Even with existing corrupt files, the disabled run must leave the files alone.
        assert_eq!(run("disabled", false, true), None);
        assert_eq!(run(prefix, true, true), None);
    }
    assert_eq!(run("warm", true, false), None);
}

fn run_case(case: &str) {
    if dxc::location().is_err() {
        return runner::skip_test("no DXC");
    }
    if xcl::hle::d3d12x::backend() != xcl::hle::d3d12x::Backend::Real {
        return runner::skip_test("no GPU");
    }
    if runner::wave64_programs_supported() != Ok(true) {
        return runner::skip_test("no wave64");
    }
    let raw = real::device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }.unwrap().clone();
    if !mesh_ready(&dev) {
        return runner::skip_test("no mesh shaders");
    }
    let ps = dxc::compile("struct V { float4 pos : SV_Position; float4 p0 : TEXCOORD0; };
float4 main(V v) : SV_Target0 { return v.p0; }
", "main", "ps_6_6", &[]).unwrap();
    let rs = empty_rs(&dev);
    let (vs, gs) = synthetic_vs_gs_containers();
    let before = real::two_pass_count();
    let (errors, corrupt) = info_queue_errors(&dev);
    let expected = match case {
        "disabled" => [(0, 0, 0); 3],
        "seed" => [(0, 0, 2), (0, 0, 3), (0, 0, 5)],
        "warm" => [(2, 0, 0), (3, 0, 0), (5, 0, 0)],
        "tp-cs-" => [(1, 1, 1), (1, 2, 2), (2, 3, 3)],
        "tp-ms-" => [(1, 1, 1), (2, 1, 1), (3, 2, 2)],
        _ => panic!("unknown case {case}"),
    };

    let ms1 = build_ms_pso(&dev, &rs, &synthetic_ms_container(false), &ps);
    assert_eq!(real::driver_cache_stats(), expected[0]);
    assert_eq!(draw_two_pass_and_read(&dev, &ms1, 0, false), [0.25; 4]);
    assert_eq!(draw_two_pass_and_read(&dev, &ms1, 0, true), [1.0, 0.0, 0.0, 1.0]);

    // The second MS shares the passthrough mesh graphics PSO but has a distinct NGG compute PSO.
    let ms2 = build_ms_pso(&dev, &rs, &synthetic_ms_container(true), &ps);
    assert_eq!(real::driver_cache_stats(), expected[1]);
    assert_eq!(draw_two_pass_and_read(&dev, &ms2, 0, true), [0.5, 0.0, 0.0, 1.0]);

    let gs = build_gs_pso(&dev, &rs, &vs, &gs, &ps);
    assert_eq!(real::driver_cache_stats(), expected[2]);
    assert_eq!(draw_two_pass_and_read(&dev, &gs, 1, false), [0.25; 4]);
    assert_eq!(draw_two_pass_and_read(&dev, &gs, 1, true), [0.0, 1.0, 0.0, 1.0]);
    assert_eq!(real::two_pass_count(), before + 3);
    drop((ms1, ms2, gs));
    assert_eq!(real::two_pass_count(), before, "releasing the title PSOs releases their two-pass state");

    if dev.cast::<ID3D12InfoQueue>().is_ok() {
        let (after_errors, after_corrupt) = info_queue_errors(&dev);
        assert_eq!(after_errors, errors, "unexpected D3D12 debug message in {case}");
        assert_eq!(after_corrupt - corrupt, expected[2].1, "unexpected corrupt-blob error count in {case}");
    }
}
