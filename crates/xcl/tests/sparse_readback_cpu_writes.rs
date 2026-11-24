//! A copy back from sparse VRAM must preserve CPU writes that have not reached the GPU yet.

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use xcl::gpu::{agility, texture};
use xcl::hle::d3d12x::sparse;

#[test]
fn sparse_readback_preserves_an_unsubmitted_cpu_chunk() {
    const CHUNK: usize = 64;
    const BYTES: usize = 2 * CHUNK;
    const BASELINE: u8 = 0x33;
    const CPU_VALUE: u8 = 0x57;
    const GPU_VALUE: u8 = 0x9a;
    const PAGE: usize = 4096;
    const TILE: usize = 65536;

    unsafe {
        std::env::set_var("XCL_UMA_MB", "64");
        std::env::set_var("XCL_COHERENT_MB", "64");
        let Ok(dev) = agility::create_device(None, D3D_FEATURE_LEVEL_11_0) else {
            xcl::gpu::recomp::runner::skip_test("no D3D12 GPU");
            return;
        };
        let windows = sparse::init(&dev);
        let Some(&(index, raw, _)) = windows.first() else {
            xcl::gpu::recomp::runner::skip_test("no sparse window");
            return;
        };
        let base = xcl::gpu::uma::get().unwrap().window(index).0;
        let raw = raw as *mut std::ffi::c_void;
        let window = ID3D12Resource::from_raw_borrowed(&raw).unwrap();
        let (addr, _) = sparse::alloc_in(index, TILE, TILE).expect("sparse page");
        let title: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: BYTES as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let buffer = |ty, state| {
            let heap = D3D12_HEAP_PROPERTIES { Type: ty, ..Default::default() };
            let mut resource: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut resource).unwrap();
            resource.unwrap()
        };
        let source = buffer(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        let result = buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        let mut ptr = std::ptr::null_mut();
        source.Map(0, None, Some(&mut ptr)).unwrap();
        std::ptr::write_bytes(ptr.cast::<u8>(), GPU_VALUE, CHUNK);
        source.Unmap(0, None);
        let allocator: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).unwrap();
        list.CopyBufferRegion(window, (addr - base) as u64, &source, 0, CHUNK as u64);
        list.Close().unwrap();
        let read_gpu = || {
            allocator.Reset().unwrap();
            list.Reset(&allocator, None).unwrap();
            list.CopyBufferRegion(&result, 0, window, (addr - base) as u64, BYTES as u64);
            list.Close().unwrap();
            title.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            texture::wait(&dev, &title).unwrap();
            let mut mapped = std::ptr::null_mut();
            result.Map(0, None, Some(&mut mapped)).unwrap();
            let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), BYTES).to_vec();
            result.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            bytes
        };

        std::ptr::write_bytes(addr as *mut u8, BASELINE, BYTES);
        sparse::before_execute(&title, &[addr], PAGE);
        title.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        // CPU and GPU change different chunks of the requested readback range.
        std::ptr::write_bytes((addr + CHUNK) as *mut u8, CPU_VALUE, CHUNK);
        sparse::readback_after_execute(&title, &[(addr, BYTES)]);
        texture::wait(&dev, &title).unwrap();
        let cpu = std::slice::from_raw_parts(addr as *const u8, BYTES).to_vec();

        // The next submit must still upload the CPU's second chunk.
        sparse::before_execute(&title, &[addr], PAGE);
        let gpu = read_gpu();

        println!("after readback: {:#04x} / {:#04x}, after next upload: {:#04x} / {:#04x}",
            cpu[0], cpu[CHUNK], gpu[0], gpu[CHUNK]);
        assert!(cpu[..CHUNK].iter().all(|&b| b == GPU_VALUE), "the GPU's first chunk should reach the CPU");
        assert!(cpu[CHUNK..].iter().all(|&b| b == CPU_VALUE), "the readback must preserve the unsubmitted CPU chunk");
        assert!(gpu[..CHUNK].iter().all(|&b| b == GPU_VALUE));
        assert!(gpu[CHUNK..].iter().all(|&b| b == CPU_VALUE), "the preserved chunk must be uploaded on the next submit");

        // A partial readback must also publish the GPU's write when the CPU changed
        // bytes outside the requested range in the very same chunk.
        const WORD: usize = 8;
        const GPU_PARTIAL: u8 = 0x73;
        source.Map(0, None, Some(&mut ptr)).unwrap();
        std::ptr::write_bytes(ptr.cast::<u8>(), GPU_PARTIAL, WORD);
        source.Unmap(0, None);
        allocator.Reset().unwrap();
        list.Reset(&allocator, None).unwrap();
        list.CopyBufferRegion(window, (addr - base + WORD) as u64, &source, 0, WORD as u64);
        list.Close().unwrap();
        title.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        std::ptr::write_bytes(addr as *mut u8, CPU_VALUE, WORD);
        sparse::readback_after_execute(&title, &[(addr + WORD, WORD)]);
        texture::wait(&dev, &title).unwrap();
        let cpu = std::slice::from_raw_parts(addr as *const u8, CHUNK).to_vec();
        sparse::before_execute(&title, &[addr], PAGE);
        let gpu = read_gpu();
        for bytes in [&cpu[..], &gpu[..CHUNK]] {
            assert!(bytes[..WORD].iter().all(|&b| b == CPU_VALUE), "the adjacent CPU write must be retained");
            assert!(bytes[WORD..2 * WORD].iter().all(|&b| b == GPU_PARTIAL), "the partial GPU write must be visible");
            assert!(bytes[2 * WORD..].iter().all(|&b| b == GPU_VALUE), "other bytes of the chunk must stay unchanged");
        }
    }
}
