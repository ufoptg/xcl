use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Dxgi::*;

#[derive(serde::Deserialize)]
struct BufferReplay {
    program: String,
    memory: String,
    positions: String,
    position_byte_offset: usize,
    packed_byte_offset: usize,
    sgprs: Vec<u32>,
    groups: [u32; 3],
    vertices: usize,
}

#[test]
fn replay_sample_dxil_forms() {
    use super::super::runner;
    #[derive(serde::Deserialize)]
    struct Sample {
        programs: Vec<std::path::PathBuf>,
        memory: std::path::PathBuf,
        expected: std::path::PathBuf,
        sgprs: Vec<u32>,
        groups: [u32; 3],
        // Some fixtures provide only the known header and guard bytes. Compare their payload
        // between forms and verify that every active record writes its first output.
        payload_words: Option<[usize; 2]>,
        records: Option<[usize; 2]>,
        reverse_input_words: Option<[usize; 2]>,
        chain_program: Option<std::path::PathBuf>,
        single_dispatch: Option<bool>,
        // Atomic lanes can reserve output slots in any order within their group.
        unordered_group_outputs: Option<[usize; 2]>,
    }
    let Some(path) = std::env::var_os("XCL_SAMPLE_DXIL_FORMS") else { return };
    let path = std::path::Path::new(&path);
    let sample: Sample = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let dir = path.parent().unwrap();
    let words = |path| {
        let bytes = std::fs::read(dir.join(path)).unwrap();
        assert_eq!(bytes.len() % 4, 0);
        bytes.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect::<Vec<_>>()
    };
    let input = words(sample.memory);
    let expected = words(sample.expected);
    assert_eq!(input.len(), expected.len());
    assert!(input != expected, "fixture must exercise an observable write");
    let programs: Vec<_> = sample.programs.iter().map(|p| std::fs::read(dir.join(p)).unwrap()).collect();
    let mut times = vec![Vec::new(); programs.len()];
    let mut reference = None;
    for i in 0..programs.len() {
        let mut out = runner::dispatch(&programs[i], sample.groups, &sample.sgprs, &input).unwrap();
        assert_eq!(out.len(), expected.len());
        if let Some([first, lanes]) = sample.unordered_group_outputs {
            assert!(lanes > 0 && sample.groups[1..] == [1, 1]);
            let count = sample.groups[0] as usize * lanes;
            for block in out[first..first + count].chunks_exact_mut(lanes) {
                block.sort_unstable();
            }
        }
        for (j, (&actual, &want)) in out.iter().zip(&expected).enumerate() {
            if sample.payload_words.is_some_and(|[first, end]| (first..end).contains(&j)) { continue; }
            assert_eq!(actual, want, "form {i} word {j}");
        }
        if let Some([first, end]) = sample.payload_words {
            let [count, stride] = sample.records.unwrap();
            assert!(count > 0 && stride > 0 && first < end && end <= out.len());
            assert!(count <= (end-first)/stride);
            let mut distinct = std::collections::HashSet::new();
            for record in 0..count {
                let j = first + stride * record;
                assert_ne!(out[j], input[j], "form {i} active payload word {j}");
                distinct.insert(&out[j..j+stride]);
            }
            println!("PAYLOAD form={i} written_records={count} distinct_records={}", distinct.len());
            if let Some([src, width]) = sample.reverse_input_words {
                assert!(width > 0 && distinct.len() == count);
                let mut permuted = input.clone();
                for n in 0..count {
                    let from = src + (count - 1 - n) * width;
                    let to = src + n * width;
                    permuted[to..to+width].copy_from_slice(&input[from..from+width]);
                }
                let reversed = runner::dispatch(&programs[i], sample.groups, &sample.sgprs, &permuted).unwrap();
                for n in 0..count {
                    let from = first + (count - 1 - n) * stride;
                    let to = first + n * stride;
                    assert_eq!(&reversed[to..to+stride], &out[from..from+stride], "form {i} reversed record {n}");
                }
                for j in 0..expected.len() {
                    if !(first..end).contains(&j) {
                        let want = if (src..src+count*width).contains(&j) { permuted[j] } else { expected[j] };
                        assert_eq!(reversed[j], want, "form {i} reversed guard or header word {j}");
                    }
                }
            }
        }
        if let Some(prior) = &reference {
            assert!(&out == prior, "form {i} differs from the reference dispatch");
        } else {
            reference = Some(out);
        }
    }
    if let Some(path) = &sample.chain_program {
        let code = std::fs::read(dir.join(path)).unwrap();
        let c = direct_compute(&code, 0, super::super::HEAP_WINDOWS | 4, true).unwrap();
        assert!(c.dxil == programs[0] && c.samplers.is_empty());
        let chain = c.dispatch_chain.as_ref().expect("fixture should split");
        runner::profile_buffer_forms(&c.dxil, [chain, chain], sample.groups, &sample.sgprs, &input,
            reference.as_ref().unwrap(), 256 << 20).unwrap();
        return;
    }
    let single = sample.single_dispatch == Some(true);
    for round in 0..if single { 26 } else { 6 } {
        for pass in 0..programs.len() {
            let i = (round + pass) % programs.len();
            let row = runner::dispatch_timed(&programs[i], sample.groups, &sample.sgprs, &input, if single { 1 } else { 12 }).unwrap();
            if round >= 2 { times[i].extend_from_slice(if single { &row } else { &row[2..] }); }
        }
    }
    for (i, times) in times.iter_mut().enumerate() {
        times.sort_by(f64::total_cmp);
        println!("DXIL_TIME form={i} groups={:?} n={} median_us={:.3}", sample.groups, times.len(),
            500.0 * (times[times.len()/2-1] + times[times.len()/2]));
    }
}

#[test]
fn replay_sample_buffer_chain() {
    use crate::gpu::recomp::runner;
    let Some(path) = std::env::var_os("XCL_SAMPLE_BUFFER_CHAIN") else { return };
    let path = std::path::Path::new(&path);
    let dir = path.parent().unwrap();
    let cfg: BufferReplay = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let read = |name: &str| std::fs::read(dir.join(name)).unwrap();
    let words = |b: &[u8]| b.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect::<Vec<_>>();
    let mem = words(&read(&cfg.memory));
    let expected = words(&read(&cfg.positions));
    let compiled = direct_compute(&read(&cfg.program), 0, super::super::HEAP_WINDOWS | 4, true).unwrap();
    let chain = compiled.dispatch_chain.as_ref().unwrap();
    assert!(compiled.samplers.is_empty());
    let reference = runner::dispatch(&compiled.dxil, cfg.groups, &cfg.sgprs, &mem).unwrap();
    let p = cfg.position_byte_offset / 4;
    let n = cfg.packed_byte_offset / 4;
    assert_eq!(expected.len(), 3 * cfg.vertices);
    let check = |result: &[u32]| {
        for (i, (&inword, &outword)) in mem.iter().zip(result).enumerate() {
            if (p..p + expected.len()).contains(&i) {
                assert_eq!(outword, expected[i - p], "position word {i}");
            } else if (n..n + cfg.vertices).contains(&i) {
                assert_ne!(inword, outword, "unwritten packed output {i}");
            } else {
                assert_eq!(inword, outword, "unexpected write at word {i}");
            }
        }
    };
    check(&reference);
    if let Some(dir) = std::env::var_os("XCL_SAMPLE_ALTERNATE_CHAIN") {
        let dir = std::path::Path::new(&dir);
        assert!(std::fs::read(dir.join("whole.dxil")).unwrap() == compiled.dxil);
        let (count, words): (usize, u32) = serde_json::from_slice(&std::fs::read(dir.join("chain.json")).unwrap()).unwrap();
        let alternate = crate::gpu::dxil::kernel::DispatchChain {
            pieces: (0..count).map(|i| std::fs::read(dir.join(format!("piece-{i}.dxil"))).unwrap()).collect(),
            dwords_per_group: words,
        };
        let budget = std::env::var("XCL_SAMPLE_SCRATCH_BYTES").ok().map(|s| s.parse().unwrap()).unwrap_or(256 << 20);
        runner::profile_buffer_forms(&compiled.dxil, [chain, &alternate], cfg.groups, &cfg.sgprs, &mem, &reference, budget).unwrap();
        return;
    }
    let mut rows = [Vec::new(), Vec::new()];
    let most = std::env::var("XCL_SAMPLE_SCRATCH_BYTES").ok().map(|s| s.parse::<u64>().unwrap() /
        (u64::from(chain.dwords_per_group) * 4)).unwrap_or(u64::MAX).max(1);
    for round in 0..4 {
        for k in 0..2 {
            let split = (round + k) % 2;
            let times = if split == 0 {
                runner::dispatch_timed(&compiled.dxil, cfg.groups, &cfg.sgprs, &mem, 5).unwrap()
            } else {
                let (out, times) = runner::dispatch_split_samples(chain, cfg.groups, &cfg.sgprs, &mem, most, 5, true).unwrap();
                assert!(out == reference, "split output differs from whole");
                times
            };
            rows[split].extend_from_slice(&times[1..]);
        }
    }
    for (variant, row) in rows.iter_mut().enumerate() {
        row.sort_by(f64::total_cmp);
        println!("BUFFER_TIME variant={variant} vertices={} pieces={} words_per_group={} n={} median_us={:.3}",
            cfg.vertices, chain.pieces.len(), chain.dwords_per_group, row.len(), 500.0*(row[7]+row[8]));
    }
}

#[test]
fn profile_sample_cold_compute() {
    let Some(path) = std::env::var_os("XCL_SAMPLE_COLD_COMPUTE") else { return };
    let bytes = std::fs::read(path).unwrap();
    let translate_start = Instant::now();
    let compiled = direct_compute(&bytes, 0, super::super::HEAP_WINDOWS | 4, true).unwrap();
    println!("TRANSLATE bytes={} ms={:.3}", compiled.dxil.len(), translate_start.elapsed().as_secs_f64() * 1000.0);
    let workers: usize = std::env::var("XCL_SAMPLE_BUILD_WORKERS").unwrap_or_else(|_| "4".into()).parse().unwrap();
    assert!(workers > 0);
    let use_chain = std::env::var("XCL_SAMPLE_BUILD_CHAIN").as_deref() == Ok("1");
    let chain = compiled.dispatch_chain.as_deref();
    if let Some(dir) = std::env::var_os("XCL_SAMPLE_EXPORT_CHAIN") {
        let dir = std::path::Path::new(&dir);
        let chain = chain.expect("sample must split");
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("whole.dxil"), &compiled.dxil).unwrap();
        for (i, part) in chain.pieces.iter().enumerate() {
            std::fs::write(dir.join(format!("piece-{i}.dxil")), part).unwrap();
        }
        std::fs::write(dir.join("chain.json"), serde_json::to_vec(&(chain.pieces.len(), chain.dwords_per_group)).unwrap()).unwrap();
        return;
    }
    if std::env::var_os("XCL_SAMPLE_PIECE_LAYOUT").is_some() {
        if let Some(chain) = chain {
            println!("LAYOUT words={} pieces={} bytes={}", chain.dwords_per_group, chain.pieces.len(),
                chain.pieces.iter().map(Vec::len).sum::<usize>());
        } else {
            println!("LAYOUT ineligible");
        }
        return;
    }
    let parts: Vec<&[u8]> = match (use_chain, chain) {
        (true, Some(c)) => c.pieces.iter().map(Vec::as_slice).collect(),
        (true, None) => { println!("PROFILE ineligible bytes={}", compiled.dxil.len()); return; }
        (false, _) => vec![&compiled.dxil],
    };
    for part in &parts {
        crate::gpu::dxil::validate::validate(part).unwrap();
    }
    let factory: IDXGIFactory6 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }.unwrap();
    let adapter: IDXGIAdapter1 = unsafe { factory.EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE) }.unwrap();
    assert_eq!(unsafe { adapter.GetDesc1() }.unwrap().VendorId, 0x10de);
    let dev = crate::gpu::agility::create_device(&adapter, D3D_FEATURE_LEVEL_12_0).unwrap();
    let cache: ID3D12Device9 = dev.cast().unwrap();
    unsafe { cache.ShaderCacheControl(D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_DRIVER_MANAGED |
        D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_D3D_CACHE_FOR_DRIVER, D3D12_SHADER_CACHE_CONTROL_FLAG_DISABLE) }.unwrap();
    let rs = if use_chain { dispatch_chain_root_signature(&dev, &compiled.samplers) }
        else { abi_root_signature_with(&dev, false, &compiled.samplers) }.unwrap();
    let imported = std::env::var_os("XCL_SAMPLE_IMPORT_PSO").map(|dir| {
        (0..parts.len()).map(|i| std::fs::read(std::path::Path::new(&dir).join(format!("{i}.pso"))).unwrap()).collect::<Vec<_>>()
    });
    let mut order: Vec<_> = (0..parts.len()).collect();
    if std::env::var_os("XCL_SAMPLE_LARGEST_FIRST").is_some() {
        order.sort_by_key(|&i| std::cmp::Reverse(parts[i].len()));
    }
    let index = AtomicUsize::new(0);
    let start = Instant::now();
    let objects = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..workers.min(parts.len())).map(|_| scope.spawn(|| {
            let mut built = Vec::new();
            loop {
                let Some(&i) = order.get(index.fetch_add(1, Ordering::Relaxed)) else { break };
                let code = parts[i];
                let cached = imported.as_ref().map(|all| &all[i]);
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
                    CachedPSO: cached.map(|data| D3D12_CACHED_PIPELINE_STATE {
                        pCachedBlob: data.as_ptr().cast(), CachedBlobSizeInBytes: data.len(),
                    }).unwrap_or_default(),
                    ..Default::default()
                };
                let part_start = Instant::now();
                let result = unsafe { dev.CreateComputePipelineState::<ID3D12PipelineState>(&desc) };
                let part_time = part_start.elapsed();
                drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
                built.push((i, result.unwrap(), part_start.duration_since(start), part_time));
            }
            built
        })).collect();
        threads.into_iter().flat_map(|t| t.join().unwrap()).collect::<Vec<_>>()
    });
    println!("PROFILE parts={} bytes={} wall_ms={:.2} cache=disabled", objects.len(),
        parts.iter().map(|p| p.len()).sum::<usize>(), start.elapsed().as_secs_f64() * 1000.0);
    for (i, _, offset, elapsed) in &objects {
        println!("PIECE index={i} bytes={} start_ms={:.2} driver_ms={:.2}", parts[*i].len(),
            offset.as_secs_f64() * 1000., elapsed.as_secs_f64() * 1000.);
    }
    if let Some(dir) = std::env::var_os("XCL_SAMPLE_EXPORT_PSO") {
        let dir = std::path::Path::new(&dir);
        std::fs::create_dir_all(dir).unwrap();
        for (i, object, _, _) in objects {
            let blob = unsafe { object.GetCachedBlob() }.unwrap();
            let data = unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()) };
            assert!(!data.is_empty());
            std::fs::write(dir.join(format!("{i}.pso")), data).unwrap();
        }
    }
}
