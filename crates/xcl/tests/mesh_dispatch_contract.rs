//! External contract test for the wave64-on-wave32 dispatcher (`mesh_dispatch`).
//!
//! Pins the scratch-layout arithmetic and the fixed-wave32 vs native-wave64
//! reference-model invariants that later lowerings rely on, without needing a GPU.

use xcl::gpu::recomp::mesh_dispatch::{
    model_fixed_wave32, model_native_wave64, validation_hlsl, ScratchLayout, GROUP_THREADS,
    LOGICAL_WAVE_LANES, LOGICAL_WAVES, MIN_RECORD_BYTES, RECORD_ALIGNMENT, SCALAR_REGISTERS,
};

const STRIDE: u32 = MIN_RECORD_BYTES.next_multiple_of(RECORD_ALIGNMENT);

fn deterministic_input(seed: u32) -> [u32; GROUP_THREADS as usize] {
    std::array::from_fn(|lane| {
        (lane as u32)
            .wrapping_mul(17)
            .wrapping_add(seed)
            .wrapping_mul(0x9E37_79B1)
            ^ ((lane as u32) >> 3)
    })
}

#[test]
fn layout_arithmetic_and_record_offsets() {
    let layout = ScratchLayout::new(512, STRIDE, 3).unwrap();
    assert_eq!(layout.base_bytes, 512);
    assert_eq!(layout.record_stride_bytes, STRIDE);
    assert_eq!(layout.record_count, 3);
    assert_eq!(layout.total_bytes, 512 + STRIDE * 3);

    assert_eq!(layout.record_base(0).unwrap(), 512);
    assert_eq!(layout.record_base(1).unwrap(), 512 + STRIDE);
    assert_eq!(layout.record_base(2).unwrap(), 512 + STRIDE * 2);
    assert!(layout.record_base(3).is_err(), "record >= count must be rejected");

    // Record-internal ordering: state, mailbox, result are distinct and ascending.
    let state0 = layout.state_offset(0, 0).unwrap();
    let state1 = layout.state_offset(0, 1).unwrap();
    let mailbox0 = layout.mailbox_offset(0, 0).unwrap();
    let mailbox_last = layout.mailbox_offset(0, GROUP_THREADS - 1).unwrap();
    let result0 = layout.result_offset(0, 0).unwrap();
    let result_last = layout.result_offset(0, GROUP_THREADS - 1).unwrap();

    assert!(state1 > state0, "wave 1 state must come after wave 0 state");
    assert!(mailbox0 > state1, "mailbox must come after both state blocks");
    assert!(mailbox_last < result0, "mailbox tail must precede result");
    assert!(result_last > result0, "result records must ascend with thread index");
    for i in 1..GROUP_THREADS {
        let prev = layout.mailbox_offset(0, i - 1).unwrap();
        let cur = layout.mailbox_offset(0, i).unwrap();
        assert_eq!(cur - prev, 4, "mailbox entries are 4 bytes apart");
    }
    // Result offsets walk one dword per thread too.
    assert_eq!(
        layout.result_offset(0, GROUP_THREADS - 1).unwrap() - layout.result_offset(0, 0).unwrap(),
        (GROUP_THREADS - 1) * 4
    );

    // Whole record stays within the record stride.
    assert!(result_last < layout.record_base(1).unwrap());
}

#[test]
fn layout_rejects_invalid_geometry() {
    // Not record-aligned: base and stride both need a multiple of 256.
    assert!(ScratchLayout::new(128, STRIDE, 1).is_err());
    assert!(ScratchLayout::new(0, 128, 1).is_err());

    // Stride too small even when aligned (2048 < MIN_RECORD_BYTES).
    assert!(ScratchLayout::new(0, 2048, 1).is_err());

    // Overflow of stride * record_count.
    assert!(ScratchLayout::new(0, u32::MAX - 255, 2).is_err());

    // Overflow of base + records.
    assert!(ScratchLayout::new(0xFFFF_FF00, 2304, 1).is_err());

    // Out-of-bounds logical waves and threads.
    let layout = ScratchLayout::new(0, STRIDE, 1).unwrap();
    assert!(layout.state_offset(0, LOGICAL_WAVES).is_err());
    assert!(layout.mailbox_offset(0, GROUP_THREADS).is_err());
    assert!(layout.result_offset(0, GROUP_THREADS).is_err());
    assert!(layout.state_offset(1, 0).is_err());
    assert!(layout.mailbox_offset(1, 0).is_err());
    assert!(layout.result_offset(1, 0).is_err());
}

#[test]
fn fixed_wave32_model_matches_native_wave64_reference() {
    for seed in [0u32, 1, u32::MAX] {
        let input = deterministic_input(seed);
        let fixed = model_fixed_wave32(&input);
        let native = model_native_wave64(&input);
        assert_eq!(fixed, native, "reference models must agree (seed {seed})");
    }
}

#[test]
fn fixed_wave32_model_invariants_and_wave_epoch_state() {
    let input = deterministic_input(0);
    let (values, states) = model_fixed_wave32(&input);

    assert_eq!(states.len(), LOGICAL_WAVES as usize);
    // Wave 0 activates for 2 epochs, wave 1 for 3 (epoch < wave + 2).
    let expected_epochs = [2u32, 3u32];
    for (wave, state) in states.iter().enumerate() {
        assert_eq!(state.epoch, expected_epochs[wave], "wave {wave} epoch");
        assert_eq!(state.pc, if wave == 0 { 0x10 } else { 0x20 });
        assert_eq!(state.scc, wave as u32);
        // Scalar register 0 tracks the wave's lane-0 thread value.
        let lane0 = wave * LOGICAL_WAVE_LANES as usize;
        assert_eq!(state.scalar[0], values[lane0], "wave {wave} scalar[0]");
        for reg in &state.scalar[1..] {
            assert_eq!(*reg, 0, "wave {wave}: scalar[1..] must stay zero");
        }
        // EXEC stayed at the default sentinel in both models.
        assert_eq!(state.exec, [u32::MAX; 2]);
        assert_eq!(state.scalar.len(), SCALAR_REGISTERS as usize);
    }

    // Wave 0 runs 2 epochs (final vcc from epoch 1); wave 1 runs 3 (epoch 2).
    assert_eq!(
        states[0].vcc,
        [0xaaaa_aaaa ^ 1, 0x5555_5555 ^ 1],
        "wave 0 vcc after last epoch"
    );
    assert_eq!(
        states[1].vcc,
        [0xaaaa_aaaa ^ 2, 0x5555_5555 ^ 2],
        "wave 1 vcc after last epoch"
    );

    // Wave 0 ran 2 epochs, wave 1 ran 3, so values must differ between the halves.
    assert_ne!(&values[..LOGICAL_WAVE_LANES as usize], &values[LOGICAL_WAVE_LANES as usize..]);
    // A thread that is inactive (wave 1 at epoch 2 does run; wave 0 stops mutating
    // after epoch 1) must still have evolved from epoch 1's partner value.
    let thread = 7usize; // wave 0, lane 7 (partner 39 in the high 32 of wave 0)
    assert_ne!(
        values[thread], input[thread],
        "thread {thread} must have consumed its partner's mailbox value"
    );
}

#[test]
fn validation_hlsl_uses_only_unconditional_rendezvous_and_scratch() {
    let hlsl = validation_hlsl();
    assert_eq!(
        hlsl.matches("DeviceMemoryBarrierWithGroupSync();").count(),
        3,
        "three unconditional rendezvous barriers (in/out of each epoch loop)"
    );
    assert!(!hlsl.contains("groupshared"), "scratch must stay external to LDS");
    assert!(!hlsl.contains("xcl_lds"), "scratch must stay external to xcl_lds");
    assert!(
        hlsl.contains("RWByteAddressBuffer scratch"),
        "scratch must be a byte-address UAV outside xcl_lds"
    );
    assert!(
        hlsl.contains("RWStructuredBuffer<uint> io"),
        "record base must come from the io input"
    );
    assert!(
        hlsl.contains("[numthreads("),
        "the shader must declare its thread count"
    );
    // Rendezvous must be unconditional: no barrier guarded by epoch/wave.
    assert!(
        !hlsl.contains("if (epoch < wave + 2u) DeviceMemoryBarrier"),
        "no barrier may be conditionally skipped by active-epoch rules"
    );
}
