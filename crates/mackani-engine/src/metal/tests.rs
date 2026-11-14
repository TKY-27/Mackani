use super::fixture::{TempFile, assert_sums, random, synthetic};
use super::*;
use crate::{Kernel, MappedModel};
use std::panic::{AssertUnwindSafe, catch_unwind};

/// The calibrated share table is for the machine it was measured on only:
/// an Apple M5 with 4 + 6 CPU cores and 10 GPU cores, run with 10 CPU
/// workers. Another chip, an M5 variant with other core counts, one whose
/// GPU cores cannot be read, or another worker count runs GPU-only.
#[test]
fn share_table_applies_only_to_the_calibrated_machine() {
    let machine = |gpu_cores: Option<usize>, levels: &[usize]| crate::Machine {
        gpu_cores,
        core_levels: levels.to_vec(),
        ..crate::Machine::default()
    };
    let measured = machine(Some(10), &[4, 6]);
    assert!(share_table_applies(
        calibrated_machine("Apple M5", &measured),
        10
    ));
    for threads in [1, 4, 9, 16] {
        assert!(!share_table_applies(true, threads), "{threads}");
    }
    for (device, facts) in [
        ("Apple M4", machine(Some(10), &[4, 6])),
        ("Apple M5 Max", machine(Some(10), &[4, 6])),
        ("Apple M1", machine(Some(8), &[4, 4])),
        ("Apple Paravirtual device", machine(Some(10), &[4, 6])),
        ("Apple M5", machine(Some(8), &[4, 6])),
        ("Apple M5", machine(Some(10), &[3, 6])),
        ("Apple M5", machine(None, &[4, 6])),
        ("Apple M5", machine(Some(10), &[])),
    ] {
        assert!(!calibrated_machine(device, &facts), "{device} {facts:?}");
    }
}

#[test]
fn cpu_rows_are_whole_tiles_and_leave_the_gpu_a_tile() {
    // Below 64 rows the GPU's minimum tile leaves no whole tile for the CPU.
    for n in [1, 11, 31, 32, 33, 40, 63] {
        assert_eq!(cpu_rows(CpuShare::MAX, n), 0, "n={n}");
    }
    assert_eq!(cpu_rows(CpuShare::MAX, 64), 32);
    assert_eq!(cpu_rows(CpuShare::MAX, 75), 32);
    assert_eq!(cpu_rows(CpuShare::MAX, 104), 64);
    assert_eq!(cpu_rows(0.3, 75), 0);
    assert_eq!(cpu_rows(0.3, 320), 96);
    assert_eq!(cpu_rows(0.0, 5120), 0);
    assert_eq!(cpu_rows(0.08, 5120), 384);
}

/// A CPU part that panics or fails after the GPU part was committed: the call
/// returns only once the GPU work has completed (its rows are in the scratch
/// outputs), and the same engine then serves a normal call exactly.
#[test]
fn gpu_work_completes_before_a_failing_cpu_part_returns_and_the_engine_is_reused() {
    let Some(mut engine) = MetalEngine::new().unwrap() else {
        println!("skipped: no Metal device");
        return;
    };
    let mut cpu = Engine::new(Kernel::fastest(), 4).unwrap();
    // Enough GPU work (eight 4096-row matrices at M=16) to still be running
    // when the panic has unwound, unless something waits for it.
    let (n, k, m) = (4096, 5120, 16);
    let file = TempFile::new(&synthetic(8, "warp", n, n, k, false));
    // SAFETY: the private file is never modified and outlives all mapping owners.
    let model = unsafe { MappedModel::open(&file.0) }.unwrap();
    let matrix = model.matrix("linear").unwrap();
    let small_file = TempFile::new(&synthetic(4, "row", 9, 11, k, false));
    // SAFETY: as above.
    let small_model = unsafe { MappedModel::open(&small_file.0) }.unwrap();
    let small = small_model.matrix("linear").unwrap();
    let refs = [&matrix; 8];
    let mut seed = 0x0dd5_eed0_0000_0007;
    let mut input = || {
        let activation: Vec<i8> = (0..m * k).map(|_| random(&mut seed) as i8).collect();
        let scales: Vec<i32> = (0..m * k / 512).map(|_| random(&mut seed) as i32).collect();
        (activation, scales)
    };
    engine.set_cpu_share(Some(0.5)).unwrap();
    let gpu_rows = n - cpu_rows(0.5, n);
    for panics in [true, false] {
        engine.scratch_outputs.clear();
        let (activation, scales) = input();
        let expected = cpu.weighted_sums(&matrix, &activation, &scales).unwrap();
        let mut failing = |_: &[(&Matrix, std::ops::Range<usize>)]| -> Result<Vec<Vec<i64>>> {
            if panics {
                panic!("expected CPU part panic");
            }
            Err(invalid("expected CPU part error"))
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            engine.shared(Some((&mut failing, 4)), &refs, &activation, &scales)
        }));
        match result {
            Err(_) => assert!(panics),
            Ok(result) => assert!(!panics && result.is_err()),
        }
        let gpu_part: Vec<i64> = (0..m)
            .flat_map(|row| expected[row * n..row * n + gpu_rows].iter().copied())
            .collect();
        for (index, output) in engine.scratch_outputs.iter().enumerate() {
            // SAFETY: `shared` has returned or unwound, and its `InFlight`
            // guard waited for the command buffer, so no GPU work writes it.
            let sums = unsafe { download(output.as_ref().unwrap(), m, gpu_rows, &[]) };
            assert_sums(
                &sums.unwrap(),
                &gpu_part,
                &format!("GPU rows of matrix {index} after panics={panics}"),
            );
        }
        // The engine is reusable: new inputs overwrite the scratch buffers.
        let (activation, scales) = input();
        let expected = cpu.weighted_sums(&matrix, &activation, &scales).unwrap();
        let sums = engine
            .weighted_sums_split(&mut cpu, &refs, &activation, &scales)
            .unwrap();
        for (index, sums) in sums.iter().enumerate() {
            assert_sums(sums, &expected, &format!("reuse {index} panics={panics}"));
        }
        // Fewer than 64 rows leave the CPU nothing, so a CPU part is never called.
        let expected = cpu.weighted_sums(&small, &activation, &scales).unwrap();
        let sums = engine
            .shared(Some((&mut failing, 4)), &[&small], &activation, &scales)
            .unwrap();
        assert_sums(&sums[0], &expected, "small matrix without CPU rows");
    }
}
