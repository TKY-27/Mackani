// Shared writer retains mackani-ref's Rust 1.85 syntax.
#[allow(clippy::manual_is_multiple_of)]
#[path = "../../mackani-ref/tests/support/mod.rs"]
mod support;

use mackani_engine::{Backend, Engine, Kernel, KvMode, Model};
use mackani_format::Container;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new(bytes: &[u8]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "mackani-model-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        fs::write(path.join("weights.yskn"), bytes).unwrap();
        fs::write(
            path.join("yasakani.json"),
            r#"{"files":{"weights":"weights.yskn"},"max_ctx":257}"#,
        )
        .unwrap();
        Self(path)
    }
    fn model(&self, kernel: Kernel, threads: usize, rows: usize) -> Model {
        // SAFETY: fixture contents are immutable until every model is dropped before Fixture.
        unsafe { Model::with_engine(&self.0, Engine::new(kernel, threads).unwrap(), rows) }.unwrap()
    }
    fn backend_model(&self, backend: Backend, threads: usize, rows: usize) -> Model {
        let engine = Engine::new(Kernel::fastest(), threads).unwrap();
        // SAFETY: fixture contents are immutable until every model is dropped before Fixture.
        unsafe { Model::with_backend(&self.0, engine, backend, rows) }.unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn same(actual: &[Vec<f32>], expected: &[Vec<f32>], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (row, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.len(), b.len(), "{context} row={row}");
        for (v, (a, b)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{context} row={row} vocabulary={v}"
            );
        }
    }
}
fn logit_bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|row| row.iter().map(|v| v.to_bits()).collect())
        .collect()
}
fn digest(rows: &[Vec<f32>]) -> String {
    let mut hash = Sha256::new();
    for row in rows {
        for value in row {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}
fn compare(bytes: Vec<u8>, count: usize, baseline: &str) {
    compare_mode(bytes, count, baseline, KvMode::Int8);
}
fn compare_mode(bytes: Vec<u8>, count: usize, baseline: &str, mode: KvMode) {
    let fixture = Fixture::new(&bytes);
    if mode != KvMode::Int8 {
        fs::write(
            fixture.0.join("kvq_u5.bin"),
            support::kv_profile::mixed_profile(),
        )
        .unwrap();
    }
    let mut reference = mackani_ref::Model::load_folder_with(&fixture.0, mode).unwrap();
    reference.prefill(&[1, 2, 3]).unwrap();
    let initial = reference.row_logits().to_vec();
    let tokens: Vec<_> = (0..count).map(|i| ((i * 7 + 11) % 65) as u32).collect();
    reference.prefill(&tokens).unwrap();
    let expected = reference.row_logits().to_vec();
    let int8_engine_logits = if mode != KvMode::Int8 {
        let mut int8_reference =
            mackani_ref::Model::load_folder_with(&fixture.0, KvMode::Int8).unwrap();
        int8_reference.prefill(&[1, 2, 3]).unwrap();
        int8_reference.prefill(&tokens).unwrap();
        assert_ne!(
            logit_bits(&expected),
            logit_bits(int8_reference.row_logits()),
            "reference ignored {mode:?}"
        );
        let mut int8_engine = fixture.model(Kernel::Portable, 1, 5);
        int8_engine.prefill(&[1, 2, 3]).unwrap();
        int8_engine.prefill(&tokens).unwrap();
        Some(logit_bits(int8_engine.row_logits()))
    } else {
        None
    };
    assert_eq!(
        digest(&expected),
        baseline,
        "frozen reference logit bits changed"
    );
    reference.prefill(&[17, 21]).unwrap();
    let continuation = reference.row_logits().to_vec();
    reference.reset().unwrap();
    reference.prefill(&[5, 13, 7, 1]).unwrap();
    let after_reset = reference.row_logits().to_vec();
    let check = |context: &str, mut model: Model| {
        assert_eq!(model.context_capacity(), 320);
        assert_eq!(model.kv_mode(), mode);
        model.prefill(&[1, 2, 3]).unwrap();
        same(model.row_logits(), &initial, context);
        model.prefill(&tokens).unwrap();
        same(model.row_logits(), &expected, context);
        if let Some(int8_logits) = &int8_engine_logits {
            assert_ne!(
                &logit_bits(model.row_logits()),
                int8_logits,
                "engine ignored {mode:?}: {context}"
            );
        }
        assert_eq!(model.position(), 3 + count);
        assert_eq!(model.logits().unwrap().len(), expected[0].len());
        model.reset().unwrap();
        assert_eq!(model.position(), 0);
        assert!(model.logits().is_none());
        model.prefill(&[5, 13, 7, 1]).unwrap();
        same(model.row_logits(), &after_reset, context);
        model.reset().unwrap();
        model.prefill(&[1, 2, 3]).unwrap();
        let mut actual = Vec::new();
        let mut first = 0;
        for chunk in [1, 3, 7, 2, 19, 5, 11, 23, 100] {
            if first == tokens.len() {
                break;
            }
            let end = (first + chunk).min(tokens.len());
            model.prefill(&tokens[first..end]).unwrap();
            actual.extend_from_slice(model.row_logits());
            first = end;
        }
        same(&actual, &expected, context);
        let last = model.row_logits().to_vec();
        model.prefill(&[]).unwrap();
        same(model.row_logits(), &last, "empty prefill");
        // Continuing the sequence proves the persistent cache/recurrence too.
        model.prefill(&[17, 21]).unwrap();
        let continued = model.row_logits().to_vec();
        same(&continued, &continuation, context);
    };
    let mut backends = vec![Backend::Cpu];
    if metal_device() {
        backends.push(Backend::Metal);
    }
    for backend in backends {
        let kernels = if mode == KvMode::Int8 {
            Kernel::available_variants()
        } else {
            vec![Kernel::fastest()]
        };
        let widths: &[usize] = if mode == KvMode::Int8 {
            &[1, 2, 5, 16, 17, 40]
        } else {
            &[1, 5, 16, 17]
        };
        for kernel in kernels {
            // Metal ignores the projection kernel; only run that backend once per width/pool.
            if backend == Backend::Metal && kernel != Kernel::fastest() {
                continue;
            }
            for threads in [1, 4] {
                for &rows in widths {
                    // Pass grouping is above Compute::weighted_sums/map_range. The
                    // linear suite keeps the full kernel × pool × M product; here
                    // cover every kernel/pool at width 17 and every width at x4.
                    if rows != 17 && (kernel != Kernel::fastest() || threads != 4) {
                        continue;
                    }
                    let context = format!(
                        "{mode:?} {backend:?} {} threads={threads} rows_per_pass={rows}",
                        kernel.name()
                    );
                    let model = if mode == KvMode::Int8 && backend == Backend::Cpu {
                        fixture.model(kernel, threads, rows)
                    } else {
                        // SAFETY: fixture contents stay immutable until this model is dropped.
                        unsafe {
                            Model::with_options(
                                &fixture.0,
                                Engine::new(kernel, threads).unwrap(),
                                backend,
                                rows,
                                mode,
                            )
                        }
                        .unwrap()
                    };
                    assert_eq!(model.backend(), backend);
                    check(&context, model);
                    println!("ran: {context}");
                }
            }
        }
    }
}
/// A missing device is reported on stdout and is the only way to skip Metal cases.
fn metal_device() -> bool {
    #[cfg(target_os = "macos")]
    if mackani_engine::MetalEngine::device_exists() {
        return true;
    }
    println!("skipped: no Metal device");
    false
}
#[test]
fn recurrent_model_is_identical_for_all_kernels_threads_pass_widths_and_chunks() {
    compare(
        support::synthetic_model(19, &["linear_attention", "linear_attention"]),
        9,
        "efd528122d8df0a2b6d0162f1acd4ccbefb699c6d4074f0981d6ac290183cc40",
    );
}
#[test]
fn mixed_model_crosses_attention_tiles_from_position_three_without_bit_changes() {
    compare(
        support::synthetic_model(
            19,
            &[
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention",
            ],
        ),
        73,
        "3cf62a9fb84c5f8d5ff3cdd434f061014b0baafc773fa38166c3091a0142a8ba",
    );
}
#[test]
fn int8_profile_mixed_model_matches_reference_across_passes_chunks_reset_and_continuation() {
    compare_mode(
        support::synthetic_model(
            107,
            &["linear_attention", "full_attention", "full_attention"],
        ),
        73,
        "bd759b4b86ab911371bf726f34e0234c321c9d6292e31133e7f0d2f5c6d53b8c",
        KvMode::Int8Profile,
    );
}
#[test]
fn kv5_mixed_model_matches_reference_across_passes_chunks_reset_and_continuation() {
    compare_mode(
        support::synthetic_model(
            107,
            &["linear_attention", "full_attention", "full_attention"],
        ),
        73,
        "af825f48eb7eb1a8337f52b31b7018ab568907e996724d81139c1594b0e5fc26",
        KvMode::Kv5,
    );
}
#[test]
fn other_hidden_ffn_widths_and_padded_vocabulary_are_identical() {
    compare(
        support::synthetic_model_with_dimensions(21, &["linear_attention"], 1024, 1024, 65),
        9,
        "24b68b1e25f9e0fc3c75f82816989d90bcfda5f6621ba222fac1296078a7d553",
    );
}
#[test]
fn dense_embeddings_are_identical() {
    compare(
        support::synthetic_model_with_dense_embeddings(21, &["linear_attention"]),
        9,
        "2621a671d18e0c7e4bfc3f8525aabb2c3b92949a0c7e6ef34b7e7b634dccf5fb",
    );
}
#[test]
fn invalid_requests_preserve_logits_position_and_future_sequence_state() {
    let bytes = support::synthetic_model(17, &["linear_attention", "full_attention"]);
    let fixture = Fixture::new(&bytes);
    fs::write(
        fixture.0.join("yasakani.json"),
        r#"{"files":{"weights":"weights.yskn"},"max_ctx":1}"#,
    )
    .unwrap();
    let mut reference = mackani_ref::Model::load_folder(&fixture.0).unwrap();
    let mut model = fixture.model(Kernel::fastest(), 4, 5);
    assert_eq!(model.context_capacity(), 64);
    reference.prefill(&[1, 2, 3]).unwrap();
    model.prefill(&[1, 2, 3]).unwrap();
    let before = model.row_logits().to_vec();
    for tokens in [vec![1, 2, 128], vec![1; 62]] {
        assert!(matches!(
            model.prefill(&tokens),
            Err(mackani_engine::Error::InvalidInput(_))
        ));
        assert!(matches!(
            reference.prefill(&tokens),
            Err(mackani_ref::Error::InvalidInput(_))
        ));
        assert_eq!(model.position(), 3);
        same(model.row_logits(), &before, "failed pass");
        same(reference.row_logits(), &before, "reference failed pass");
    }
    model.prefill(&[9; 61]).unwrap();
    reference.prefill(&[9; 61]).unwrap();
    same(
        model.row_logits(),
        reference.row_logits(),
        "continuation after rejected pass",
    );
    assert!(model.prefill(&[1]).is_err());
    assert_eq!(model.position(), 64);
    // SAFETY: fixture contents remain stable until this rejected construction completes.
    assert!(
        unsafe {
            Model::with_engine(
                fixture.0.join("weights.yskn"),
                Engine::new(Kernel::Portable, 1).unwrap(),
                0,
            )
        }
        .is_err()
    );
}
#[test]
fn failure_after_a_completed_pass_preserves_logits_position_and_state() {
    // Token 100's dense embedding is finite but its squares overflow, so the
    // first layer norm fails in whichever pass contains it.
    let mut bytes = support::synthetic_model_with_dense_embeddings(21, &["linear_attention"]);
    let embedding = Container::from_bytes(bytes.clone())
        .unwrap()
        .header()
        .vectors
        .iter()
        .find(|v| v.name == "embed_tokens")
        .unwrap()
        .clone();
    let huge = mackani_format::f32_to_bf16(1.0e38).to_le_bytes();
    let row = embedding.offset as usize + 100 * 512 * 2;
    for pair in bytes[row..row + 1024].as_chunks_mut::<2>().0 {
        *pair = huge;
    }
    let fixture = Fixture::new(&bytes);
    let mut reference = mackani_ref::Model::load_folder(&fixture.0).unwrap();
    reference.prefill(&[1, 2, 3]).unwrap();
    assert!(reference.prefill(&[4, 5, 6, 7, 8, 100, 9]).is_err());
    reference.prefill(&[4, 5, 6, 7, 8]).unwrap();
    let continuation = reference.row_logits().to_vec();
    let mut backends = vec![Backend::Cpu];
    if metal_device() {
        backends.push(Backend::Metal);
    }
    for backend in backends {
        let context = backend.name();
        let mut model = fixture.backend_model(backend, 4, 2);
        model.prefill(&[1, 2, 3]).unwrap();
        let before = model.row_logits().to_vec();
        let timings = model.pass_timings().len();
        // Passes [4, 5] and [6, 7] complete before [8, 100] fails.
        assert!(matches!(
            model.prefill(&[4, 5, 6, 7, 8, 100, 9]),
            Err(mackani_engine::Error::NonFinite(_))
        ));
        assert_eq!(model.position(), 3, "{context}");
        assert_eq!(model.pass_timings().len(), timings, "{context}");
        same(model.row_logits(), &before, context);
        // Unchanged recurrent state: continuing equals the reference continuing.
        model.prefill(&[4, 5, 6, 7, 8]).unwrap();
        same(model.row_logits(), &continuation, context);
    }
}
#[test]
fn persistent_range_pool_preserves_order_borrows_and_recovers_after_errors() {
    let mut engine = Engine::new(Kernel::Portable, 4).unwrap();
    let input: Vec<_> = (0..101).map(|i| i * 3).collect();
    assert_eq!(
        engine.map_range(3..99, |i| Ok(input[i] + 1)).unwrap(),
        input[3..99].iter().map(|x| x + 1).collect::<Vec<_>>()
    );
    assert!(
        engine
            .map_range(0..101, |i| if i == 8 {
                Err(mackani_engine::Error::InvalidInput("expected".into()))
            } else {
                Ok(input[i])
            })
            .is_err()
    );
    assert!(
        engine
            .map_range(0..101, |i| -> mackani_engine::Result<usize> {
                if i == 8 {
                    panic!("expected range closure panic");
                }
                Ok(i)
            })
            .is_err()
    );
    // Several failures, including a panic after the first: the lowest index wins.
    for _ in 0..20 {
        let error = engine
            .map_range(0..101, |i| match i {
                97 => panic!("expected range closure panic"),
                _ if i % 10 == 3 => Err(mackani_engine::Error::InvalidInput(format!("index {i}"))),
                _ => Ok(i),
            })
            .unwrap_err();
        assert_eq!(error.to_string(), "invalid input: index 3");
    }
    assert_eq!(engine.map_range(0..101, |i| Ok(input[i])).unwrap(), input);
    assert!(engine.map_range(0..0, |_| Ok(())).unwrap().is_empty());
}

#[test]
fn single_item_range_runs_inline_and_preserves_errors() {
    use mackani_engine::{Error, Result};
    use std::{panic, thread};

    struct PanicOnDrop;
    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!("expected panic from payload destructor");
        }
    }

    let mut engine = Engine::new(Kernel::Portable, 4).unwrap();
    let caller = thread::current().id();
    let input = [7, 23];
    let map = |i| {
        assert_eq!(thread::current().id(), caller);
        Ok(&input[i])
    };
    assert_eq!(engine.map_range(1..2, map).unwrap(), vec![&23]);
    let error = engine
        .map_range(1..2, |i| -> Result<()> {
            Err(Error::InvalidInput(format!("index {i}")))
        })
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput(message) if message == "index 1"));
    for fail in [
        || -> Result<()> { panic!("expected range closure panic") },
        || -> Result<()> { panic::panic_any(PanicOnDrop) },
    ] {
        let error = engine.map_range(1..2, |_| fail()).unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput(message) if message == "range worker panicked")
        );
    }
    assert_eq!(engine.map_range(1..2, map).unwrap(), vec![&23]);
}

#[test]
#[ignore = "CPU benchmark; run in release mode with --nocapture"]
fn bench_single_item_pool() {
    use std::{hint::black_box, time::Instant};

    let mut engine = Engine::new(Kernel::Portable, 4).unwrap();
    let mut map = || black_box(engine.map_range(black_box(0..1), |i| Ok(i + 1)).unwrap());
    for _ in 0..1_000 {
        map();
    }
    let iterations = 20_000;
    let started = Instant::now();
    for _ in 0..iterations {
        map();
    }
    println!(
        "map_ns_per_call={:.2}",
        started.elapsed().as_secs_f64() * 1e9 / f64::from(iterations)
    );

    let fixture = Fixture::new(&support::synthetic_model(
        19,
        &[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
    ));
    let mut reference = mackani_ref::Model::load_folder(&fixture.0).unwrap();
    let mut model = fixture.model(Kernel::Portable, 4, 8);
    let prefix: Vec<_> = (0..8).map(|i| i * 7 + 11).collect();
    let suffix: Vec<_> = (8..16).map(|i| (i * 7 + 11) % 128).collect();
    model.prefill(&prefix).unwrap();
    reference.prefill(&prefix).unwrap();
    let checkpoint = model.snapshot().unwrap();
    let ref_checkpoint = reference.snapshot().unwrap();
    for rows in [1, 8] {
        let tokens = &suffix[..rows];
        reference.restore(&ref_checkpoint).unwrap();
        reference.prefill(tokens).unwrap();
        let mut elapsed = Vec::new();
        for _ in 0..86 {
            model.restore(&checkpoint).unwrap();
            let started = Instant::now();
            model.prefill(tokens).unwrap();
            elapsed.push(started.elapsed());
            same(model.row_logits(), reference.row_logits(), "logits");
            assert_eq!(model.state_digest(), reference.state_digest());
        }
        let samples = &mut elapsed[5..]; // Discard warm-up passes.
        samples.sort_unstable();
        println!(
            "rows={rows} pass_median_us={:.2} digest={}",
            samples[samples.len() / 2].as_secs_f64() * 1e6,
            digest(model.row_logits()),
        );
    }
}

#[test]
fn default_and_explicit_constructors_keep_int8() {
    let fixture = Fixture::new(&support::synthetic_model(
        107,
        &["linear_attention", "full_attention"],
    ));
    // SAFETY: the fixture owns immutable weights until the loaded model is dropped.
    let loaded = unsafe { Model::load(&fixture.0) }.unwrap();
    assert_eq!(loaded.kv_mode(), KvMode::Int8);
    let engine = || Engine::new(Kernel::Portable, 1).unwrap();
    // SAFETY: the fixture keeps its weights immutable until both models are dropped.
    let cpu = unsafe { Model::with_engine(&fixture.0, engine(), 5) }.unwrap();
    // SAFETY: the same fixture owns these immutable weights.
    let backend = unsafe { Model::with_backend(&fixture.0, engine(), Backend::Cpu, 5) }.unwrap();
    assert_eq!(cpu.kv_mode(), KvMode::Int8);
    assert_eq!(backend.kv_mode(), KvMode::Int8);
}

#[test]
fn explicit_metal_share_preserves_logits() {
    if !metal_device() {
        return;
    }
    let fixture = Fixture::new(&support::synthetic_model(
        107,
        &["linear_attention", "full_attention"],
    ));
    let load = || {
        // SAFETY: the fixture keeps the weights immutable until every model is dropped.
        unsafe {
            Model::with_backend(
                &fixture.0,
                Engine::new(Kernel::fastest(), 2).unwrap(),
                Backend::Metal,
                5,
            )
        }
        .unwrap()
    };
    let mut baseline = load();
    let tokens = [3, 17, 42, 5, 99, 0, 7];
    baseline.prefill(&tokens).unwrap();
    for cpu_share in [
        mackani_engine::CpuShare::Auto,
        mackani_engine::CpuShare::Off,
        mackani_engine::CpuShare::Fixed(0.5),
    ] {
        let mut model = load();
        model.set_metal_options(mackani_engine::MetalOptions { cpu_share });
        assert_eq!(
            model.metal_options(),
            Some(mackani_engine::MetalOptions { cpu_share })
        );
        model.prefill(&tokens).unwrap();
        same(
            model.row_logits(),
            baseline.row_logits(),
            "explicit Metal share",
        );
    }
}

/// Per-model CPU stages preserve bits through provisional passes and partial commits.
#[test]
fn explicit_cpu_stage_choice_preserves_bits_and_rejects_pending_changes() {
    let fixture = Fixture::new(&support::synthetic_model(
        107,
        &["linear_attention", "full_attention"],
    ));
    let tokens = [3, 17, 42, 5, 99, 0, 7, 64, 11];
    let mut reference = mackani_ref::Model::load_folder(&fixture.0).unwrap();
    reference.prefill(&tokens).unwrap();
    for stages in [
        mackani_engine::CpuStages::Reference,
        mackani_engine::CpuStages::Auto,
    ] {
        let mut model = fixture.model(Kernel::fastest(), 3, 4);
        model.set_cpu_stages(stages).unwrap();
        assert_eq!(model.cpu_stages(), stages.fast(), "{stages:?}");
        model.prefill(&tokens[..5]).unwrap();
        model.run_provisional(&tokens[5..]).unwrap();
        assert!(model.set_cpu_stages(stages).is_err(), "{stages:?}");
        model.commit(2).unwrap();
        model.prefill(&tokens[7..]).unwrap();
        same(
            &model.row_logits()[model.row_logits().len() - 1..],
            &reference.row_logits()[reference.row_logits().len() - 1..],
            &format!("{stages:?}"),
        );
        assert_eq!(model.state_digest(), reference.state_digest(), "{stages:?}");
    }
}
