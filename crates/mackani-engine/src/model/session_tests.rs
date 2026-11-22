use super::execution::{attention_pass, recurrence};
use super::*;
use crate::Kernel;
use std::sync::atomic::{AtomicU64, Ordering};

#[test]
fn dense_gate_errors_check_all_partials_then_results_then_next_projection() {
    use mackani_ref::gated_deltanet::{GatedDeltaNetDimensions, GatedDeltaNetWeights};
    for threads in [1, 4] {
        let mut engine = Compute::new(
            Engine::new(Kernel::Portable, threads).unwrap(),
            Backend::Cpu,
        )
        .unwrap();
        for rows in [1, 2] {
            for (partial_in_a, expected) in [
                (true, "DeltaNet dense gate partial"),
                (false, "DeltaNet dense gates"),
            ] {
                let mut a = vec![0.0; 1024];
                let mut b = vec![0.0; 1024];
                a[0] = f32::from_bits(0x7f00_0000);
                if partial_in_a {
                    a[513] = f32::from_bits(0x7f00_0000);
                } else {
                    b[513] = f32::from_bits(0x7f00_0000);
                }
                let layer = GatedDeltaNet::new(
                    GatedDeltaNetDimensions {
                        hidden_width: 512,
                        query_key_heads: 1,
                        value_heads: 2,
                        head_dim: 128,
                        convolution_width: 4,
                    },
                    GatedDeltaNetWeights {
                        projection_a: a,
                        projection_b: b,
                        convolution_weight: vec![0.0; 2048],
                        log_decay_weight: vec![0.0; 2],
                        time_bias: vec![0.0; 2],
                        normalization_weight: vec![1.0; 128],
                    },
                    1e-6,
                )
                .unwrap();
                let input: Vec<_> = (0..rows)
                    .map(|_| {
                        let mut hidden = vec![0.0; 512];
                        hidden[..2].copy_from_slice(&[1.0, 4.0]);
                        NormalizedInput {
                            prepared: quantize_activation(&hidden, &[1; 512]).unwrap(),
                            inverse_rms: 4.0,
                            weighted_hidden_bf16: hidden,
                        }
                    })
                    .collect();
                let result = recurrence(
                    &mut engine,
                    &layer,
                    &mut layer.new_state().unwrap(),
                    None,
                    &vec![vec![0.0; 512]; rows],
                    &vec![vec![0.0; 256]; rows],
                    &input,
                    None,
                    &mut PassTiming::default(),
                );
                assert!(
                    matches!(result, Err(Error::NonFinite(context)) if context == expected),
                    "threads={threads} rows={rows} partial_in_a={partial_in_a}"
                );
            }
        }
    }
}

#[test]
fn dense_gate_errors_follow_scalar_row_order_with_different_failures() {
    use mackani_ref::gated_deltanet::{GatedDeltaNetDimensions, GatedDeltaNetWeights};
    let mut a = vec![0.0; 1024];
    a[0] = f32::from_bits(0x7f00_0000);
    a[513] = f32::from_bits(0x7f00_0000);
    let layer = GatedDeltaNet::new(
        GatedDeltaNetDimensions {
            hidden_width: 512,
            query_key_heads: 1,
            value_heads: 2,
            head_dim: 128,
            convolution_width: 4,
        },
        GatedDeltaNetWeights {
            projection_a: a,
            projection_b: vec![0.0; 1024],
            convolution_weight: vec![0.0; 2048],
            log_decay_weight: vec![0.0; 2],
            time_bias: vec![0.0; 2],
            normalization_weight: vec![1.0; 128],
        },
        1e-6,
    )
    .unwrap();
    let mut input: Vec<_> = [[1.0, 0.0], [0.0, 4.0]]
        .into_iter()
        .map(|first_two| {
            let mut hidden = vec![0.0; 512];
            hidden[..2].copy_from_slice(&first_two);
            NormalizedInput {
                prepared: quantize_activation(&hidden, &[1; 512]).unwrap(),
                inverse_rms: 4.0,
                weighted_hidden_bf16: hidden,
            }
        })
        .collect();
    let qkv = vec![vec![0.0; 512]; 2];
    let z = vec![vec![0.0; 256]; 2];
    // Prove each distinct error with the scalar reference before comparing passes.
    for (row, expected) in input
        .iter()
        .zip(["DeltaNet dense gates", "DeltaNet dense gate partial"])
    {
        let error = layer
            .step(
                &mut layer.new_state().unwrap(),
                &qkv[0],
                &z[0],
                &row.weighted_hidden_bf16,
                row.inverse_rms,
            )
            .unwrap_err();
        assert!(matches!(error, Error::NonFinite(context) if context == expected));
    }
    for reversed in [false, true] {
        if reversed {
            input.reverse();
        }
        for threads in [1, 4] {
            let mut engine = Compute::new(
                Engine::new(Kernel::Portable, threads).unwrap(),
                Backend::Cpu,
            )
            .unwrap();
            let mut reference_state = layer.new_state().unwrap();
            let expected = input
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    layer.step(
                        &mut reference_state,
                        &qkv[i],
                        &z[i],
                        &row.weighted_hidden_bf16,
                        row.inverse_rms,
                    )
                })
                .collect::<Result<Vec<_>>>()
                .unwrap_err();
            let actual = recurrence(
                &mut engine,
                &layer,
                &mut layer.new_state().unwrap(),
                None,
                &qkv,
                &z,
                &input,
                None,
                &mut PassTiming::default(),
            )
            .unwrap_err();
            assert_eq!(
                actual.to_string(),
                expected.to_string(),
                "threads={threads} reversed={reversed}"
            );
        }
    }
}

#[test]
fn attention_pass_errors_follow_scalar_row_and_stage_order() {
    use mackani_ref::attention::{Attention, AttentionDimensions};
    let layer = Attention::new(
        AttentionDimensions {
            query_heads: 2,
            kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
        },
        vec![0.0; 256],
        vec![0.0; 256],
        1e-6,
    )
    .unwrap();
    for threads in [1, 4] {
        let mut engine = Compute::new(
            Engine::new(Kernel::Portable, threads).unwrap(),
            Backend::Cpu,
        )
        .unwrap();
        // Same-row query before value, earlier query before later value,
        // earlier value before later query, and final gate check after cache encoding.
        for (rows, query_row, value_row, bad_gate, expected_context) in [
            (1, 0, 0, false, "attention head sum of squares"),
            (2, 0, 1, false, "attention head sum of squares"),
            (2, 1, 0, false, "attention cache input"),
            (1, 0, 0, true, "attention cache input"),
        ] {
            let mut q = vec![vec![0.25; 1024]; rows];
            let k = vec![vec![0.25; 512]; rows];
            let mut v = vec![vec![0.25; 512]; rows];
            if bad_gate {
                q[query_row][256..512].fill(f32::MAX);
            } else {
                q[query_row][..256].fill(1e20);
            }
            v[value_row][..256].fill(2e38);
            let expected = (0..rows)
                .map(|i| layer.prepare_row(i, &q[i], &k[i], &v[i]))
                .collect::<Result<Vec<_>>>()
                .err()
                .unwrap();
            assert!(matches!(expected, Error::NonFinite(context) if context == expected_context));
            let mut state = layer.new_state().unwrap();
            let before = state.clone();
            let actual = attention_pass(
                &mut engine,
                &layer,
                &mut state,
                [&q, &k, &v],
                &mut PassTiming::default(),
            )
            .unwrap_err();
            assert_eq!(
                actual.to_string(),
                expected.to_string(),
                "threads={threads} rows={rows} query={query_row} value={value_row} gate={bad_gate}"
            );
            assert_eq!(state, before, "failed preparation appended cache rows");
            // The same pool/state can still complete the actual attention pass.
            let q = vec![vec![0.25; 1024]; rows];
            let v = vec![vec![0.25; 512]; rows];
            let actual = attention_pass(
                &mut engine,
                &layer,
                &mut state,
                [&q, &k, &v],
                &mut PassTiming::default(),
            )
            .unwrap();
            let mut reference = layer.new_state().unwrap();
            for i in 0..rows {
                let expected = layer.step(&mut reference, i, &q[i], &k[i], &v[i]).unwrap();
                assert_eq!(
                    actual[i].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
            assert_eq!(state, reference);
        }
    }
}

fn fixture_path() -> std::path::PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "mackani-rollback-{}-{}.yskn",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let bytes =
        super::session_test_support::synthetic_model(187, &["full_attention", "linear_attention"]);
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn detailed_timing_is_opt_in_and_preserves_coarse_times_and_bits() {
    let path = fixture_path();
    for rows in [1, 2] {
        // SAFETY: the synthetic fixture stays immutable until the model is dropped.
        let mut model =
            unsafe { Model::with_engine(&path, Engine::new(Kernel::Portable, 2).unwrap(), rows) }
                .unwrap();
        assert!(!model.detailed_timing);
        let mut expected = None;
        for detailed in [false, true, false] {
            model.reset().unwrap();
            model.set_detailed_timing(detailed);
            model.prefill(&[1, 2]).unwrap();
            let actual = (
                model.state_digest(),
                model
                    .row_logits()
                    .iter()
                    .map(|row| row.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            );
            if let Some(expected) = &expected {
                assert_eq!(&actual, expected);
            } else {
                expected = Some(actual);
            }
            assert!(!model.pass_timings().is_empty());
            for timing in model.pass_timings() {
                for coarse in [
                    timing.total,
                    timing.kernel,
                    timing.recurrence,
                    timing.attention,
                    timing.kv_encode,
                    timing.rest(),
                ] {
                    assert!(!coarse.is_zero());
                }
                for subset in [
                    timing.recurrence_prepare,
                    timing.dense_gate_work,
                    timing.normalization,
                    timing.activation_encode,
                    timing.projection_restore,
                    timing.mlp_gate,
                    timing.residual,
                ] {
                    assert_eq!(!subset.is_zero(), detailed);
                }
            }
        }
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn failed_later_layer_truncates_already_appended_attention() {
    let path = fixture_path();
    let mut reference = mackani_ref::Model::load(&path).unwrap();
    // SAFETY: this test never modifies the backing file while the mapping is alive.
    let mut engine =
        unsafe { Model::with_engine(&path, Engine::new(Kernel::Portable, 2).unwrap(), 4) }.unwrap();
    engine
        .set_capture(CaptureOptions {
            layers: vec![0, 1],
            max_rows: 3,
        })
        .unwrap();
    engine.prefill(&[1, 2]).unwrap();
    reference.prefill(&[1, 2]).unwrap();
    let digest = engine.state_digest();
    let logits = engine.row_logits().to_vec();
    let captures = engine.captured_rows().to_vec();
    // Owned norm data can be poisoned without mutating any mapped weight bytes.
    // Layer zero has appended both attention rows before layer one reads this.
    let saved = engine.weights.layers[1].input_norm[0];
    engine.weights.layers[1].input_norm[0] = f32::NAN;
    assert!(matches!(
        engine.run_provisional(&[3, 4]),
        Err(Error::NonFinite(_))
    ));
    engine.weights.layers[1].input_norm[0] = saved;
    assert_eq!(engine.state_digest(), digest);
    assert_eq!(engine.row_logits(), logits);
    assert!(!engine.has_provisional_pass());
    assert_eq!(engine.captured_rows(), captures);
    assert!(engine.provisional_captures().is_empty());
    // The digest only includes the committed prefix: a successful probe is
    // essential to detect an untruncated, otherwise invisible cache suffix.
    engine.run_provisional(&[5, 6]).unwrap();
    reference.run_provisional(&[5, 6]).unwrap();
    assert_eq!(engine.row_logits(), reference.row_logits());
    engine.commit(2).unwrap();
    reference.commit(2).unwrap();
    assert_eq!(engine.state_digest(), reference.state_digest());
    drop(engine);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn failed_partial_replay_retains_pending_pass_and_full_retry_replays_correctly() {
    use mackani_ref::gated_deltanet::{GatedDeltaNetDimensions, GatedDeltaNetWeights};
    let path = fixture_path();
    let mut reference = mackani_ref::Model::load(&path).unwrap();
    // SAFETY: backing file remains immutable until model drop.
    let mut engine =
        unsafe { Model::with_engine(&path, Engine::new(Kernel::Portable, 2).unwrap(), 4) }.unwrap();
    engine.prefill(&[1, 2]).unwrap();
    reference.prefill(&[1, 2]).unwrap();
    engine.run_provisional(&[3, 4, 5]).unwrap();
    reference.run_provisional(&[3, 4, 5]).unwrap();
    let digest = engine.state_digest();
    let logits = engine.row_logits().to_vec();
    // An incompatible cached row forces the recurrence replay error path.
    // The actual row is restored before retrying the still-pending pass.
    let other = GatedDeltaNet::new(
        GatedDeltaNetDimensions {
            hidden_width: 512,
            query_key_heads: 1,
            value_heads: 1,
            head_dim: 128,
            convolution_width: 4,
        },
        GatedDeltaNetWeights {
            projection_a: vec![0.0; 512],
            projection_b: vec![0.0; 512],
            convolution_weight: vec![0.0; 384 * 4],
            log_decay_weight: vec![0.0],
            time_bias: vec![0.0],
            normalization_weight: vec![1.0; 128],
        },
        1e-6,
    )
    .unwrap();
    let saved = std::mem::replace(&mut engine.replay[1][0], other.new_row().unwrap());
    assert!(engine.commit(1).is_err());
    assert!(engine.has_provisional_pass());
    assert_eq!(engine.state_digest(), digest);
    assert_eq!(engine.row_logits(), logits);
    assert!(engine.snapshot().is_err());
    engine.replay[1][0] = saved;
    engine.commit(3).unwrap();
    reference.commit(3).unwrap();
    assert_eq!(engine.state_digest(), reference.state_digest());
    assert_eq!(engine.row_logits(), reference.row_logits());
    drop(engine);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn cpu_stage_attention_pass_matches_scalar_steps_at_published_shape() {
    use mackani_ref::attention::{Attention, AttentionDimensions};
    let mut seed = 0x2545_f491_u32;
    let mut values = |len: usize| {
        (0..len)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (seed as i32 as f32) / (i32::MAX as f32) * 3.0
            })
            .collect::<Vec<_>>()
    };
    let layer = Attention::new(
        AttentionDimensions {
            query_heads: 24,
            kv_heads: 4,
            head_dim: 256,
            rotary_dim: 64,
        },
        values(256),
        values(256),
        1e-6,
    )
    .unwrap();
    for threads in [1, 4] {
        let mut engine = Compute::new(
            Engine::new(Kernel::Portable, threads).unwrap(),
            Backend::Cpu,
        )
        .unwrap();
        engine.cpu_stages = true;
        let mut state = layer.new_state().unwrap();
        let mut reference = layer.new_state().unwrap();
        let mut position = 0;
        for rows in [1, 8, 16, 1] {
            let q = (0..rows).map(|_| values(12_288)).collect::<Vec<_>>();
            let k = (0..rows).map(|_| values(1024)).collect::<Vec<_>>();
            let v = (0..rows).map(|_| values(1024)).collect::<Vec<_>>();
            let mut timing = PassTiming {
                position,
                ..PassTiming::default()
            };
            let actual =
                attention_pass(&mut engine, &layer, &mut state, [&q, &k, &v], &mut timing).unwrap();
            for i in 0..rows {
                let expected = layer
                    .step(&mut reference, position + i, &q[i], &k[i], &v[i])
                    .unwrap();
                assert_eq!(
                    actual[i].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "threads={threads} rows={rows} row={i}"
                );
            }
            assert_eq!(state, reference);
            position += rows;
        }
    }
}

/// Times attention-row preparation (`PassTiming::kv_encode`) at the published
/// model's attention shape, with the CPU-stage scheduling Apple Silicon uses.
#[test]
#[ignore = "timing benchmark: cargo test --release -p mackani-engine attention_preparation_timing -- --ignored --nocapture"]
fn attention_preparation_timing() {
    use mackani_ref::attention::{Attention, AttentionDimensions};
    const LAYERS: f64 = 16.0;
    const CONTEXT: usize = 1024;
    const SAMPLES: usize = 400;
    let mut seed = 0x9e37_79b9_u32;
    let mut values = |len: usize| {
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1 << 23) as f32 - 1.0
            })
            .collect::<Vec<_>>()
    };
    let layer = Attention::new(
        AttentionDimensions {
            query_heads: 24,
            kv_heads: 4,
            head_dim: 256,
            rotary_dim: 64,
        },
        values(256),
        values(256),
        1e-6,
    )
    .unwrap();
    let threads = std::env::var("MACKANI_BENCH_THREADS")
        .ok()
        .and_then(|threads| threads.parse().ok())
        .unwrap_or_else(|| crate::Machine::detect().default_threads());
    let mut engine = Compute::new(
        Engine::new(Kernel::fastest(), threads).unwrap(),
        Backend::Cpu,
    )
    .unwrap();
    engine.cpu_stages = true;
    let mut prefix = layer.new_state().unwrap();
    for position in 0..CONTEXT {
        let prepared = layer
            .prepare_row(position, &values(12_288), &values(1024), &values(1024))
            .unwrap();
        layer
            .append_prepared(&mut prefix, position, &prepared)
            .unwrap();
    }
    println!("threads={threads} context={CONTEXT} samples={SAMPLES}");
    println!("| M | median µs/pass | min µs/pass | ms/token (16 layers) | output digest |");
    for rows in [1, 8, 16] {
        let q = (0..rows).map(|_| values(12_288)).collect::<Vec<_>>();
        let k = (0..rows).map(|_| values(1024)).collect::<Vec<_>>();
        let v = (0..rows).map(|_| values(1024)).collect::<Vec<_>>();
        let mut digest = 0;
        let mut samples = (0..SAMPLES + SAMPLES / 4)
            .map(|_| {
                let mut state = prefix.clone();
                let mut timing = PassTiming {
                    position: CONTEXT,
                    ..PassTiming::default()
                };
                let output =
                    attention_pass(&mut engine, &layer, &mut state, [&q, &k, &v], &mut timing)
                        .unwrap();
                // FNV-1a over the output bits, folded into the cache digest.
                digest = output
                    .iter()
                    .flatten()
                    .fold(0xcbf2_9ce4_8422_2325_u64, |hash, x| {
                        (hash ^ u64::from(x.to_bits())).wrapping_mul(0x0100_0000_01b3)
                    });
                state.update_digest(&mut digest);
                timing.kv_encode.as_secs_f64() * 1e6
            })
            .skip(SAMPLES / 4)
            .collect::<Vec<_>>();
        samples.sort_by(f64::total_cmp);
        let median = samples[SAMPLES / 2];
        println!(
            "| {rows} | {median:.1} | {:.1} | {:.3} | {digest:016x} |",
            samples[0],
            median * LAYERS / rows as f64 / 1e3
        );
    }
}
