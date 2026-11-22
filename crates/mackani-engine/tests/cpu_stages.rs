#[allow(dead_code, clippy::manual_is_multiple_of)]
#[path = "../../mackani-ref/tests/support/mod.rs"]
mod support;
use mackani_engine::{Engine, Kernel, KvMode};
use mackani_ref::attention::{Attention, AttentionDimensions};
use mackani_ref::kv_profile::KvProfile;
use std::sync::Arc;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn independent_cache_sides_match_scalar_rows_and_errors() {
    let profile = Arc::new(KvProfile::parse(&support::kv_profile::mixed_profile()).unwrap());
    let mut engine = Engine::new(Kernel::Portable, 4).unwrap();
    let mut seed = 7919u32;
    let mut sample = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        match seed % 16 {
            0 => 0.0,
            1 => -0.0,
            2 => f32::from_bits(1),
            3 => f32::from_bits(0x80000001),
            4 => f32::from_bits(0x3f808000),
            5 => -127.0,
            _ => (seed as i32 as f32) / (i32::MAX as f32) * 4.0,
        }
    };
    for mode in [KvMode::Int8, KvMode::Int8Profile, KvMode::Kv5] {
        for ordinal in 0..2 {
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
            .unwrap()
            .with_kv(mode, Some(profile.clone()), ordinal)
            .unwrap();
            let mut scalar = layer.new_state().unwrap();
            let mut parallel = layer.new_state().unwrap();
            for position in 0..34 {
                let q: Vec<_> = (0..1024).map(|_| sample()).collect();
                let k: Vec<_> = (0..512).map(|_| sample()).collect();
                let v: Vec<_> = (0..512).map(|_| sample()).collect();
                let expected = layer.prepare_row(position, &q, &k, &v).unwrap();
                let sides = engine
                    .map_range(0..4, |i| {
                        layer.prepare_cache_side(
                            position,
                            i / 2,
                            i % 2,
                            if i % 2 == 0 { &k } else { &v },
                        )
                    })
                    .unwrap();
                let actual = layer
                    .finish_prepared(layer.prepare_queries(position, &q, &k, &v).unwrap(), sides)
                    .unwrap();
                assert_eq!(
                    bits(&layer.output_row(&parallel, position, &actual).unwrap()),
                    bits(&layer.output_row(&scalar, position, &expected).unwrap())
                );
                layer
                    .append_prepared(&mut scalar, position, &expected)
                    .unwrap();
                layer
                    .append_prepared(&mut parallel, position, &actual)
                    .unwrap();
                let (mut a, mut b) = (0, 0);
                scalar.update_digest(&mut a);
                parallel.update_digest(&mut b);
                assert_eq!(a, b);
            }
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let q = vec![0.0; 1024];
                let mut k = vec![0.0; 512];
                let v = vec![0.0; 512];
                k[7] = invalid;
                assert_eq!(
                    layer
                        .prepare_queries(0, &q, &k, &v)
                        .err()
                        .unwrap()
                        .to_string(),
                    layer.prepare_row(0, &q, &k, &v).err().unwrap().to_string()
                );
            }
        }
    }
}

#[test]
fn independent_query_heads_with_shared_rope_match_scalar_rows_and_errors() {
    use mackani_ref::attention::RopeTable;
    let profile = Arc::new(KvProfile::parse(&support::kv_profile::mixed_profile()).unwrap());
    let mut engine = Engine::new(Kernel::Portable, 4).unwrap();
    let mut seed = 104_729u32;
    let mut sample = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        match seed % 16 {
            0 => -0.0,
            1 => f32::from_bits(1),
            2 => f32::from_bits(0x3f808000),
            _ => (seed as i32 as f32) / (i32::MAX as f32) * 4.0,
        }
    };
    let dimensions = AttentionDimensions {
        query_heads: 24,
        kv_heads: 4,
        head_dim: 256,
        rotary_dim: 64,
    };
    let norm: Vec<_> = (0..256).map(|_| sample()).collect();
    // The mixed profile covers two KV heads; keep the published six queries per KV head.
    let profiled = AttentionDimensions {
        query_heads: 12,
        kv_heads: 2,
        ..dimensions
    };
    for mode in [KvMode::Int8, KvMode::Int8Profile, KvMode::Kv5] {
        let layer = Attention::new(profiled, norm.clone(), norm.clone(), 1e-6)
            .unwrap()
            .with_kv(mode, Some(profile.clone()), 0)
            .unwrap();
        let mut scalar = layer.new_state().unwrap();
        let mut parallel = layer.new_state().unwrap();
        for position in 0..34 {
            let q: Vec<_> = (0..6144).map(|_| sample()).collect();
            let k: Vec<_> = (0..512).map(|_| sample()).collect();
            let v: Vec<_> = (0..512).map(|_| sample()).collect();
            let expected = layer.prepare_row(position, &q, &k, &v).unwrap();
            let rope = RopeTable::new(position).unwrap();
            let heads = engine
                .map_range(0..12, |head| layer.prepare_query_head(&rope, head, &q))
                .unwrap();
            let sides = engine
                .map_range(0..4, |i| {
                    layer.prepare_cache_side_with_search(
                        &rope,
                        i / 2,
                        i % 2,
                        if i % 2 == 0 { &k } else { &v },
                        mackani_ref::kv_codec::search_chain,
                    )
                })
                .unwrap();
            let queries = layer.finish_query_heads(&rope, heads).unwrap();
            let actual = layer.finish_prepared(queries, sides).unwrap();
            assert_eq!(
                bits(&layer.output_row(&parallel, position, &actual).unwrap()),
                bits(&layer.output_row(&scalar, position, &expected).unwrap())
            );
            layer
                .append_prepared(&mut scalar, position, &expected)
                .unwrap();
            layer
                .append_prepared(&mut parallel, position, &actual)
                .unwrap();
            let (mut a, mut b) = (0, 0);
            scalar.update_digest(&mut a);
            parallel.update_digest(&mut b);
            assert_eq!(a, b);
        }
    }
    // A head task fails exactly when, and as, the scalar row fails on its inputs.
    let layer = Attention::new(dimensions, norm.clone(), norm, 1e-6).unwrap();
    let rope = RopeTable::new(5).unwrap();
    let (k, v) = (vec![0.5; 1024], vec![0.5; 1024]);
    for index in [7, 300] {
        let mut q = vec![0.5; 12_288];
        q[index] = f32::NAN;
        assert_eq!(
            layer
                .prepare_query_head(&rope, 0, &q)
                .err()
                .unwrap()
                .to_string(),
            layer.prepare_row(5, &q, &k, &v).err().unwrap().to_string()
        );
    }
    assert!(layer.prepare_query_head(&rope, 24, &[0.5; 12_288]).is_err());
    assert!(layer.prepare_query_head(&rope, 0, &[0.5; 12_287]).is_err());
    let missing_head = (0..23)
        .map(|head| {
            layer
                .prepare_query_head(&rope, head, &[0.5; 12_288])
                .unwrap()
        })
        .collect();
    assert!(layer.finish_query_heads(&rope, missing_head).is_err());
}

#[test]
fn cpu_stage_parser_rejects_unknown_values() {
    for value in ["", "invalid"] {
        assert!(value.parse::<mackani_engine::CpuStages>().is_err());
    }
}

fn check_public_stages(variant: &str, mode: KvMode) {
    let neon = {
        #[cfg(target_arch = "aarch64")]
        {
            std::arch::is_aarch64_feature_detected!("neon")
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    };
    println!(
        "CPU stages: requested={variant}, ran={}",
        if variant == "auto" && neon {
            "NEON"
        } else {
            "scalar"
        }
    );
    let folder = std::env::temp_dir().join(format!(
        "mackani-cpu-stages-{}-{variant}-{mode:?}",
        std::process::id()
    ));
    std::fs::create_dir(&folder).unwrap();
    std::fs::write(
        folder.join("weights.yskn"),
        support::synthetic_model(
            107,
            &["linear_attention", "full_attention", "full_attention"],
        ),
    )
    .unwrap();
    std::fs::write(
        folder.join("kvq_u5.bin"),
        support::kv_profile::mixed_profile(),
    )
    .unwrap();
    std::fs::write(
        folder.join("yasakani.json"),
        r#"{"files":{"weights":"weights.yskn"},"max_ctx":64}"#,
    )
    .unwrap();
    let mut reference = mackani_ref::Model::load_folder_with(&folder, mode).unwrap();
    // SAFETY: these synthetic weights remain immutable until both models are dropped.
    let mut model = unsafe {
        mackani_engine::Model::with_execution_options(
            &folder,
            Engine::new(Kernel::Portable, 4).unwrap(),
            mackani_engine::ExecutionOptions {
                backend: mackani_engine::Backend::Cpu,
                cpu_stages: variant.parse().unwrap(),
                ..mackani_engine::ExecutionOptions::default()
            },
            16,
            mode,
        )
    }
    .unwrap();
    assert_eq!(model.cpu_stages(), variant == "auto" && neon);
    let tokens: Vec<_> = (0..36).map(|i| (i * 7 + 11) % 65).collect();
    for chunk in [&[1, 2, 3][..], &tokens, &[17, 21]] {
        reference.prefill(chunk).unwrap();
        model.prefill(chunk).unwrap();
        assert_eq!(model.row_logits().len(), reference.row_logits().len());
        for (actual, expected) in model.row_logits().iter().zip(reference.row_logits()) {
            assert_eq!(bits(actual), bits(expected), "{variant} {mode:?}");
        }
        assert_eq!(
            model.state_digest(),
            reference.state_digest(),
            "{variant} {mode:?}"
        );
    }
    std::fs::remove_dir_all(folder).unwrap();
}

macro_rules! stage_case {
    ($name:ident, $variant:expr, $mode:expr) => {
        #[test]
        fn $name() {
            check_public_stages($variant, $mode);
        }
    };
}
stage_case!(
    auto_int8_model_matches_scalar_across_pass_and_tile_boundaries,
    "auto",
    KvMode::Int8
);
stage_case!(
    auto_int8_profile_model_matches_scalar_across_pass_and_tile_boundaries,
    "auto",
    KvMode::Int8Profile
);
stage_case!(
    auto_kv5_model_matches_scalar_across_pass_and_tile_boundaries,
    "auto",
    KvMode::Kv5
);
stage_case!(
    forced_reference_int8_model_matches_scalar_across_pass_and_tile_boundaries,
    "reference",
    KvMode::Int8
);
stage_case!(
    forced_reference_int8_profile_model_matches_scalar_across_pass_and_tile_boundaries,
    "reference",
    KvMode::Int8Profile
);
stage_case!(
    forced_reference_kv5_model_matches_scalar_across_pass_and_tile_boundaries,
    "reference",
    KvMode::Kv5
);
