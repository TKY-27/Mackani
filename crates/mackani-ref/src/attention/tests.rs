use super::*;

fn layer() -> Attention {
    Attention::new(
        AttentionDimensions {
            query_heads: 2,
            kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
        },
        vec![0.0; 256],
        vec![0.0; 256],
        1.0e-6,
    )
    .unwrap()
}

fn plan(tiles: &[(usize, std::ops::Range<usize>)], include_self: bool) -> SplitPlan {
    SplitPlan {
        tiles: tiles.to_vec(),
        include_self,
    }
}

#[test]
fn prefix_schedule_empty_and_short_prefixes() {
    // Each split owns one slot at these lengths; tile boundaries stay aligned to zero.
    let only_self = prefix_schedule(1);
    assert_eq!(only_self.len(), 31);
    assert_eq!(only_self[0], plan(&[], true));
    assert_eq!(only_self[1..], vec![plan(&[], false); 30]);

    let short = prefix_schedule(5);
    assert_eq!(short.len(), 31);
    assert_eq!(
        short[..5],
        [
            plan(&[(0, 0..1)], false),
            plan(&[(0, 1..2)], false),
            plan(&[(0, 2..3)], false),
            plan(&[(0, 3..4)], false),
            plan(&[], true),
        ]
    );
    assert_eq!(short[5..], vec![plan(&[], false); 26]);
}

#[test]
fn prefix_schedule_33_stops_before_the_self_tile() {
    let schedule = prefix_schedule(33);
    assert_eq!(schedule.len(), 31);
    assert_eq!(
        schedule[..17],
        [
            plan(&[(0, 0..2)], false),
            plan(&[(0, 2..4)], false),
            plan(&[(0, 4..6)], false),
            plan(&[(0, 6..8)], false),
            plan(&[(0, 8..10)], false),
            plan(&[(0, 10..12)], false),
            plan(&[(0, 12..14)], false),
            plan(&[(0, 14..16)], false),
            plan(&[(0, 16..18)], false),
            plan(&[(0, 18..20)], false),
            plan(&[(0, 20..22)], false),
            plan(&[(0, 22..24)], false),
            plan(&[(0, 24..26)], false),
            plan(&[(0, 26..28)], false),
            plan(&[(0, 28..30)], false),
            plan(&[(0, 30..32)], false),
            plan(&[], true),
        ]
    );
    assert_eq!(schedule[17..], vec![plan(&[], false); 14]);
}

#[test]
fn prefix_schedule_76_splits_across_absolute_tile_boundaries() {
    // From chapter 04: 31 splits, three slots each; self is slot 75 in split 25.
    let schedule = prefix_schedule(76);
    assert_eq!(schedule.len(), 31);
    assert_eq!(
        schedule[..26],
        [
            plan(&[(0, 0..3)], false),
            plan(&[(0, 3..6)], false),
            plan(&[(0, 6..9)], false),
            plan(&[(0, 9..12)], false),
            plan(&[(0, 12..15)], false),
            plan(&[(0, 15..18)], false),
            plan(&[(0, 18..21)], false),
            plan(&[(0, 21..24)], false),
            plan(&[(0, 24..27)], false),
            plan(&[(0, 27..30)], false),
            plan(&[(0, 30..32), (32, 32..33)], false),
            plan(&[(32, 33..36)], false),
            plan(&[(32, 36..39)], false),
            plan(&[(32, 39..42)], false),
            plan(&[(32, 42..45)], false),
            plan(&[(32, 45..48)], false),
            plan(&[(32, 48..51)], false),
            plan(&[(32, 51..54)], false),
            plan(&[(32, 54..57)], false),
            plan(&[(32, 57..60)], false),
            plan(&[(32, 60..63)], false),
            plan(&[(32, 63..64), (64, 64..66)], false),
            plan(&[(64, 66..69)], false),
            plan(&[(64, 69..72)], false),
            plan(&[(64, 72..75)], false),
            plan(&[], true),
        ]
    );
    assert_eq!(schedule[26..], vec![plan(&[], false); 5]);
}

#[test]
fn prefix_schedule_4096_uses_42_splits_and_covers_each_prefix_slot_once() {
    let schedule = prefix_schedule(4096);
    assert_eq!(schedule.len(), 42);
    assert_eq!(
        schedule[0],
        plan(
            &[(0, 0..32), (32, 32..64), (64, 64..96), (96, 96..98)],
            false
        )
    );
    assert_eq!(
        schedule[41],
        plan(
            &[(4000, 4018..4032), (4032, 4032..4064), (4064, 4064..4095)],
            true
        )
    );
    let mut visits = [0; 4095];
    for split in &schedule {
        for (tile_start, visible) in &split.tiles {
            assert_eq!(tile_start % 32, 0);
            assert!(!visible.is_empty());
            assert!(visible.start >= *tile_start && visible.end <= tile_start + 32);
            for slot in visible.clone() {
                visits[slot] += 1;
            }
        }
    }
    assert_eq!(visits, [1; 4095]);
    assert_eq!(
        schedule.iter().filter(|split| split.include_self).count(),
        1
    );
}

#[test]
fn length_76_attention_pins_absolute_tile_output_bits() {
    let layer = layer();
    let query: Vec<f32> = (0..256)
        .map(|channel| ((channel * 7 % 29) as f32 - 14.0) * 0.03125)
        .collect();
    let rows: Vec<CacheToken> = (0..76)
        .map(|position| CacheToken {
            keys: vec![
                CacheRow::Int8(Int8Row {
                    integers: (0..256)
                        .map(|channel| (((position * 13 + channel * 7) % 255) as i16 - 127) as i8)
                        .collect(),
                    scale: (position % 7 + 1) as f32 * 0.0078125,
                });
                2
            ],
            values: vec![
                CacheRow::Int8(Int8Row {
                    integers: (0..256)
                        .map(
                            |channel| (((position * 19 + (channel % 16) * 11) % 255) as i16 - 127)
                                as i8
                        )
                        .collect(),
                    scale: (position % 11 + 1) as f32 * 0.015625,
                });
                2
            ],
        })
        .collect();
    let output = layer
        .attend_head(&query, 0, &rows[..75], &rows[75])
        .unwrap();
    // Fixed output bits pin absolute tile placement; values repeat every 16 channels.
    let expected = [
        0xbede_440b,
        0xbeac_aee9,
        0x3ce1_ab7d,
        0xbdb7_14de,
        0xbd93_8526,
        0x3ef5_474a,
        0x3e3f_be6c,
        0x3f4c_2f7e,
        0x3f26_e245,
        0x3f81_1d5b,
        0x3f09_e0ce,
        0x3f0e_8ea2,
        0xbde6_e0e7,
        0xbeb1_494b,
        0xbccd_e3de,
        0xbf2b_79d6,
    ];
    assert_eq!(
        output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected.repeat(16)
    );
}

#[test]
fn int8_encoding_handles_zero_and_halfway_ties() {
    let CacheRow::Int8(zero) = encode_int8(&[0.0; 256]).unwrap() else {
        panic!("int8");
    };
    assert_eq!(zero.scale.to_bits(), 0.0_f32.to_bits());
    assert!(zero.integers.iter().all(|&value| value == 0));
    let CacheRow::Int8(encoded) = encode_int8(&[127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]).unwrap()
    else {
        panic!("int8");
    };
    assert_eq!(encoded.scale, 1.0);
    assert_eq!(encoded.integers, [127, 0, 2, 2, 0, -2, -2]);
}

#[test]
fn cache_rotation_roundtrip_preserves_small_integer_vectors() {
    // Nonzero integers avoid asserting that inverse signs preserve the sign of zero.
    let values: Vec<f32> = (0..256)
        .map(|index| (index % 23) as f32 * 2.0 - 21.0)
        .collect();
    let restored = inverse_rotate_cache(&rotate_cache(&values));
    assert_eq!(
        restored
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn query_projection_keeps_each_heads_gate_interleaved() {
    let layer = layer();
    let mut query_gate = vec![0.0; 1024];
    query_gate[256..512].fill(0.25);
    query_gate[768..1024].fill(-0.5);
    let prepared = layer
        .prepare_row(0, &query_gate, &[0.0; 512], &[0.0; 512])
        .unwrap();
    assert!(prepared.heads[0].gates.iter().all(|&value| value == 0.25));
    assert!(prepared.heads[1].gates.iter().all(|&value| value == -0.5));
    assert!(
        prepared
            .heads
            .iter()
            .all(|head| head.queries.iter().all(|&value| value == 0.0))
    );
}

#[test]
fn normalization_rejects_infinite_argument_before_rsqrt() {
    let mut layer = layer();
    layer.epsilon = f32::MAX;
    assert!(matches!(
        layer.normalize_head(&[1.0e18; 256], &[0.0; 256]),
        Err(Error::NonFinite(_))
    ));
}

#[test]
fn successful_step_appends_cache_and_failed_step_preserves_it() {
    let layer = layer();
    let mut state = layer.new_state().unwrap();
    let output = layer
        .step(&mut state, 0, &[0.0; 1024], &[0.0; 512], &[0.0; 512])
        .unwrap();
    assert_eq!(state.len(), 1);
    assert!(output.iter().all(|value| value.is_finite()));
    let original = state.clone();
    assert!(matches!(
        layer.step(
            &mut state,
            1,
            &[0.0; 1024],
            &[f32::INFINITY; 512],
            &[0.0; 512]
        ),
        Err(Error::NonFinite(_))
    ));
    assert_eq!(state, original);
}

#[test]
fn masked_prefix_cache_contents_cannot_change_output_bits() {
    let layer = layer();
    let mut prefix = vec![
        CacheToken {
            keys: vec![
                CacheRow::Int8(Int8Row {
                    integers: vec![2; 256],
                    scale: 0.25
                });
                2
            ],
            values: vec![
                CacheRow::Int8(Int8Row {
                    integers: vec![-3; 256],
                    scale: 0.125
                });
                2
            ],
        };
        64
    ];
    let visible = 35..38;
    let evaluate = |prefix: &[CacheToken]| {
        let mut split = AttentionSplit {
            maximum: EMPTY_MAXIMUM,
            denominator: 0.0,
            numerator: vec![0.0; 256],
        };
        layer
            .prefix_tile(&mut split, &[0.25; 256], 0, prefix, 32, visible.clone())
            .unwrap();
        (
            split.maximum.to_bits(),
            split.denominator.to_bits(),
            split
                .numerator
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        )
    };
    let expected = evaluate(&prefix);
    assert_eq!(expected.0, 2.0_f32.to_bits());
    assert_eq!(expected.1, 3.0_f32.to_bits());
    assert_eq!(expected.2, vec![(-1.125_f32).to_bits(); 256]);
    for (position, token) in prefix.iter_mut().enumerate() {
        if !visible.contains(&position) {
            for row in token.keys.iter_mut().chain(&mut token.values) {
                if let CacheRow::Int8(row) = row {
                    row.scale = f32::NAN;
                    row.integers.fill(-127);
                }
            }
        }
    }
    assert_eq!(evaluate(&prefix), expected);
    // Neither future slots nor padding beyond the causal cache need to exist.
    prefix.truncate(visible.end);
    assert_eq!(evaluate(&prefix), expected);

    // Substituting +0 operands for masked slots would erase these negative zeros.
    for token in &mut prefix[visible.clone()] {
        if let CacheRow::Int8(row) = &mut token.keys[0] {
            row.integers.fill(0);
        }
        if let CacheRow::Int8(row) = &mut token.values[0] {
            row.scale = f32::from_bits(32);
            row.integers.fill(-127);
        }
    }
    let mut split = AttentionSplit {
        maximum: 0.0,
        denominator: 0.0,
        numerator: vec![-0.0; 256],
    };
    layer
        .prefix_tile(&mut split, &[0.0; 256], 0, &prefix, 32, visible)
        .unwrap();
    assert!(
        split
            .numerator
            .iter()
            .all(|value| value.to_bits() == (-0.0_f32).to_bits())
    );
}

#[test]
fn non_finite_cache_input_is_an_error() {
    assert!(matches!(
        encode_int8(&[f32::INFINITY]),
        Err(Error::NonFinite(_))
    ));
}

#[test]
fn shared_rope_table_matches_per_head_sincos_and_scalar_rows() {
    let layer = Attention::new(
        AttentionDimensions {
            query_heads: 24,
            kv_heads: 4,
            head_dim: 256,
            rotary_dim: 64,
        },
        (0..256).map(|i| i as f32 / 512.0 - 0.25).collect(),
        (0..256).map(|i| 0.125 - i as f32 / 1024.0).collect(),
        1.0e-6,
    )
    .unwrap();
    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let q: Vec<f32> = (0..12_288)
        .map(|i| ((i * 37 % 113) as f32 - 56.0) / 17.0)
        .collect();
    let k: Vec<f32> = (0..1024)
        .map(|i| ((i * 53 % 97) as f32 - 48.0) / 13.0)
        .collect();
    let v: Vec<f32> = (0..1024)
        .map(|i| ((i * 29 % 89) as f32 - 44.0) / 11.0)
        .collect();
    for position in [0, 1, 2, 33, 4095, 30_000, 262_143, 1 << 24] {
        let rope = RopeTable::new(position).unwrap();
        for (pair, &frequency) in FREQUENCY_BITS.iter().enumerate() {
            // The per-head computation this table replaces.
            let (cosine, sine) = sincos(position as f32 * f32::from_bits(frequency)).unwrap();
            assert_eq!(rope.cosines[pair].to_bits(), bfloat(cosine).to_bits());
            assert_eq!(rope.sines[pair].to_bits(), bfloat(sine).to_bits());
        }
        let expected = layer.prepare_row(position, &q, &k, &v).unwrap();
        let heads = (0..24)
            .map(|head| layer.prepare_query_head(&rope, head, &q).unwrap())
            .collect();
        let queries = layer.finish_query_heads(&rope, heads).unwrap();
        let standalone = layer.prepare_queries(position, &q, &k, &v).unwrap();
        for prepared in [&queries, &standalone] {
            assert_eq!(prepared.position, position);
            for (actual, expected) in prepared.heads.iter().zip(&expected.heads) {
                assert_eq!(bits(&actual.queries), bits(&expected.queries));
                assert_eq!(bits(&actual.gates), bits(&expected.gates));
            }
        }
        // Flattened in head order, the heads keep each query beside its gate.
        let flat: Vec<f32> = expected
            .heads
            .iter()
            .flat_map(|head| head.queries.iter().chain(&head.gates).copied())
            .collect();
        assert_eq!(flat.len(), q.len());
        for head in 0..4 {
            assert_eq!(
                layer.prepare_cache_side(position, head, 0, &k).unwrap(),
                expected.cache.keys[head]
            );
            assert_eq!(
                layer.prepare_cache_side(position, head, 1, &v).unwrap(),
                expected.cache.values[head]
            );
        }
    }
}
