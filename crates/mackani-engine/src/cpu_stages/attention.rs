//! SIMD across keys for scores and across channels for values.
use crate::Result;
use mackani_ref::{
    Error,
    attention::{Attention, AttentionOperands, AttentionState, PreparedRow},
    kv_codec::CacheRow,
    portable_math::{bfloat, exp, exp2},
};
const LOG2_E: f32 = std::f32::consts::LOG2_E;
const EMPTY_MAXIMUM: f32 = -1.0e30;

struct AttentionSplit {
    maximum: f32,
    denominator: f32,
    numerator: Vec<f32>,
}

fn combine_splits(splits: &[AttentionSplit]) -> Result<Vec<f32>> {
    let largest_maximum = splits
        .iter()
        .map(|split| split.maximum)
        .fold(EMPTY_MAXIMUM, f32::max);
    let mut scales = Vec::with_capacity(splits.len());
    for split in splits {
        if split.maximum > -1.0e29 {
            let difference = split.maximum - largest_maximum;
            finite(&[difference], "attention split maximum difference")?;
            scales.push(exp(difference));
        } else {
            scales.push(0.0);
        }
    }
    let mut denominator = 0.0;
    // Splits, including zero padding, combine in ascending order with separate products.
    for (scale, split) in scales.iter().zip(splits) {
        denominator += *scale * split.denominator;
    }
    finite(&[denominator], "attention combined denominator")?;
    if denominator <= 0.0 {
        return Err(Error::InvalidModel(
            "attention combined denominator is not positive".into(),
        ));
    }
    let mut attended = vec![0.0; 256];
    for (scale, split) in scales.iter().zip(splits) {
        add_scaled(&mut attended, &split.numerator, *scale);
    }
    for value in &mut attended {
        *value /= denominator;
    }
    finite(&attended, "attention combined output")?;
    Ok(attended)
}

fn prefix_probabilities(split: &mut AttentionSplit, scores: &[f32; 32]) -> Result<[f32; 32]> {
    let new_maximum = scores.iter().copied().fold(split.maximum, f32::max);
    let rescale_argument = (split.maximum - new_maximum) * LOG2_E;
    let scaled_maximum = new_maximum * LOG2_E;
    finite(
        &[rescale_argument, scaled_maximum],
        "attention prefix exponent argument",
    )?;
    let rescale = exp2(rescale_argument);
    split.maximum = new_maximum;
    let mut probabilities = [0.0; 32];
    for (index, probability) in probabilities.iter_mut().enumerate() {
        if scores[index] != f32::NEG_INFINITY {
            // The score and maximum are multiplied separately before subtraction.
            let argument = scores[index] * LOG2_E - scaled_maximum;
            finite(&[argument], "attention prefix probability argument")?;
            *probability = argument;
        }
        // The causal mask denotes an exact zero probability, not a model activation.
    }
    super::math::exp2(&mut probabilities);
    for (index, probability) in probabilities.iter_mut().enumerate() {
        if scores[index] == f32::NEG_INFINITY {
            *probability = 0.0;
        }
    }
    finite(&probabilities, "attention prefix probability")?;
    let mut partial = [0.0; 4];
    for (lane, sum) in partial.iter_mut().enumerate() {
        // Preserve the paired probabilities and four interleaved partial sums.
        for group in 0..4 {
            let key = 8 * group + 2 * lane;
            *sum += probabilities[key] + probabilities[key + 1];
        }
    }
    let probability_sum = xor_sum4(partial);
    split.denominator = split.denominator * rescale + probability_sum;
    for value in &mut split.numerator {
        *value *= rescale;
    }
    Ok(probabilities)
}

fn add_self(
    split: &mut AttentionSplit,
    query: &[f32],
    key: &CacheRow,
    value: &CacheRow,
) -> Result<()> {
    let key_bases = key.bases();
    let value_bases = value.bases();
    let mut partial = [0.0; 4];
    for (lane, sum) in partial.iter_mut().enumerate() {
        for chunk in 0..256 / 16 {
            // Self has its own interleaved FP32 order; it is not the prefix mma16.
            for offset in [2 * lane, 2 * lane + 1, 8 + 2 * lane, 9 + 2 * lane] {
                let channel = 16 * chunk + offset;
                *sum += query[channel] * key_bases[channel];
            }
        }
    }
    let score = xor_sum4(partial) * (key.scale() * 0.0625);
    finite(&[score], "attention self score")?;
    let new_maximum = split.maximum.max(score);
    let rescale_argument = (split.maximum - new_maximum) * LOG2_E;
    // Self subtracts before multiplying by LOG2_E, unlike prefix probabilities.
    let probability_argument = (score - new_maximum) * LOG2_E;
    finite(
        &[rescale_argument, probability_argument],
        "attention self exponent argument",
    )?;
    let rescale = exp2(rescale_argument);
    let probability = exp2(probability_argument);
    split.maximum = new_maximum;
    split.denominator = split.denominator * rescale + probability;
    // FP32 self weight and own encoded row: src/q35_atxpk3.cuh:421-458;
    // src/q35_atxpk4.cuh:363-400. Packed self still uses the BF16 table base.
    let value_weight = probability * value.scale();
    for (channel, numerator) in split.numerator.iter_mut().enumerate() {
        *numerator = *numerator * rescale + value_weight * value_bases[channel];
    }
    finite(&[split.denominator], "attention self denominator")?;
    finite(&split.numerator, "attention self numerator")
}
fn add_value_mean(split: &mut AttentionSplit, mean: Option<&[f32; 256]>) -> Result<()> {
    if let Some(mean) = mean {
        for (numerator, &mean) in split.numerator.iter_mut().zip(mean) {
            // After self; denominator is FP32 including self, not BF16 weight sums.
            // src/q35_atxpk3.cuh:443-475; src/q35_atxpk4.cuh:385-417.
            *numerator += split.denominator * mean;
        }
        finite(&split.numerator, "attention split value mean")?;
    }
    // Int8 fallback already restored mean/basis (src/kvq_fast.cuh:214-218,234).
    Ok(())
}

fn xor_sum4(mut partial: [f32; 4]) -> f32 {
    // Self scores and prefix denominators use simultaneous XOR distances 1 then 2.
    for distance in [1, 2] {
        let previous = partial;
        for (lane, value) in partial.iter_mut().enumerate() {
            *value = previous[lane] + previous[lane ^ distance];
        }
    }
    partial[0]
}

fn finite(values: &[f32], context: &'static str) -> Result<()> {
    use mackani_ref::portable_math::Elementwise;
    super::math::FastMath::finite(values, context)
}

fn add_scaled(output: &mut [f32], bases: &[f32], weight: f32) {
    assert_eq!(output.len(), 256);
    assert_eq!(bases.len(), 256);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: output_rows checks enabled() (runtime NEON detection); direct kernel
        // tests check neon_available(). Both checked slices have 256 elements.
        unsafe {
            add_scaled_neon(output, bases, weight);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (out, base) in output.iter_mut().zip(bases) {
        *out += weight * base;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_scaled_neon(output: &mut [f32], bases: &[f32], weight: f32) {
    use std::arch::aarch64::*;
    // SAFETY: both slices contain 256 elements; each load/store covers i..i+4.
    unsafe {
        let output_ptr = output.as_mut_ptr();
        for i in (0..256).step_by(4) {
            let sum = vaddq_f32(
                vld1q_f32(output_ptr.add(i)),
                vmulq_n_f32(vld1q_f32(bases.as_ptr().add(i)), weight),
            );
            vst1q_f32(output_ptr.add(i), sum);
        }
    }
}

#[cfg(test)]
use crate::model::session_test_support::kv_profile as profile_fixture;

#[cfg(test)]
mod tests {
    use super::*;
    use mackani_ref::{KvMode, attention::AttentionDimensions, kv_profile::KvProfile};
    use std::sync::Arc;
    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn attention_matches_reference_at_tile_and_split_boundaries_with_mixed_sides() {
        check_attention_at_boundaries(
            &profile_fixture::mixed_profile(),
            &[(true, true), (true, false)],
        );
    }

    #[test]
    fn attention_matches_reference_at_tile_and_split_boundaries_with_int8_keys_packed_values() {
        check_attention_at_boundaries(
            &profile_fixture::int8_keys_packed_values_profile(),
            &[(false, true)],
        );
    }

    fn check_attention_at_boundaries(profile: &[u8], packed_sides: &[(bool, bool)]) {
        if !super::super::neon_available() {
            return;
        }
        let profile = Arc::new(KvProfile::parse(profile).unwrap());
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
                _ => seed as i32 as f32 / i32::MAX as f32 * 3.0,
            }
        };
        for mode in [KvMode::Int8, KvMode::Int8Profile, KvMode::Kv5] {
            for (ordinal, &packed_sides) in packed_sides.iter().enumerate() {
                let layer = Attention::new(
                    AttentionDimensions {
                        query_heads: 4,
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
                let mut state = layer.new_state().unwrap();
                for position in 0..4107 {
                    let q: Vec<_> = (0..2048).map(|_| sample()).collect();
                    let k: Vec<_> = (0..512).map(|_| sample()).collect();
                    let v: Vec<_> = (0..512).map(|_| sample()).collect();
                    let row = layer.prepare_row(position, &q, &k, &v).unwrap();
                    if [
                        0, 1, 30, 31, 32, 33, 61, 62, 63, 64, 2000, 4094, 4095, 4096, 4106,
                    ]
                    .contains(&position)
                    {
                        // Four causal rows share the cache, including passes crossing
                        // tile edges and the 31 -> 42 split-schedule threshold.
                        let mut pass_state = state.clone();
                        let mut prepared = Vec::new();
                        for offset in 0..4 {
                            let q: Vec<_> = (0..2048).map(|_| sample()).collect();
                            let k: Vec<_> = (0..512).map(|_| sample()).collect();
                            let v: Vec<_> = (0..512).map(|_| sample()).collect();
                            let p = layer.prepare_row(position + offset, &q, &k, &v).unwrap();
                            layer
                                .append_prepared(&mut pass_state, position + offset, &p)
                                .unwrap();
                            prepared.push(p);
                        }
                        // Inspect the stored prefix and final current row for both KV heads:
                        // fixture changes must not silently remove either mixed direction.
                        for head in [0, 2] {
                            let operands = layer
                                .head_operands(&pass_state, position + 3, &prepared[3], head)
                                .unwrap();
                            let expected_sides = if mode == KvMode::Kv5 {
                                packed_sides
                            } else {
                                (false, false)
                            };
                            for (cached_position, token) in operands
                                .prefix
                                .iter()
                                .chain(std::iter::once(operands.current))
                                .enumerate()
                            {
                                assert_eq!(
                                    (
                                        matches!(token.key(operands.kv_head), CacheRow::Packed(_)),
                                        matches!(
                                            token.value(operands.kv_head),
                                            CacheRow::Packed(_)
                                        ),
                                    ),
                                    expected_sides,
                                    "cache {mode:?} ordinal={ordinal} position={cached_position} kv_head={}",
                                    operands.kv_head
                                );
                            }
                        }
                        let operands = || {
                            prepared
                                .iter()
                                .enumerate()
                                .flat_map(|(i, p)| {
                                    let layer = &layer;
                                    let pass_state = &pass_state;
                                    (0..4).map(move |head| {
                                        layer
                                            .head_operands(pass_state, position + i, p, head)
                                            .unwrap()
                                    })
                                })
                                .collect()
                        };
                        let batch = grouped::Batch::new(operands(), 2);
                        // Reverse task completion order must not change split combination.
                        let parts = (0..batch.tasks())
                            .rev()
                            .map(|task| batch.run(task).unwrap())
                            .collect();
                        let actual = batch.finish(parts).unwrap();
                        // Compare real pool scheduling against the reversed completion
                        // order, then compare that result to the independent scalar oracle.
                        for workers in [1, 4, 10] {
                            // Fresh locks also exercise concurrent first construction.
                            let batch = grouped::Batch::new(operands(), 2);
                            let mut pool =
                                crate::Engine::new(crate::Kernel::Portable, workers).unwrap();
                            let parts = pool
                                .map_range(0..batch.tasks(), |task| batch.run(task))
                                .unwrap();
                            let parallel = batch.finish(parts).unwrap();
                            for (got, expected) in parallel.iter().zip(&actual) {
                                assert_eq!(bits(got), bits(expected), "workers={workers}");
                            }
                        }
                        for (i, attended) in actual.into_iter().enumerate() {
                            let expected = layer
                                .output_head(&pass_state, position + i / 4, &prepared[i / 4], i % 4)
                                .unwrap();
                            let actual = layer
                                .output_head_with(
                                    &pass_state,
                                    position + i / 4,
                                    &prepared[i / 4],
                                    i % 4,
                                    |_| Ok(attended),
                                )
                                .unwrap();
                            assert_eq!(
                                bits(&actual),
                                bits(&expected),
                                "grouped {mode:?} ordinal={ordinal} position={} head={}",
                                position + i / 4,
                                i % 4
                            );
                        }
                    }
                    layer.append_prepared(&mut state, position, &row).unwrap();
                }
            }
        }
    }
}

#[path = "attention_grouped.rs"]
mod grouped;
pub(crate) use grouped::output_rows;
