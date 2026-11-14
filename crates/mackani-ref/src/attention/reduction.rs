//! Scalar split attention in fixed multiply/add and reduction order.
use super::*;

impl Attention {
    pub(super) fn attend_head(
        &self,
        query: &[f32],
        query_head: usize,
        prefix: &[CacheToken],
        current: &CacheToken,
    ) -> Result<Vec<f32>> {
        let length = prefix
            .len()
            .checked_add(1)
            .ok_or_else(|| Error::InvalidModel("attention context length overflow".into()))?;
        let kv_head = query_head / (self.dimensions.query_heads / self.dimensions.kv_heads);
        let mean = self.value_mean(kv_head, &current.values[kv_head])?;
        let schedule = prefix_schedule(length);
        let padded_count = schedule.len().next_power_of_two();
        let mut splits = Vec::with_capacity(padded_count);
        for plan in schedule {
            let mut split = AttentionSplit {
                maximum: EMPTY_MAXIMUM,
                denominator: 0.0,
                numerator: vec![0.0; self.dimensions.head_dim],
            };
            for (tile_start, visible) in plan.tiles {
                self.prefix_tile(&mut split, query, kv_head, prefix, tile_start, visible)?;
            }
            if plan.include_self {
                self.add_self(
                    &mut split,
                    query,
                    &current.keys[kv_head],
                    &current.values[kv_head],
                )?;
            }
            Self::add_value_mean(&mut split, mean)?;
            splits.push(split);
        }
        splits.resize_with(padded_count, || AttentionSplit {
            maximum: EMPTY_MAXIMUM,
            denominator: 0.0,
            numerator: vec![0.0; self.dimensions.head_dim],
        });
        let largest_maximum = splits
            .iter()
            .map(|split| split.maximum)
            .fold(EMPTY_MAXIMUM, f32::max);
        let mut scales = Vec::with_capacity(splits.len());
        for split in &splits {
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
        for (scale, split) in scales.iter().zip(&splits) {
            denominator += *scale * split.denominator;
        }
        finite(&[denominator], "attention combined denominator")?;
        if denominator <= 0.0 {
            return Err(Error::InvalidModel(
                "attention combined denominator is not positive".into(),
            ));
        }
        let mut attended = vec![0.0; self.dimensions.head_dim];
        for (channel, output) in attended.iter_mut().enumerate() {
            let mut numerator = 0.0;
            for (scale, split) in scales.iter().zip(&splits) {
                numerator += *scale * split.numerator[channel];
            }
            *output = numerator / denominator;
        }
        finite(&attended, "attention combined output")?;
        Ok(attended)
    }

    pub(super) fn prefix_tile(
        &self,
        split: &mut AttentionSplit,
        query: &[f32],
        kv_head: usize,
        prefix: &[CacheToken],
        tile_start: usize,
        visible: std::ops::Range<usize>,
    ) -> Result<()> {
        let mut scores = [f32::NEG_INFINITY; 32];
        for (tile_index, score) in scores.iter_mut().enumerate() {
            let position = tile_start + tile_index;
            if visible.contains(&position) {
                let key = &prefix[position].keys[kv_head];
                let bases = key.bases();
                let mut accumulator = 0.0;
                for chunk in 0..self.dimensions.head_dim / 16 {
                    // Each 16-key product chain uses ascending scalar multiply then add.
                    for (offset, &query_value) in
                        query[chunk * 16..(chunk + 1) * 16].iter().enumerate()
                    {
                        accumulator += query_value * bases[chunk * 16 + offset];
                    }
                }
                // Shipping QK8=false: no table-int8 scale, only the packed row scale.
                // src/q35_atxpk3.cuh:379; src/q35_atxpk4.cuh:325.
                *score = accumulator * (key.scale() * 0.0625);
                finite(&[*score], "attention prefix score")?;
            }
        }
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
                *probability = exp2(argument);
            }
            // The causal mask denotes an exact zero probability, not a model activation.
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
        // Widen each visible row once, before the channel/weight accumulation loops.
        // Masked slots initialize only None, without zeroing an unread 256-value row.
        let value_bases: [Option<[f32; 256]>; 32] = std::array::from_fn(|tile_index| {
            let position = tile_start + tile_index;
            visible
                .contains(&position)
                .then(|| prefix[position].values[kv_head].bases())
        });
        let mut value_weights = [0.0; 32];
        for (tile_index, weight) in value_weights.iter_mut().enumerate() {
            let position = tile_start + tile_index;
            if visible.contains(&position) {
                let value_scale = prefix[position].values[kv_head].scale();
                // Round probability * scale, not the value: src/q35_atxpk3.cuh:405-406;
                // src/q35_atxpk4.cuh:348-349; src/q35_atxpk.cu:55-56.
                *weight = bfloat(probabilities[tile_index] * value_scale);
            }
        }
        for half in 0..2 {
            for (channel, numerator) in split.numerator.iter_mut().enumerate() {
                // Each 16-key product chain consumes keys in ascending scalar order.
                for (offset, &value_weight) in
                    value_weights[half * 16..(half + 1) * 16].iter().enumerate()
                {
                    // Skip masked operations entirely: adding zero can change signed-zero bits.
                    if let Some(bases) = &value_bases[half * 16 + offset] {
                        *numerator += value_weight * bases[channel];
                    }
                }
            }
        }
        finite(&[split.denominator], "attention prefix denominator")?;
        finite(&split.numerator, "attention prefix numerator")
    }

    pub(super) fn add_self(
        &self,
        split: &mut AttentionSplit,
        query: &[f32],
        key: &CacheRow,
        value: &CacheRow,
    ) -> Result<()> {
        let key_bases = key.bases();
        let value_bases = value.bases();
        let mut partial = [0.0; 4];
        for (lane, sum) in partial.iter_mut().enumerate() {
            for chunk in 0..self.dimensions.head_dim / 16 {
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
}
