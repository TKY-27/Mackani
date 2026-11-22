//! Scalar selectable-cache attention from [chapter 04](../../../docs/spec/04_ATTENTION.md).

use std::sync::Arc;

use crate::gated_deltanet::xor_sum32;
use crate::kv_codec::{
    CacheRow, Int8Row, encode_packed_with_search, requantize_int8_with_search, search_chain,
};
use crate::kv_profile::KvProfile;
use crate::portable_math::{bfloat, exp, exp2, rsqrt, sincos};
use crate::{Error, Result};

const FREQUENCY_BITS: [u32; 32] = [
    0x3f80_0000,
    0x3f1a_b32b,
    0x3eba_f81b,
    0x3e61_f835,
    0x3e08_8d77,
    0x3da5_0956,
    0x3d47_763f,
    0x3cf1_1177,
    0x3c91_ad39,
    0x3c30_1052,
    0x3bd4_ca15,
    0x3b80_967d,
    0x3b1b_690d,
    0x3abb_d3ed,
    0x3a63_01e2,
    0x3a09_2e02,
    0x39a5_cb60,
    0x3948_60c1,
    0x38f2_2ce2,
    0x3892_587e,
    0x3830_df52,
    0x37d5_c441,
    0x3781_2dab,
    0x371c_1fc4,
    0x36bc_b0c1,
    0x3664_0cc6,
    0x3609_cf4a,
    0x35a6_8e4c,
    0x3549_4c57,
    0x34f3_499d,
    0x3493_048e,
    0x3431_af44,
];
const LOG2_E: f32 = std::f32::consts::LOG2_E;
const EMPTY_MAXIMUM: f32 = -1.0e30;

/// Cache arithmetic contract; plain Int8 remains the loader default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvMode {
    #[default]
    Int8,
    Int8Profile,
    Kv5,
}

impl KvMode {
    /// Stable spelling used by command-line options and result files.
    pub fn name(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Int8Profile => "int8-profile",
            Self::Kv5 => "kv5",
        }
    }
}

/// The query, KV and rotary dimensions required by attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionDimensions {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
}

/// Scalar causal attention with a selectable KV cache representation.
#[derive(Clone, Debug)]
pub struct Attention {
    dimensions: AttentionDimensions,
    query_norm: Vec<f32>,
    key_norm: Vec<f32>,
    epsilon: f32,
    query_width: usize,
    kv_width: usize,
    kv_mode: KvMode,
    kv_profile: Option<Arc<KvProfile>>,
    attention_ordinal: usize,
}

/// The encoded key and value heads at one absolute position.
#[derive(Clone, Debug, PartialEq)]
pub struct CacheToken {
    keys: Vec<CacheRow>,
    values: Vec<CacheRow>,
}

impl CacheToken {
    /// The encoded key row for one KV head.
    pub fn key(&self, head: usize) -> &CacheRow {
        &self.keys[head]
    }
    /// The encoded value row for one KV head.
    pub fn value(&self, head: usize) -> &CacheRow {
        &self.values[head]
    }
}

/// Validated, immutable operands for an engine's independent head implementation.
pub struct AttentionOperands<'a> {
    pub query: &'a [f32],
    pub kv_head: usize,
    pub prefix: &'a [CacheToken],
    pub current: &'a CacheToken,
    pub value_mean: Option<&'a [f32; 256]>,
}

/// Causal cache rows. Engines may append provisionally and roll back by truncation.
#[derive(Clone, Debug, PartialEq)]
pub struct AttentionState {
    dimensions: AttentionDimensions,
    rows: Vec<CacheToken>,
}

impl AttentionState {
    /// Remove a provisional suffix without copying or reallocating the prefix.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.rows.len() {
            return Err(Error::InvalidInput(
                "cannot extend an attention cache by truncation".into(),
            ));
        }
        self.rows.truncate(len);
        Ok(())
    }

    /// The allocated capacity in cache positions.
    pub fn capacity(&self) -> usize {
        self.rows.capacity()
    }

    /// Diagnostic FNV-1a over every stored code and scale, independent of allocation.
    pub fn update_digest(&self, digest: &mut u64) {
        self.update_prefix_digest(digest, self.rows.len());
    }

    /// Digest only the committed prefix while later rows are provisional.
    pub fn update_prefix_digest(&self, digest: &mut u64, len: usize) {
        assert!(len <= self.rows.len());
        fn bytes(digest: &mut u64, values: &[u8]) {
            for &byte in values {
                *digest = (*digest ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        }
        bytes(digest, &(len as u64).to_le_bytes());
        for token in &self.rows[..len] {
            for side in [&token.keys, &token.values] {
                bytes(digest, &(side.len() as u64).to_le_bytes());
                for row in side {
                    bytes(digest, &row.scale().to_bits().to_le_bytes());
                    match row {
                        CacheRow::Int8(row) => {
                            bytes(digest, &[0]);
                            for &value in row.integers() {
                                bytes(digest, &[value as u8]);
                            }
                        }
                        CacheRow::Packed(row) => {
                            bytes(digest, &[row.bits()]);
                            for value in row.words() {
                                bytes(digest, &value.to_le_bytes());
                            }
                        }
                    }
                }
            }
        }
    }

    /// The number of encoded cache positions.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the cache contains any encoded positions.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// One position's BF16-rounded RoPE cosines and sines, shared by every head at that position.
#[derive(Clone, Debug, PartialEq)]
pub struct RopeTable {
    position: usize,
    cosines: [f32; FREQUENCY_BITS.len()],
    sines: [f32; FREQUENCY_BITS.len()],
}

impl RopeTable {
    /// Compute the table for one causal position.
    pub fn new(position: usize) -> Result<Self> {
        let mut cosines = [0.0; FREQUENCY_BITS.len()];
        let mut sines = [0.0; FREQUENCY_BITS.len()];
        for (pair, &frequency) in FREQUENCY_BITS.iter().enumerate() {
            // These are chapter 04's exact frequency bits, never regenerated from theta.
            let (cosine, sine) = sincos(position as f32 * f32::from_bits(frequency))?;
            // The cached cosine/sine are individually rounded to BF16.
            cosines[pair] = bfloat(cosine);
            sines[pair] = bfloat(sine);
        }
        Ok(Self {
            position,
            cosines,
            sines,
        })
    }

    /// The causal position these angles belong to.
    pub fn position(&self) -> usize {
        self.position
    }
}

/// One query head's rounded rotated query and rounded gate.
#[derive(Clone)]
pub struct QueryHead {
    queries: Vec<f32>,
    gates: Vec<f32>,
    // Checked when the row is finished, after its cache sides, as in `prepare_row`.
    finite_queries: bool,
    finite_gates: bool,
}

/// Normalized query heads and encoded KV operands ready for attention.
pub struct PreparedRow {
    dimensions: AttentionDimensions,
    position: usize,
    heads: Vec<QueryHead>,
    cache: CacheToken,
}

struct AttentionSplit {
    maximum: f32,
    denominator: f32,
    numerator: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SplitPlan {
    tiles: Vec<(usize, std::ops::Range<usize>)>,
    include_self: bool,
}

/// Use 31/42 splits over absolute, per-row prefix tiles.
/// `length` includes self and is therefore positive.
fn prefix_schedule(length: usize) -> Vec<SplitPlan> {
    let split_count = if length < 4096 { 31 } else { 42 };
    let per_split = length.div_ceil(split_count);
    (0..split_count)
        .map(|split_index| {
            let first = split_index * per_split;
            let last = first.saturating_add(per_split).min(length);
            let prefix_end = first.max(last.min(length - 1));
            let mut tiles = Vec::new();
            if first < prefix_end {
                let mut tile_start = first / 32 * 32;
                while tile_start < prefix_end {
                    tiles.push((
                        tile_start,
                        first.max(tile_start)..prefix_end.min(tile_start + 32),
                    ));
                    tile_start += 32;
                }
            }
            SplitPlan {
                tiles,
                include_self: first < length && length - 1 < first.saturating_add(per_split),
            }
        })
        .collect()
}

impl Attention {
    /// Validate attention dimensions, head weights and normalization epsilon.
    pub fn new(
        dimensions: AttentionDimensions,
        query_norm: Vec<f32>,
        key_norm: Vec<f32>,
        epsilon: f32,
    ) -> Result<Self> {
        if dimensions.query_heads == 0
            || dimensions.kv_heads == 0
            || dimensions.query_heads % dimensions.kv_heads != 0
        {
            return Err(Error::InvalidModel(
                "attention requires positive heads and an integral query/KV head ratio".into(),
            ));
        }
        if dimensions.head_dim != 256 || dimensions.rotary_dim != 64 {
            return Err(Error::InvalidModel(
                "chapter 04 specifies only 256-channel heads and 64 rotary channels".into(),
            ));
        }
        if query_norm.len() != dimensions.head_dim || key_norm.len() != dimensions.head_dim {
            return Err(Error::InvalidModel(
                "attention head normalization weights have the wrong length".into(),
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::InvalidModel(
                "attention epsilon must be finite and positive".into(),
            ));
        }
        finite(&query_norm, "attention query normalization weights")?;
        finite(&key_norm, "attention key normalization weights")?;
        let query_width = dimensions
            .query_heads
            .checked_mul(dimensions.head_dim)
            .ok_or_else(|| Error::InvalidModel("attention query width overflow".into()))?;
        query_width
            .checked_mul(2)
            .ok_or_else(|| Error::InvalidModel("attention query/gate width overflow".into()))?;
        let kv_width = dimensions
            .kv_heads
            .checked_mul(dimensions.head_dim)
            .ok_or_else(|| Error::InvalidModel("attention KV width overflow".into()))?;
        Ok(Self {
            dimensions,
            query_norm,
            key_norm,
            epsilon,
            query_width,
            kv_width,
            kv_mode: KvMode::Int8,
            kv_profile: None,
            attention_ordinal: 0,
        })
    }

    /// Configure before creating sequence state. Arc keeps engine stage calls shareable.
    pub fn with_kv(
        mut self,
        mode: KvMode,
        profile: Option<Arc<KvProfile>>,
        attention_ordinal: usize,
    ) -> Result<Self> {
        if mode != KvMode::Int8 {
            let profile = profile.as_ref().ok_or_else(|| {
                Error::InvalidModel("profile KV mode requires a KVQP profile".into())
            })?;
            if profile.kv_heads() != self.dimensions.kv_heads
                || attention_ordinal >= profile.attention_layers()
            {
                return Err(Error::InvalidModel(
                    "attention/profile dimensions differ".into(),
                ));
            }
        }
        self.kv_mode = mode;
        self.kv_profile = if mode == KvMode::Int8 { None } else { profile };
        self.attention_ordinal = attention_ordinal;
        Ok(self)
    }

    fn group_index(&self, head: usize, side: usize) -> usize {
        // src/kvq_atxpk.hpp:65-68; ordinal counts attention layers only.
        (self.attention_ordinal * self.dimensions.kv_heads + head) * 2 + side
    }

    fn encode_row(
        &self,
        head: usize,
        side: usize,
        position: usize,
        row: &[f32],
    ) -> Result<CacheRow> {
        self.encode_row_with_search(head, side, position, row, search_chain)
    }

    fn encode_row_with_search(
        &self,
        head: usize,
        side: usize,
        position: usize,
        row: &[f32],
        search: impl Fn(&[i32; 64], &crate::kv_profile::CodeTable, u8, bool) -> Result<[u8; 64]>,
    ) -> Result<CacheRow> {
        if self.kv_mode == KvMode::Int8 {
            return encode_int8(row);
        }
        let profile = self.kv_profile.as_ref().expect("validated KV profile");
        let group = self.group_index(head, side);
        let row = row.try_into().expect("256-channel cache row");
        // src/kvq_fast.cuh:78,189,218-234: guard uses the absolute cache position.
        match self.kv_mode {
            KvMode::Int8 => unreachable!(),
            KvMode::Int8Profile => Ok(CacheRow::Int8(requantize_int8_with_search(
                profile, group, position, row, &search,
            )?)),
            KvMode::Kv5 => encode_packed_with_search(profile, group, position, row, &search),
        }
    }

    fn value_mean<'a>(&'a self, head: usize, row: &CacheRow) -> Result<Option<&'a [f32; 256]>> {
        match row {
            CacheRow::Int8(_) => Ok(None),
            CacheRow::Packed(_) => {
                let profile = self.kv_profile.as_ref().ok_or_else(|| {
                    Error::InvalidInput("packed attention row requires a profile".into())
                })?;
                let group = &profile.groups()[self.group_index(head, 1)];
                // Missing packed means still add +0, preserving signed-zero operations.
                Ok(Some(group.mean.as_ref().unwrap_or(&[0.0; 256])))
            }
        }
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

    /// Allocate an empty numeric state matching these layer dimensions.
    pub fn new_state(&self) -> Result<AttentionState> {
        Ok(AttentionState {
            dimensions: self.dimensions,
            rows: Vec::new(),
        })
    }

    /// Consume one causal row and return the input to the compressed output projection.
    pub fn step(
        &self,
        state: &mut AttentionState,
        position: usize,
        projected_query_gate: &[f32],
        projected_key: &[f32],
        projected_value: &[f32],
    ) -> Result<Vec<f32>> {
        if state.dimensions != self.dimensions || state.rows.len() != position {
            return Err(Error::InvalidModel(
                "attention position or state dimensions do not match the causal prefix".into(),
            ));
        }
        self.validate_projected_rows(projected_query_gate, projected_key, projected_value)?;
        let prepared = self.prepare_row(
            position,
            projected_query_gate,
            projected_key,
            projected_value,
        )?;
        let output = self.output_row(state, position, &prepared)?;
        self.append_prepared(state, position, &prepared)?;
        Ok(output)
    }

    /// Append cache rows in ascending position before running independent attention queries.
    pub fn append_prepared(
        &self,
        state: &mut AttentionState,
        position: usize,
        prepared: &PreparedRow,
    ) -> Result<()> {
        if state.dimensions != self.dimensions
            || state.rows.len() != position
            || prepared.dimensions != self.dimensions
            || prepared.position != position
        {
            return Err(Error::InvalidInput("invalid attention cache append".into()));
        }
        state
            .rows
            .try_reserve(1)
            .map_err(|_| Error::InvalidModel("attention cache exceeds available memory".into()))?;
        state.rows.push(prepared.cache.clone());
        Ok(())
    }

    /// The number of independent query heads.
    pub fn query_heads(&self) -> usize {
        self.dimensions.query_heads
    }

    /// Attend and gate every query head for one prepared row.
    pub fn output_row(
        &self,
        state: &AttentionState,
        position: usize,
        prepared: &PreparedRow,
    ) -> Result<Vec<f32>> {
        let mut output = Vec::with_capacity(self.query_width);
        for query_head in 0..self.dimensions.query_heads {
            output.extend(self.output_head(state, position, prepared, query_head)?);
        }
        finite(&output, "attention gated output")?;
        Ok(output)
    }

    /// Attend one head using only this row's causal prefix, even if later cache rows exist.
    pub fn output_head(
        &self,
        state: &AttentionState,
        position: usize,
        prepared: &PreparedRow,
        query_head: usize,
    ) -> Result<Vec<f32>> {
        self.output_head_with(state, position, prepared, query_head, |operands| {
            self.attend_head(
                operands.query,
                query_head,
                operands.prefix,
                operands.current,
            )
        })
    }

    /// Engine head hook with the scalar validation, inverse rotation and output boundaries.
    pub fn output_head_with(
        &self,
        state: &AttentionState,
        position: usize,
        prepared: &PreparedRow,
        query_head: usize,
        attend: impl FnOnce(AttentionOperands<'_>) -> Result<Vec<f32>>,
    ) -> Result<Vec<f32>> {
        self.output_head_with_math::<crate::portable_math::ScalarMath>(
            state, position, prepared, query_head, attend,
        )
    }

    /// Shared attention boundary with independent elementwise output kernels.
    pub fn output_head_with_math<M: crate::portable_math::Elementwise>(
        &self,
        state: &AttentionState,
        position: usize,
        prepared: &PreparedRow,
        query_head: usize,
        attend: impl FnOnce(AttentionOperands<'_>) -> Result<Vec<f32>>,
    ) -> Result<Vec<f32>> {
        let operands = self.head_operands(state, position, prepared, query_head)?;
        let mut output = Vec::with_capacity(self.dimensions.head_dim);
        {
            let attended = attend(operands)?;
            if attended.len() != self.dimensions.head_dim {
                return Err(Error::InvalidInput(
                    "invalid attention head output width".into(),
                ));
            }
            let unrotated = inverse_rotate_cache(&attended);
            M::finite(&unrotated, "attention inverse cache rotation")?;
            let mut gates: [f32; 256] =
                std::array::from_fn(|i| prepared.heads[query_head].gates[i]);
            M::sigmoid(&mut gates);
            M::bfloat(&mut gates);
            for (channel, value) in unrotated.into_iter().enumerate() {
                // The inverse rotation, sigmoid, and product have separate BF16 boundaries.
                let rounded_attended = bfloat(value);
                let gate = gates[channel];
                output.push(bfloat(rounded_attended * gate));
            }
        }
        M::finite(&output, "attention gated output")?;
        Ok(output)
    }

    /// Borrow validated causal operands for engines that share KV tiles across queries.
    /// The scalar output hook uses this same validation and operand selection body.
    pub fn head_operands<'a>(
        &'a self,
        state: &'a AttentionState,
        position: usize,
        prepared: &'a PreparedRow,
        query_head: usize,
    ) -> Result<AttentionOperands<'a>> {
        if state.dimensions != self.dimensions
            || position > state.rows.len()
            || query_head >= self.dimensions.query_heads
            || prepared.dimensions != self.dimensions
            || prepared.position != position
        {
            return Err(Error::InvalidInput(
                "invalid attention prefix or head".into(),
            ));
        }
        let kv_head = query_head / (self.dimensions.query_heads / self.dimensions.kv_heads);
        Ok(AttentionOperands {
            query: &prepared.heads[query_head].queries,
            kv_head,
            prefix: &state.rows[..position],
            current: &prepared.cache,
            value_mean: self.value_mean(kv_head, &prepared.cache.values[kv_head])?,
        })
    }

    fn validate_projected_rows(
        &self,
        projected_query_gate: &[f32],
        projected_key: &[f32],
        projected_value: &[f32],
    ) -> Result<()> {
        if projected_query_gate.len() != 2 * self.query_width
            || projected_key.len() != self.kv_width
            || projected_value.len() != self.kv_width
        {
            return Err(Error::InvalidModel(
                "attention projected row dimensions do not match the layer".into(),
            ));
        }
        finite(projected_query_gate, "attention query/gate projection")?;
        finite(projected_key, "attention key projection")?;
        finite(projected_value, "attention value projection")
    }

    /// Normalize queries and encode both cache sides for one projected row.
    pub fn prepare_row(
        &self,
        position: usize,
        projected_query_gate: &[f32],
        projected_key: &[f32],
        projected_value: &[f32],
    ) -> Result<PreparedRow> {
        self.validate_projected_rows(projected_query_gate, projected_key, projected_value)?;
        let rope = RopeTable::new(position)?;
        let mut prepared = self.prepare_query_heads(&rope, projected_query_gate)?;
        let mut keys = Vec::with_capacity(self.dimensions.kv_heads);
        let mut values = Vec::with_capacity(self.dimensions.kv_heads);
        for head in 0..self.dimensions.kv_heads {
            let start = head * self.dimensions.head_dim;
            let end = start + self.dimensions.head_dim;
            // The compressed projection boundary is BF16, before head norm or rotation.
            let key: Vec<f32> = projected_key[start..end]
                .iter()
                .map(|&value| bfloat(value))
                .collect();
            let value: Vec<f32> = projected_value[start..end]
                .iter()
                .map(|&value| bfloat(value))
                .collect();
            let normalized_key = self.normalize_head(&key, &self.key_norm)?;
            let rotated_key = rotate_cache(&self.rope(&normalized_key, &rope)?);
            keys.push(self.encode_row(head, 0, position, &rotated_key)?);
            // Values receive neither head normalization nor RoPE.
            values.push(self.encode_row(head, 1, position, &rotate_cache(&value))?);
        }
        finite_queries(&prepared)?;
        prepared.cache = CacheToken { keys, values };
        Ok(prepared)
    }

    /// Independent query/gate preparation for engines that schedule KV sides separately.
    pub fn prepare_queries(
        &self,
        position: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
    ) -> Result<PreparedRow> {
        self.validate_projected_rows(q, k, v)?;
        let prepared = self.prepare_query_heads(&RopeTable::new(position)?, q)?;
        finite_queries(&prepared)?;
        Ok(prepared)
    }

    /// A single independent query head, preserving the scalar boundaries of `prepare_queries`.
    pub fn prepare_query_head(
        &self,
        rope: &RopeTable,
        head: usize,
        q: &[f32],
    ) -> Result<QueryHead> {
        if head >= self.dimensions.query_heads || q.len() != 2 * self.query_width {
            return Err(Error::InvalidInput("invalid attention query head".into()));
        }
        let start = 2 * head * self.dimensions.head_dim;
        let projected = &q[start..start + 2 * self.dimensions.head_dim];
        // Rows validated by `prepare_row` never fail here; engine tasks rely on it.
        finite(projected, "attention query/gate projection")?;
        // Each head stores its query followed immediately by that head's raw gate.
        let (query, gate) = projected.split_at(self.dimensions.head_dim);
        let query: Vec<f32> = query.iter().map(|&value| bfloat(value)).collect();
        let normalized = self.normalize_head(&query, &self.query_norm)?;
        // Only queries have a BF16 boundary after the cache rotation.
        let queries: Vec<f32> = rotate_cache(&self.rope(&normalized, rope)?)
            .into_iter()
            .map(bfloat)
            .collect();
        let gates: Vec<f32> = gate.iter().map(|&value| bfloat(value)).collect();
        Ok(QueryHead {
            finite_queries: queries.iter().all(|value| value.is_finite()),
            finite_gates: gates.iter().all(|value| value.is_finite()),
            queries,
            gates,
        })
    }

    /// Assemble independently prepared query heads, in head order, exactly as `prepare_queries`.
    /// The row's cache sides validate its key and value projections.
    pub fn finish_query_heads(
        &self,
        rope: &RopeTable,
        heads: Vec<QueryHead>,
    ) -> Result<PreparedRow> {
        let prepared = self.query_row(rope.position, heads)?;
        finite_queries(&prepared)?;
        Ok(prepared)
    }

    fn prepare_query_heads(&self, rope: &RopeTable, q: &[f32]) -> Result<PreparedRow> {
        let heads = (0..self.dimensions.query_heads)
            .map(|head| self.prepare_query_head(rope, head, q))
            .collect::<Result<Vec<_>>>()?;
        self.query_row(rope.position, heads)
    }

    fn query_row(&self, position: usize, heads: Vec<QueryHead>) -> Result<PreparedRow> {
        if heads.len() != self.dimensions.query_heads {
            return Err(Error::InvalidInput(
                "invalid attention prepared query heads".into(),
            ));
        }
        Ok(PreparedRow {
            dimensions: self.dimensions,
            position,
            heads,
            cache: CacheToken {
                keys: Vec::new(),
                values: Vec::new(),
            },
        })
    }

    /// The number of shared key/value heads.
    pub fn kv_heads(&self) -> usize {
        self.dimensions.kv_heads
    }

    /// A single independent head/side, preserving the scalar projection boundaries.
    pub fn prepare_cache_side(
        &self,
        position: usize,
        head: usize,
        side: usize,
        projected: &[f32],
    ) -> Result<CacheRow> {
        self.prepare_cache_side_with_search(
            &RopeTable::new(position)?,
            head,
            side,
            projected,
            search_chain,
        )
    }

    /// Same projection boundaries with an independent integer trellis search.
    pub fn prepare_cache_side_with_search(
        &self,
        rope: &RopeTable,
        head: usize,
        side: usize,
        projected: &[f32],
        search: impl Fn(&[i32; 64], &crate::kv_profile::CodeTable, u8, bool) -> Result<[u8; 64]>,
    ) -> Result<CacheRow> {
        if head >= self.dimensions.kv_heads || side > 1 || projected.len() != self.kv_width {
            return Err(Error::InvalidInput("invalid attention cache side".into()));
        }
        finite(
            projected,
            if side == 0 {
                "attention key projection"
            } else {
                "attention value projection"
            },
        )?;
        let start = head * self.dimensions.head_dim;
        let row: Vec<_> = projected[start..start + self.dimensions.head_dim]
            .iter()
            .copied()
            .map(bfloat)
            .collect();
        let rotated = if side == 0 {
            rotate_cache(&self.rope(&self.normalize_head(&row, &self.key_norm)?, rope)?)
        } else {
            rotate_cache(&row)
        };
        self.encode_row_with_search(head, side, rope.position, &rotated, search)
    }

    /// Finish an engine-prepared row; sides are interleaved key/value in head order.
    pub fn finish_prepared(
        &self,
        mut queries: PreparedRow,
        sides: Vec<CacheRow>,
    ) -> Result<PreparedRow> {
        if queries.dimensions != self.dimensions || sides.len() != self.dimensions.kv_heads * 2 {
            return Err(Error::InvalidInput(
                "invalid attention prepared sides".into(),
            ));
        }
        for (i, side) in sides.into_iter().enumerate() {
            if i % 2 == 0 {
                queries.cache.keys.push(side);
            } else {
                queries.cache.values.push(side);
            }
        }
        Ok(queries)
    }

    fn normalize_head(&self, values: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
        finite(values, "attention head input")?;
        let mut square_sum = 0.0;
        // Each consecutive 32-channel group uses XOR reduction; groups fold from zero.
        for group in 0..self.dimensions.head_dim / 32 {
            let products: [f32; 32] = std::array::from_fn(|lane| {
                let value = values[group * 32 + lane];
                value * value
            });
            square_sum += xor_sum32(&products);
        }
        finite(&[square_sum], "attention head sum of squares")?;
        let normalization_argument = square_sum / self.dimensions.head_dim as f32 + self.epsilon;
        finite(&[normalization_argument], "attention head norm argument")?;
        let inverse_rms = rsqrt(normalization_argument);
        // Norm rounding follows both multiplications, not the intermediate RMS product.
        let normalized: Vec<f32> = values
            .iter()
            .zip(weight)
            .map(|(&value, &gain)| bfloat((value * inverse_rms) * (1.0 + gain)))
            .collect();
        finite(&normalized, "attention normalized head")?;
        Ok(normalized)
    }

    fn rope(&self, normalized: &[f32], rope: &RopeTable) -> Result<Vec<f32>> {
        let mut rotated = normalized.to_vec();
        let pairs = self.dimensions.rotary_dim / 2;
        for pair in 0..pairs {
            let (cosine, sine) = (rope.cosines[pair], rope.sines[pair]);
            // Both products are individually rounded to BF16.
            rotated[pair] = bfloat(
                bfloat(normalized[pair] * cosine) + bfloat(-normalized[pair + pairs] * sine),
            );
            rotated[pair + pairs] =
                bfloat(bfloat(normalized[pair + pairs] * cosine) + bfloat(normalized[pair] * sine));
        }
        finite(&rotated, "attention rotary head")?;
        Ok(rotated)
    }
}

fn finite_queries(prepared: &PreparedRow) -> Result<()> {
    if !prepared.heads.iter().all(|head| head.finite_queries) {
        return Err(Error::NonFinite("attention rounded rotated query"));
    }
    if !prepared.heads.iter().all(|head| head.finite_gates) {
        return Err(Error::NonFinite("attention rounded gate"));
    }
    Ok(())
}

mod reduction;

fn cache_sign(channel: usize) -> f32 {
    let mut value = channel as u32 + 1;
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    if value & 1 == 0 { 1.0 } else { -1.0 }
}

fn hadamard(values: &mut [f32]) {
    debug_assert!(values.len().is_power_of_two());
    let mut distance = 1;
    // Ascending butterfly distance is observable after every binary32 add/subtract.
    while distance < values.len() {
        for block in (0..values.len()).step_by(2 * distance) {
            for offset in 0..distance {
                let first = values[block + offset];
                let second = values[block + offset + distance];
                values[block + offset] = first + second;
                values[block + offset + distance] = first - second;
            }
        }
        distance *= 2;
    }
}

fn rotate_cache(values: &[f32]) -> Vec<f32> {
    let mut rotated: Vec<f32> = values
        .iter()
        .enumerate()
        .map(|(channel, &value)| value * cache_sign(channel))
        .collect();
    hadamard(&mut rotated);
    for value in &mut rotated {
        // The supported 256-channel rotation has exactly representable normalization 1/16.
        *value *= 0.0625;
    }
    rotated
}

fn inverse_rotate_cache(values: &[f32]) -> Vec<f32> {
    let mut rotated = values.to_vec();
    hadamard(&mut rotated);
    for (channel, value) in rotated.iter_mut().enumerate() {
        *value = (*value * 0.0625) * cache_sign(channel);
    }
    rotated
}

fn encode_int8(values: &[f32]) -> Result<CacheRow> {
    finite(values, "attention cache input")?;
    let maximum = values.iter().map(|value| value.abs()).fold(0.0, f32::max);
    let scale = maximum / 127.0;
    let integers = values
        .iter()
        .map(|&value| {
            if scale > 0.0 {
                // Cache quantization is nearest-even, then saturation to the symmetric range.
                (value / scale).round_ties_even().clamp(-127.0, 127.0) as i8
            } else {
                0
            }
        })
        .collect();
    Ok(CacheRow::Int8(Int8Row { integers, scale }))
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
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(Error::NonFinite(context))
    }
}

#[cfg(test)]
use crate::test_support::kv_profile as synthetic_kv_profile;

#[cfg(test)]
#[path = "attention_kv_tests.rs"]
mod kv_tests;

#[cfg(test)]
mod tests;
