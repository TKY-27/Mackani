//! The scalar reference draft (`mackani_ref::draft`) with mapped weights,
//! compressed projections on Metal or the CPU kernel, and parallel/vectorized
//! attention and head stages, and mapped code streams. Ring contents, proposals,
//! candidates and scores equal the reference bit for bit.
use std::{
    fs::File,
    io::Cursor,
    ops::Range,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use mackani_format::{VectorType, bf16_to_f32, f32_to_bf16};
use mackani_ref::compressed_linear::{PreparedActivation, quantize_activation, restore_projection};
use mackani_ref::draft::{
    self as reference, BLOCK_ROWS, BlockDraft, ContextFeatures, DraftDimensions, DraftOptions,
    DraftProposal, DraftSnapshot, correct_candidates, model_paths, validate_options,
    validate_signs,
};
use mackani_ref::draft_head::{Candidate, DraftChoice, HeadLayout, choose, shortlist, top16};
use mackani_ref::portable_math::bfloat;
use memmap2::Mmap;

use crate::draft_math::{self, HEAD_DIM, RANK, Shape, TILE};
use crate::{Backend, Engine, Error, Kernel, MappedModel, Matrix, Result, model::Compute};

mod head;
mod loading;

/// Rows per push batch: bounds activation and output scratch for long prefills.
const PUSH_BATCH: usize = 256;
/// Approximate-score tiles per pool task (2,048 tokens).
const SCORE_TILES: usize = 256;
/// Rank rows per projection task.
const RANK_TASK: usize = 32;
/// Shortlist candidates per rescoring task.
const RESCORE_TASK: usize = 64;

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidModel(message.into())
}

/// Wall time of the last `propose`, by stage. Stages do not overlap.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct DraftTiming {
    pub total: Duration,
    /// Activation quantization, compressed kernel and restoration (45 matrices).
    pub projections: Duration,
    /// The compressed kernel calls alone (Metal command buffers or CPU pool), included above.
    pub projection_kernel: Duration,
    pub attention: Duration,
    /// Norms, RoPE, dynamic filters, GLU and residual adds.
    pub block_other: Duration,
    /// Rank-512 projection of the seven rows.
    pub head_projection: Duration,
    /// Approximate scores for the whole vocabulary.
    pub head_scores: Duration,
    pub head_shortlist: Duration,
    /// Full-width rescoring and top 16.
    pub head_rescore: Duration,
    /// Correction feature, predecessor correction and choice.
    pub head_select: Duration,
}
impl DraftTiming {
    /// Elapsed time across all reduced-head selection stages.
    pub fn head(&self) -> Duration {
        self.head_projection
            + self.head_scores
            + self.head_shortlist
            + self.head_rescore
            + self.head_select
    }
    /// Embedding calls, validation and assembly outside the timed stages.
    pub fn rest(&self) -> Duration {
        self.total
            .saturating_sub(self.projections + self.attention + self.block_other + self.head())
    }
}

struct Layer {
    query: Matrix,
    key: Matrix,
    value: Matrix,
    output: Matrix,
    gate: Matrix,
    up: Matrix,
    down: Matrix,
    attention_filter: Matrix,
    mlp_filter: Matrix,
    input_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    query_norm: Vec<f32>,
    key_norm: Vec<f32>,
    attention_bases: Vec<f32>,
    mlp_bases: Vec<f32>,
}

/// A mapped BF16 matrix read one row at a time (`bdraft2.sel.pred/succ`).
struct Bf16Rows {
    mapping: Arc<Mmap>,
    range: Range<usize>,
    width: usize,
}
impl Bf16Rows {
    fn row(&self, index: usize) -> Vec<f32> {
        let start = self.range.start + index * self.width * 2;
        self.mapping[start..start + self.width * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16_to_f32(u16::from_le_bytes(*b)))
            .collect()
    }
}

/// The SHEAD file, mapped. Floats are copied (and checked finite), the int8 rescoring
/// rows stay mapped, and the approximate rows are repacked into token tiles.
struct Head {
    mapping: Mmap,
    layout: HeadLayout,
    projection: Vec<f32>,
    approximate: Vec<i8>,
    approximate_scale: Vec<f32>,
    rescore_scale: Vec<f32>,
}
impl Head {
    /// # Safety
    /// The file must not be modified or truncated until the head is dropped.
    unsafe fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the caller guarantees file stability while the mapping lives.
        let mapping = unsafe { Mmap::map(&file)? };
        // The reference reader validates magic, dimensions, every extent and EOF.
        let layout = HeadLayout::read(&mut Cursor::new(&mapping[..]))?;
        let floats = |range: &Range<u64>| -> Result<Vec<f32>> {
            let values: Vec<f32> = mapping[range.start as usize..range.end as usize]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            if values.iter().any(|v| !v.is_finite()) {
                return Err(Error::NonFinite("SHEAD weights/scales"));
            }
            Ok(values)
        };
        let projection = floats(&layout.arrays[0])?;
        let approximate_scale = floats(&layout.arrays[2])?;
        let rescore_scale = floats(&layout.arrays[4])?;
        // Touch every page of the mapped rescoring rows once, so proposals do not
        // pay first-access page faults for newly shortlisted tokens.
        let r = &layout.arrays[3];
        let touched = mapping[r.start as usize..r.end as usize]
            .iter()
            .step_by(4096)
            .fold(0_u8, |acc, &b| acc ^ b);
        std::hint::black_box(touched);
        let a = &layout.arrays[1];
        let approximate = draft_math::pack_approximate(
            &mapping[a.start as usize..a.end as usize],
            layout.vocabulary,
        );
        Ok(Self {
            mapping,
            layout,
            projection,
            approximate,
            approximate_scale,
            rescore_scale,
        })
    }
    fn rescore_row(&self, token: usize) -> &[u8] {
        let start = self.layout.arrays[3].start as usize + token * self.layout.hidden;
        &self.mapping[start..start + self.layout.hidden]
    }
}

/// Mapped block proposals with parallel projection and candidate scoring.
pub struct FastDraft {
    dimensions: DraftDimensions,
    options: DraftOptions,
    signs: Vec<i8>,
    feature_projection: Matrix,
    hidden_norm: Vec<f32>,
    final_norm: Vec<f32>,
    layers: Vec<Layer>,
    correction_projection: Vec<f32>,
    predecessor_vectors: Bf16Rows,
    successor_vectors: Bf16Rows,
    head: Head,
    ring: DraftSnapshot,
    compute: Compute,
    simd: bool,
    timing: DraftTiming,
}

impl FastDraft {
    /// The validated architecture dimensions used by this session.
    pub fn dimensions(&self) -> &DraftDimensions {
        &self.dimensions
    }
    /// The proposal options used by this draft.
    pub fn options(&self) -> &DraftOptions {
        &self.options
    }
    /// The absolute position immediately following committed input.
    pub fn position(&self) -> usize {
        self.ring.position()
    }
    /// The device selected for compressed projections.
    pub fn backend(&self) -> Backend {
        self.compute.backend()
    }
    /// Set the CPU share of Metal projections, as `Model::set_metal_options`.
    pub fn set_metal_options(&mut self, options: crate::MetalOptions) {
        self.compute.set_metal_options(options);
    }
    /// The Metal CPU share in use; `None` on the CPU backend.
    pub fn metal_options(&self) -> Option<crate::MetalOptions> {
        self.compute.metal_options()
    }
    /// The projection device: Metal device name, or CPU kernel and threads.
    pub fn device(&self) -> String {
        self.compute.device()
    }
    /// Whether attention/head float stages use NEON (else the portable loops).
    pub fn vectorized(&self) -> bool {
        self.simd
    }
    /// Stage times of the last `propose`.
    pub fn timing(&self) -> &DraftTiming {
        &self.timing
    }
    /// Capture owned state for restoring this session's committed prefix.
    pub fn snapshot(&self) -> DraftSnapshot {
        self.ring.clone()
    }
    /// Validate and restore an owned checkpoint for this session.
    pub fn restore(&mut self, snapshot: &DraftSnapshot) -> Result<()> {
        if self.ring.shape() != snapshot.shape() {
            return Err(Error::InvalidInput("incompatible draft snapshot".into()));
        }
        self.ring = snapshot.clone();
        Ok(())
    }
    /// Clear committed and provisional context while retaining model weights.
    pub fn reset(&mut self) {
        self.ring.clear(0);
    }

    /// Append one observed residual-feature row to the draft context.
    pub fn push_context(&mut self, position: usize, features: ContextFeatures<'_>) -> Result<()> {
        self.push_rows(position, &[features.values])
    }
    /// Row-major [row, capture_slot, channel], as the reference's batch push.
    pub fn push_context_rows(&mut self, position: usize, features: &[f32]) -> Result<()> {
        let width = self.dimensions.features;
        if !features.len().is_multiple_of(width) || position != self.position() {
            return Err(Error::InvalidInput(
                "draft context batch shape/position".into(),
            ));
        }
        let rows: Vec<_> = features.chunks_exact(width).collect();
        self.push_rows(position, &rows)
    }

    /// Consecutive committed rows from `position`, batched per dispatch. Exactly the
    /// reference's per-row operations; rows before a failing row stay committed.
    pub fn push_rows(&mut self, position: usize, rows: &[&[f32]]) -> Result<()> {
        let mut next = position;
        for batch in rows.chunks(PUSH_BATCH) {
            self.push_batch(next, batch)?;
            next += batch.len();
        }
        Ok(())
    }

    fn push_batch(&mut self, position: usize, rows: &[&[f32]]) -> Result<()> {
        let d = &self.dimensions;
        let mut limit = RowLimit::new(rows.len());
        // The reference's per-row validation, in its order.
        for (i, values) in rows.iter().enumerate() {
            if let Err(error) = validate_context(position + i, self.position() + i, values, d) {
                limit.fail(i, error);
                break;
            }
        }
        let rows = &rows[..limit.rows];
        let signs = &self.signs;
        let prepared = limit.take(self.compute.map_range(0..rows.len(), |i| {
            Ok(quantize_activation(rows[i], &signs[..rows[i].len()]))
        })?);
        let context = if prepared.is_empty() {
            Vec::new()
        } else {
            let [sums] = weighted_sums(&mut self.compute, [&self.feature_projection], &prepared)?;
            let (fc, norm, epsilon) = (&self.feature_projection, &self.hidden_norm, d.epsilon);
            limit.take(self.compute.map_range(0..prepared.len(), |i| {
                Ok(restore(fc, &sums, i, &prepared[i])
                    .and_then(|y| reference::norm(&y, norm, epsilon)))
            })?)
        };
        let hidden = d.hidden;
        let prepared = limit.take(self.compute.map_range(0..context.len(), |i| {
            Ok(quantize_activation(&context[i], &signs[..hidden]))
        })?);
        let rows = prepared.len();
        if rows > 0 {
            // All layers project the SAME context; not sequential blocks (engine35.cpp:1778-1782).
            let matrices: Vec<&Matrix> = self
                .layers
                .iter()
                .flat_map(|l| [&l.key, &l.value])
                .collect();
            let sums = weighted_sums_dyn(&mut self.compute, &matrices, &prepared)?;
            let layers = &self.layers;
            let base = self.ring.position();
            let entries = self.compute.map_range(0..rows * layers.len(), |index| {
                let (i, layer) = (index / layers.len(), index % layers.len());
                Ok(ring_entry(
                    &layers[layer],
                    &sums[2 * layer],
                    &sums[2 * layer + 1],
                    i,
                    &prepared[i],
                    base + i,
                    d.epsilon,
                ))
            })?;
            let mut entries = entries.into_iter();
            let mut ready = Vec::with_capacity(rows);
            for i in 0..rows {
                let row: Vec<_> = entries.by_ref().take(self.layers.len()).collect();
                match row.into_iter().collect::<Result<Vec<_>>>() {
                    Ok(row) => ready.push(row),
                    Err(error) => {
                        limit.fail(i, error);
                        break;
                    }
                }
            }
            for row in ready {
                self.ring.push_row(&row);
            }
        }
        limit.finish()
    }

    /// Embedding provider returns BF16-expanded main embedding rows for frontier/mask.
    pub fn propose(
        &mut self,
        frontier: u32,
        mut embedding: impl FnMut(u32) -> Result<Vec<f32>>,
        choice: DraftChoice,
    ) -> Result<DraftProposal> {
        let started = Instant::now();
        self.timing = DraftTiming::default();
        let result = self.propose_untimed(frontier, &mut embedding, choice);
        self.timing.total = started.elapsed();
        result
    }

    fn propose_untimed(
        &mut self,
        frontier: u32,
        embedding: &mut dyn FnMut(u32) -> Result<Vec<f32>>,
        choice: DraftChoice,
    ) -> Result<DraftProposal> {
        if frontier as usize >= self.dimensions.vocabulary {
            return Err(Error::InvalidInput(
                "draft frontier outside vocabulary".into(),
            ));
        }
        let mut rows = Vec::with_capacity(BLOCK_ROWS);
        for row in 0..BLOCK_ROWS {
            rows.push(embedding(if row == 0 {
                frontier
            } else {
                self.options.mask_token
            })?);
        }
        let hidden = self.block_hidden(&rows)?;
        self.select(&hidden, frontier, choice)
    }

    /// Execute eight explicit embedding rows; returns normalized rows 1..7.
    pub fn block_hidden(&mut self, embeddings: &[Vec<f32>]) -> Result<Vec<f32>> {
        let d = self.dimensions.clone();
        if embeddings.len() != BLOCK_ROWS || embeddings.iter().any(|r| r.len() != d.hidden) {
            return Err(Error::InvalidInput(
                "draft block needs eight hidden-width embedding rows".into(),
            ));
        }
        let position = self.position();
        if position
            .checked_add(BLOCK_ROWS)
            .is_none_or(|n| n > i32::MAX as usize)
        {
            return Err(Error::InvalidInput("draft position overflow".into()));
        }
        let mut hidden: Vec<_> = embeddings.iter().flatten().copied().collect();
        if hidden.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFinite("draft embeddings"));
        }
        if hidden.iter().any(|&x| bfloat(x).to_bits() != x.to_bits()) {
            return Err(Error::InvalidInput(
                "draft embeddings must be BF16 expanded to f32".into(),
            ));
        }
        let shape = Shape {
            query_heads: d.query_heads,
            kv_heads: d.kv_heads,
            window: d.window,
            position,
        };
        let (eps, h) = (d.epsilon, d.hidden);
        for i in 0..self.layers.len() {
            let mut clock = Instant::now();
            let l = &self.layers[i];
            let normalized = reference::norm_rows(&hidden, &l.input_norm, eps)?;
            self.timing.block_other += lap(&mut clock);
            let [dynamic] = project(
                &mut self.compute,
                [&l.attention_filter],
                &normalized,
                &self.signs,
                &mut self.timing.projection_kernel,
            )?;
            self.timing.projections += lap(&mut clock);
            let filtered =
                reference::dynamic_filter(&normalized, &dynamic, &l.attention_bases, h, 0);
            self.timing.block_other += lap(&mut clock);
            let [mut q, mut k, v] = project(
                &mut self.compute,
                [&l.query, &l.key, &l.value],
                &filtered,
                &self.signs,
                &mut self.timing.projection_kernel,
            )?;
            self.timing.projections += lap(&mut clock);
            rope_rows(
                &mut self.compute,
                &mut q,
                &mut k,
                &l.query_norm,
                &l.key_norm,
                &d,
                position,
            )?;
            self.timing.block_other += lap(&mut clock);
            let attended = attention(
                &mut self.compute,
                &q,
                &k,
                &v,
                &self.ring,
                i,
                shape,
                self.simd,
            )?;
            self.timing.attention += lap(&mut clock);
            let [projected] = project(
                &mut self.compute,
                [&l.output],
                &attended,
                &self.signs,
                &mut self.timing.projection_kernel,
            )?;
            self.timing.projections += lap(&mut clock);
            reference::add(
                &mut hidden,
                &reference::dynamic_filter(&projected, &dynamic, &l.attention_bases, h, 1),
            )?;
            let normalized = reference::norm_rows(&hidden, &l.post_attention_norm, eps)?;
            self.timing.block_other += lap(&mut clock);
            let [dynamic] = project(
                &mut self.compute,
                [&l.mlp_filter],
                &normalized,
                &self.signs,
                &mut self.timing.projection_kernel,
            )?;
            self.timing.projections += lap(&mut clock);
            let filtered = reference::dynamic_filter(&normalized, &dynamic, &l.mlp_bases, h, 0);
            self.timing.block_other += lap(&mut clock);
            let [gate, up] = project(
                &mut self.compute,
                [&l.gate, &l.up],
                &filtered,
                &self.signs,
                &mut self.timing.projection_kernel,
            )?;
            self.timing.projections += lap(&mut clock);
            let width = d.feed_forward;
            let glu: Vec<f32> = self
                .compute
                .map_range(0..BLOCK_ROWS, |r| {
                    Ok(reference::glu(
                        &gate[r * width..(r + 1) * width],
                        &up[r * width..(r + 1) * width],
                    ))
                })?
                .concat();
            self.timing.block_other += lap(&mut clock);
            let [projected] = project(
                &mut self.compute,
                [&l.down],
                &glu,
                &self.signs,
                &mut self.timing.projection_kernel,
            )?;
            self.timing.projections += lap(&mut clock);
            reference::add(
                &mut hidden,
                &reference::dynamic_filter(&projected, &dynamic, &l.mlp_bases, h, 1),
            )?;
            self.timing.block_other += lap(&mut clock);
        }
        let mut clock = Instant::now();
        let result = reference::norm_rows(&hidden[h..], &self.final_norm, eps);
        self.timing.block_other += lap(&mut clock);
        result
    }
}

impl BlockDraft for FastDraft {
    fn dimensions(&self) -> &DraftDimensions {
        FastDraft::dimensions(self)
    }
    fn options(&self) -> &DraftOptions {
        FastDraft::options(self)
    }
    fn position(&self) -> usize {
        FastDraft::position(self)
    }
    fn snapshot(&self) -> DraftSnapshot {
        FastDraft::snapshot(self)
    }
    fn restore(&mut self, snapshot: &DraftSnapshot) -> Result<()> {
        FastDraft::restore(self, snapshot)
    }
    fn reset(&mut self) {
        FastDraft::reset(self)
    }
    fn reset_at(&mut self, position: usize) {
        self.ring.clear(position);
    }
    fn push_rows(&mut self, position: usize, rows: &[&[f32]]) -> Result<()> {
        FastDraft::push_rows(self, position, rows)
    }
    fn propose(
        &mut self,
        frontier: u32,
        embedding: &mut dyn FnMut(u32) -> Result<Vec<f32>>,
        choice: DraftChoice,
    ) -> Result<DraftProposal> {
        FastDraft::propose(self, frontier, embedding, choice)
    }
}

fn lap(clock: &mut Instant) -> Duration {
    let now = Instant::now();
    let elapsed = now - *clock;
    *clock = now;
    elapsed
}

/// The rows of a batch still valid; the first failure (lowest row) is kept.
struct RowLimit {
    rows: usize,
    error: Option<Error>,
}
impl RowLimit {
    fn new(rows: usize) -> Self {
        Self { rows, error: None }
    }
    fn fail(&mut self, row: usize, error: Error) {
        if row < self.rows {
            self.rows = row;
            self.error = Some(error);
        }
    }
    /// Keep the leading successful results; a failure truncates later rows.
    fn take<T>(&mut self, results: Vec<Result<T>>) -> Vec<T> {
        let mut kept = Vec::with_capacity(results.len());
        for (row, result) in results.into_iter().enumerate().take(self.rows) {
            match result {
                Ok(value) => kept.push(value),
                Err(error) => {
                    self.fail(row, error);
                    break;
                }
            }
        }
        kept
    }
    fn finish(self) -> Result<()> {
        self.error.map_or(Ok(()), Err)
    }
}

/// The reference `push_context` checks for the row that will land at `expected`.
fn validate_context(
    position: usize,
    expected: usize,
    values: &[f32],
    d: &DraftDimensions,
) -> Result<()> {
    if position != expected || values.len() != d.features {
        return Err(Error::InvalidInput(
            "draft context position/shape mismatch".into(),
        ));
    }
    position
        .checked_add(BLOCK_ROWS)
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or_else(|| Error::InvalidInput("draft position overflow".into()))?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFinite("draft context features"));
    }
    if values.iter().any(|&x| bfloat(x).to_bits() != x.to_bits()) {
        return Err(Error::InvalidInput(
            "context residual must be BF16 expanded to f32".into(),
        ));
    }
    Ok(())
}

/// One layer's ring key/value BF16 bits for context row `i` at `position`.
fn ring_entry(
    layer: &Layer,
    key_sums: &[i64],
    value_sums: &[i64],
    i: usize,
    prepared: &PreparedActivation,
    position: usize,
    epsilon: f32,
) -> Result<(Vec<u16>, Vec<u16>)> {
    let mut k = restore(&layer.key, key_sums, i, prepared)?;
    let v = restore(&layer.value, value_sums, i, prepared)?;
    reference::rope(&mut k, &layer.key_norm, position, epsilon)?;
    let key: Vec<_> = k.into_iter().map(f32_to_bf16).collect();
    let value: Vec<_> = v.into_iter().map(f32_to_bf16).collect();
    if key
        .iter()
        .chain(&value)
        .any(|&b| !bf16_to_f32(b).is_finite())
    {
        return Err(Error::NonFinite("draft ring BF16"));
    }
    Ok((key, value))
}

/// Row-wise RoPE of the block's queries and keys at positions `position + row`.
fn rope_rows(
    compute: &mut Compute,
    query: &mut [f32],
    key: &mut [f32],
    query_norm: &[f32],
    key_norm: &[f32],
    d: &DraftDimensions,
    position: usize,
) -> Result<()> {
    let (qw, kw) = (d.query_heads * HEAD_DIM, d.kv_heads * HEAD_DIM);
    let (q_in, k_in): (&[f32], &[f32]) = (query, key);
    // Reference order per row: queries then keys; errors surface in row order.
    let rotated = compute.map_range(0..2 * BLOCK_ROWS, |index| {
        let (row, keys) = (index / 2, index % 2 == 1);
        let (source, width, weights) = if keys {
            (k_in, kw, key_norm)
        } else {
            (q_in, qw, query_norm)
        };
        let mut x = source[row * width..(row + 1) * width].to_vec();
        reference::rope(&mut x, weights, position + row, d.epsilon)?;
        Ok(x)
    })?;
    for (index, x) in rotated.into_iter().enumerate() {
        let (row, keys) = (index / 2, index % 2 == 1);
        if keys {
            key[row * kw..(row + 1) * kw].copy_from_slice(&x);
        } else {
            query[row * qw..(row + 1) * qw].copy_from_slice(&x);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn attention(
    compute: &mut Compute,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    ring: &DraftSnapshot,
    layer: usize,
    shape: Shape,
    simd: bool,
) -> Result<Vec<f32>> {
    let (keys, values) = (&ring.keys()[layer], &ring.values()[layer]);
    let parts = compute.map_range(0..shape.kv_heads * 32, |index| {
        Ok(draft_math::history_part(
            query,
            keys,
            values,
            shape,
            index / 32,
            index % 32,
            simd,
        ))
    })?;
    let heads = shape.query_heads;
    let group = heads / shape.kv_heads;
    let out = compute.map_range(0..BLOCK_ROWS * heads, |index| {
        let (row, head) = (index / heads, index % heads);
        let kv = head / group;
        draft_math::finish(
            query,
            key,
            value,
            &parts[kv * 32..(kv + 1) * 32],
            shape,
            row,
            head,
        )
    })?;
    let out = out.concat();
    if out.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFinite("draft attention output"));
    }
    Ok(out)
}

fn restore(
    matrix: &Matrix,
    sums: &[i64],
    row: usize,
    prepared: &PreparedActivation,
) -> Result<Vec<f32>> {
    let d = matrix.descriptor();
    let n = d.physical_rows;
    restore_projection(
        &sums[row * n..(row + 1) * n],
        prepared,
        None,
        matrix.effective_row_scale(),
        None,
        d.rows,
    )
}

fn weighted_sums<const N: usize>(
    compute: &mut Compute,
    matrices: [&Matrix; N],
    prepared: &[PreparedActivation],
) -> Result<[Vec<i64>; N]> {
    let sums = weighted_sums_dyn(compute, &matrices, prepared)?;
    Ok(sums.try_into().expect("one sum array per matrix"))
}

/// Exact sums of matrices sharing one prepared input: one Metal command buffer.
fn weighted_sums_dyn(
    compute: &mut Compute,
    matrices: &[&Matrix],
    prepared: &[PreparedActivation],
) -> Result<Vec<Vec<i64>>> {
    let activation: Vec<i8> = prepared
        .iter()
        .flat_map(|p| p.activation8().iter().copied())
        .collect();
    let scales: Vec<i32> = prepared
        .iter()
        .flat_map(|p| p.integer_scale().iter().copied())
        .collect();
    compute.weighted_sums(matrices, &activation, &scales)
}

/// The reference's `linear_rows` for matrices sharing one input: quantization,
/// exact sums and restoration per row; each matrix's rows are concatenated.
fn project<const N: usize>(
    compute: &mut Compute,
    matrices: [&Matrix; N],
    input: &[f32],
    signs: &[i8],
    kernel: &mut Duration,
) -> Result<[Vec<f32>; N]> {
    let cols = matrices[0].descriptor().cols;
    let m = input.len() / cols;
    let prepared = compute.map_range(0..m, |i| {
        quantize_activation(&input[i * cols..(i + 1) * cols], &signs[..cols])
    })?;
    let start = Instant::now();
    let sums = weighted_sums(compute, matrices, &prepared)?;
    *kernel += start.elapsed();
    let rows = compute.map_range(0..N * m, |index| {
        let (j, i) = (index / m, index % m);
        restore(matrices[j], &sums[j], i, &prepared[i])
    })?;
    let mut rows = rows.into_iter();
    Ok(std::array::from_fn(|_| {
        rows.by_ref().take(m).flatten().collect()
    }))
}
