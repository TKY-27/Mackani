//! Scalar block proposals over a sliding context window.
//! No main weights are opened: the caller supplies frontier/mask embeddings.
use std::fs;
use std::io::{Read, Seek};
use std::path::{Component, Path, PathBuf};

use mackani_format::{Container, Header, VectorType};
use serde_json::Value;

use crate::compressed_linear::{CompressedLinear, quantize_activation};
use crate::draft_head::{Candidate, DraftChoice, HeadLayout, ReducedHead, choose};
use crate::portable_math::{bfloat, exp, exp2, rsqrt, sincos};
use crate::{Error, Result, require_finite};

mod loading;
mod math;
pub use math::{add, attention, dynamic_filter, glu, norm, norm_rows, rope, xor_sum32};
use math::{linear, linear_rows};

/// Residual-layer taps expected by the published draft weights.
pub const DEFAULT_CAPTURE_LAYERS: [usize; 5] = [5, 19, 33, 47, 61];
/// The vocabulary token used to fill unknown draft block positions.
pub const DEFAULT_MASK_TOKEN: u32 = 248070;
/// The number of token positions evaluated by one draft block.
pub const BLOCK_ROWS: usize = 8;
const HEAD_DIM: usize = 128;
const SCALE: f32 = 0.088_388_346;
fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidModel(message.into())
}
fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("draft dimension overflow"))
}
fn dimension(model: &Value, name: &str) -> Result<usize> {
    model
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| invalid(format!("draft model.{name} must be positive")))
}
/// Capture taps and candidate limits for block proposals.
#[derive(Clone, Debug)]
pub struct DraftOptions {
    pub mask_token: u32,
    pub capture_layers: Vec<usize>,
    pub shortlist: usize,
}
impl Default for DraftOptions {
    fn default() -> Self {
        Self {
            mask_token: DEFAULT_MASK_TOKEN,
            capture_layers: DEFAULT_CAPTURE_LAYERS.to_vec(),
            shortlist: 512,
        }
    }
}
/// Validated architecture dimensions of the block draft model.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftDimensions {
    pub hidden: usize,
    pub features: usize,
    pub feed_forward: usize,
    pub vocabulary: usize,
    pub layers: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub window: usize,
    pub correction_rank: usize,
    pub epsilon: f32,
}
impl DraftDimensions {
    /// Read and validate the stored dimensions and byte extents.
    pub fn read(header: &Header) -> Result<Self> {
        let m = &header.model;
        if m.get("arch").and_then(Value::as_str) != Some("bdraft2") {
            return Err(invalid("requires arch=bdraft2"));
        }
        let hidden = dimension(m, "hidden")?;
        let features = header
            .tensors
            .iter()
            .find(|t| t.name == "bdraft2.fc")
            .ok_or_else(|| invalid("missing bdraft2.fc"))?
            .cols;
        let result = Self {
            hidden,
            features,
            feed_forward: dimension(m, "ffn")?,
            vocabulary: dimension(m, "vocab")?,
            layers: dimension(m, "layers")?,
            query_heads: dimension(m, "heads")?,
            kv_heads: dimension(m, "kv_heads")?,
            window: dimension(m, "window")?,
            correction_rank: dimension(m, "rank")?,
            epsilon: m
                .get("eps")
                .and_then(Value::as_f64)
                .ok_or_else(|| invalid("draft eps missing"))? as f32,
        };
        if hidden % 512 != 0
            || result.feed_forward % 512 != 0
            || result.query_heads % result.kv_heads != 0
            || product(result.query_heads, HEAD_DIM)? % 512 != 0
            || features % hidden != 0
            || result.vocabulary < 16
            || result.vocabulary > i32::MAX as usize
            || result.window < 2
            || result.window > 2048
            || !result.window.is_power_of_two()
            || result.correction_rank % 32 != 0
            || !result.epsilon.is_finite()
            || result.epsilon <= 0.0
            || dimension(m, "head_dim")? != HEAD_DIM
            || dimension(m, "block")? != BLOCK_ROWS
            || dimension(m, "top_k")? != 16
        {
            return Err(invalid(
                "unsupported draft dimensions (128 channels, block 8, top16 required)",
            ));
        }
        // Bound all composite extents before loading/allocating them.
        for (a, b) in [
            (hidden, 4),
            (hidden, result.correction_rank),
            (result.vocabulary, result.correction_rank),
            (
                result.layers,
                product(result.window, product(result.kv_heads, HEAD_DIM)?)?,
            ),
        ] {
            product(a, b)?;
        }
        Ok(result)
    }
}

/// One committed main-model row, concatenated [capture_slot, hidden_channel].
/// Slots default to zero-based layers 5,19,33,47,61. Tap AFTER the layer's MLP
/// down residual addition, BEFORE the next layer/final normalization. Values
/// are BF16 residuals expanded exactly to f32: bf(old_residual + bf(down)).
/// No extra gain/norm/rotation. Source: engine35.cpp:798–799,830–833,1650;
/// q35_kernels.cu:1437–1444. Batch capture `[slot,row,channel]` must be gathered
/// into this `[slot,channel]` order (q35_bdraft.cu:249–252,490–499).
pub struct ContextFeatures<'a> {
    pub values: &'a [f32],
}

/// An owned copy of the draft's sliding context ring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftSnapshot {
    dimensions: (usize, usize, usize),
    position: usize,
    keys: Vec<Vec<u16>>,
    values: Vec<Vec<u16>>,
}
impl DraftSnapshot {
    /// A zeroed ring at position 0: `layers` arrays of [kv_head, slot, channel] BF16 bits.
    pub fn empty(layers: usize, kv_heads: usize, window: usize) -> Self {
        let n = kv_heads * window * HEAD_DIM;
        Self {
            dimensions: (layers, kv_heads, window),
            position: 0,
            keys: vec![vec![0; n]; layers],
            values: vec![vec![0; n]; layers],
        }
    }
    /// The absolute position immediately following committed input.
    pub fn position(&self) -> usize {
        self.position
    }
    /// (layers, kv_heads, window); restore requires equal shapes.
    pub fn shape(&self) -> (usize, usize, usize) {
        self.dimensions
    }
    /// Zero every slot and continue at `position` (reset, or reseeding a full ring suffix).
    pub fn clear(&mut self, position: usize) {
        self.position = position;
        for row in self.keys.iter_mut().chain(&mut self.values) {
            row.fill(0);
        }
    }
    /// Publish one complete context row: each layer's [kv_head, channel] key and value
    /// BF16 bits go to slot `position % window`, then the position advances.
    pub fn push_row(&mut self, layers: &[(Vec<u16>, Vec<u16>)]) {
        let (_, kv_heads, window) = self.dimensions;
        for (layer, (key, value)) in layers.iter().enumerate() {
            for head in 0..kv_heads {
                let offset = (head * window + self.position % window) * HEAD_DIM;
                self.keys[layer][offset..offset + HEAD_DIM]
                    .copy_from_slice(&key[head * HEAD_DIM..(head + 1) * HEAD_DIM]);
                self.values[layer][offset..offset + HEAD_DIM]
                    .copy_from_slice(&value[head * HEAD_DIM..(head + 1) * HEAD_DIM]);
            }
        }
        self.position += 1;
    }
    /// Head-major BF16 bits `[kv_head,slot,channel]` for each layer.
    pub fn keys(&self) -> &[Vec<u16>] {
        &self.keys
    }
    /// BF16 value rows retained by the sliding draft ring.
    pub fn values(&self) -> &[Vec<u16>] {
        &self.values
    }
}
struct Layer {
    query: CompressedLinear,
    key: CompressedLinear,
    value: CompressedLinear,
    output: CompressedLinear,
    gate: CompressedLinear,
    up: CompressedLinear,
    down: CompressedLinear,
    attention_filter: CompressedLinear,
    mlp_filter: CompressedLinear,
    input_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    query_norm: Vec<f32>,
    key_norm: Vec<f32>,
    attention_bases: Vec<f32>,
    mlp_bases: Vec<f32>,
}
/// Scalar block proposals backed by owned draft weights.
pub struct Draft {
    dimensions: DraftDimensions,
    options: DraftOptions,
    signs: Vec<i8>,
    feature_projection: CompressedLinear,
    hidden_norm: Vec<f32>,
    final_norm: Vec<f32>,
    layers: Vec<Layer>,
    correction_projection: Vec<f32>,
    predecessor_vectors: Vec<f32>,
    successor_vectors: Vec<f32>,
    head: ReducedHead,
    ring: DraftSnapshot,
}
/// Chosen proposal tokens and their candidate scores.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftProposal {
    pub tokens: Vec<u32>,
    /// Pre-correction top16, in rescore order.
    pub candidates: Vec<Vec<Candidate>>,
    /// Same IDs/order, with scores corrected using the selected predecessor.
    pub corrected: Vec<Vec<Candidate>>,
}
/// Options must match the container and head dimensions (shared with fast drafts).
pub fn validate_options(
    d: &DraftDimensions,
    options: &DraftOptions,
    head: &HeadLayout,
) -> Result<()> {
    if options.capture_layers.is_empty()
        || product(options.capture_layers.len(), d.hidden)? != d.features
        || options.capture_layers.windows(2).any(|w| w[0] >= w[1])
        || options.mask_token as usize >= d.vocabulary
        || options.shortlist < 16
        || options.shortlist > 2048
        || options.shortlist % 8 != 0
        || options.shortlist > d.vocabulary
        || head.hidden != d.hidden
        || head.vocabulary != d.vocabulary
    {
        return Err(invalid("draft options/head incompatible with dimensions"));
    }
    Ok(())
}
/// Every projection uses a prefix of `bdraft2.fc`'s input signs.
pub fn validate_signs(d: &DraftDimensions, signs: &[i8]) -> Result<()> {
    if [d.hidden, d.feed_forward, product(d.query_heads, HEAD_DIM)?]
        .iter()
        .any(|&n| n > signs.len())
    {
        return Err(invalid("bdraft2.fc input signs too short"));
    }
    Ok(())
}
/// Manifest defaults and canonical containment check, including symlinks.
/// Only draft files are resolved; files.weights is never opened.
pub fn model_paths(folder: impl AsRef<Path>) -> Result<(PathBuf, PathBuf)> {
    let root = folder.as_ref().canonicalize()?;
    let manifest = match fs::read(root.join("yasakani.json")) {
        Ok(b) => serde_json::from_slice::<Value>(&b)?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => serde_json::json!({}),
        Err(e) => return Err(e.into()),
    };
    if !manifest.is_object() || manifest.get("files").is_some_and(|v| !v.is_object()) {
        return Err(invalid("invalid draft manifest"));
    }
    let resolve = |key: &str, default: &str| -> Result<PathBuf> {
        let name = match manifest.get("files").and_then(|f| f.get(key)) {
            Some(v) => v
                .as_str()
                .ok_or_else(|| invalid("draft filename must be a string"))?,
            None => default,
        };
        let path = Path::new(name);
        if name.is_empty()
            || path
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err(invalid("draft path must stay inside model folder"));
        }
        let path = root.join(path).canonicalize()?;
        if !path.starts_with(&root) {
            return Err(invalid("draft symlink escapes model folder"));
        }
        Ok(path)
    };
    Ok((
        resolve("draft", "bdraft2_k8.yskn")?,
        resolve("draft_head", "bdraft2_k8.shead")?,
    ))
}
impl Draft {
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
        self.ring.position
    }
    /// Capture owned state for restoring this session's committed prefix.
    pub fn snapshot(&self) -> DraftSnapshot {
        self.ring.clone()
    }
    /// Validate and restore an owned checkpoint for this session.
    pub fn restore(&mut self, snapshot: &DraftSnapshot) -> Result<()> {
        if self.ring.dimensions != snapshot.dimensions {
            return Err(Error::InvalidInput("incompatible draft snapshot".into()));
        }
        self.ring = snapshot.clone();
        Ok(())
    }
    /// Clear committed and provisional context while retaining model weights.
    pub fn reset(&mut self) {
        self.ring.clear(0);
    }
    // Used only when a long prefill retained a complete ring-capacity suffix.
    pub(crate) fn reset_at(&mut self, position: usize) {
        self.ring.clear(position);
    }
    /// Append one observed residual-feature row to the draft context.
    pub fn push_context(&mut self, position: usize, features: ContextFeatures<'_>) -> Result<()> {
        let d = &self.dimensions;
        if position != self.position() || features.values.len() != d.features {
            return Err(Error::InvalidInput(
                "draft context position/shape mismatch".into(),
            ));
        }
        position
            .checked_add(BLOCK_ROWS)
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or_else(|| Error::InvalidInput("draft position overflow".into()))?;
        require_finite(features.values, "draft context features")?;
        if features
            .values
            .iter()
            .any(|&x| bfloat(x).to_bits() != x.to_bits())
        {
            return Err(Error::InvalidInput(
                "context residual must be BF16 expanded to f32".into(),
            ));
        }
        let context = norm(
            &linear(&self.feature_projection, features.values, &self.signs)?,
            &self.hidden_norm,
            d.epsilon,
        )?;
        // All layers project the SAME context; not sequential blocks (engine35.cpp:1778–1782).
        let mut pending = Vec::with_capacity(d.layers);
        for l in &self.layers {
            let mut k = linear(&l.key, &context, &self.signs)?;
            let v = linear(&l.value, &context, &self.signs)?;
            rope(&mut k, &l.key_norm, position, d.epsilon)?;
            let key: Vec<_> = k.into_iter().map(mackani_format::f32_to_bf16).collect();
            let value: Vec<_> = v.into_iter().map(mackani_format::f32_to_bf16).collect();
            if key
                .iter()
                .chain(&value)
                .any(|&b| !mackani_format::bf16_to_f32(b).is_finite())
            {
                return Err(Error::NonFinite("draft ring BF16"));
            }
            pending.push((key, value));
        }
        self.ring.push_row(&pending);
        Ok(())
    }
    /// Row-major `[row,capture_slot,channel]`; exactly the same per-row operations.
    /// Completed rows stay committed if a later row fails.
    pub fn push_context_rows(&mut self, position: usize, features: &[f32]) -> Result<()> {
        if features.len() % self.dimensions.features != 0 || position != self.position() {
            return Err(Error::InvalidInput(
                "draft context batch shape/position".into(),
            ));
        }
        for row in features.chunks_exact(self.dimensions.features) {
            self.push_context(self.position(), ContextFeatures { values: row })?;
        }
        Ok(())
    }
    /// Embedding provider returns BF16-expanded main embedding rows for frontier/mask.
    pub fn propose(
        &self,
        frontier: u32,
        mut embedding: impl FnMut(u32) -> Result<Vec<f32>>,
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
        let mut result = DraftProposal {
            tokens: Vec::new(),
            candidates: Vec::new(),
            corrected: Vec::new(),
        };
        let mut predecessor = frontier;
        for (j, x) in hidden.chunks_exact(self.dimensions.hidden).enumerate() {
            let candidates = self.head.candidates(x, self.options.shortlist)?;
            let corrected = self.correct(x, predecessor, &candidates)?;
            let token = choose(&corrected, choice, (self.position() + 1 + j) as u64)?;
            result.tokens.push(token);
            result.candidates.push(candidates);
            result.corrected.push(corrected);
            predecessor = token;
        }
        Ok(result)
    }
    /// Execute eight explicit embedding rows; returns normalized rows 1..7.
    /// Exposed for fast-path conformance and non-causal block experiments.
    pub fn block_hidden(&self, embeddings: &[Vec<f32>]) -> Result<Vec<f32>> {
        let d = &self.dimensions;
        if embeddings.len() != BLOCK_ROWS || embeddings.iter().any(|r| r.len() != d.hidden) {
            return Err(Error::InvalidInput(
                "draft block needs eight hidden-width embedding rows".into(),
            ));
        }
        if self
            .position()
            .checked_add(BLOCK_ROWS)
            .is_none_or(|n| n > i32::MAX as usize)
        {
            return Err(Error::InvalidInput("draft position overflow".into()));
        }
        let mut hidden: Vec<_> = embeddings.iter().flatten().copied().collect();
        require_finite(&hidden, "draft embeddings")?;
        if hidden.iter().any(|&x| bfloat(x).to_bits() != x.to_bits()) {
            return Err(Error::InvalidInput(
                "draft embeddings must be BF16 expanded to f32".into(),
            ));
        }
        for (i, l) in self.layers.iter().enumerate() {
            let normalized = norm_rows(&hidden, &l.input_norm, d.epsilon)?;
            let dynamic = linear_rows(&l.attention_filter, &normalized, &self.signs)?;
            let filtered = dynamic_filter(&normalized, &dynamic, &l.attention_bases, d.hidden, 0);
            let mut q = linear_rows(&l.query, &filtered, &self.signs)?;
            let mut k = linear_rows(&l.key, &filtered, &self.signs)?;
            let v = linear_rows(&l.value, &filtered, &self.signs)?;
            for row in 0..BLOCK_ROWS {
                rope(
                    &mut q[row * d.query_heads * HEAD_DIM..(row + 1) * d.query_heads * HEAD_DIM],
                    &l.query_norm,
                    self.position() + row,
                    d.epsilon,
                )?;
                rope(
                    &mut k[row * d.kv_heads * HEAD_DIM..(row + 1) * d.kv_heads * HEAD_DIM],
                    &l.key_norm,
                    self.position() + row,
                    d.epsilon,
                )?;
            }
            let attended = attention(
                &q,
                &k,
                &v,
                &self.ring.keys[i],
                &self.ring.values[i],
                self.position(),
                d,
            )?;
            let projected = linear_rows(&l.output, &attended, &self.signs)?;
            add(
                &mut hidden,
                &dynamic_filter(&projected, &dynamic, &l.attention_bases, d.hidden, 1),
            )?;
            let normalized = norm_rows(&hidden, &l.post_attention_norm, d.epsilon)?;
            let dynamic = linear_rows(&l.mlp_filter, &normalized, &self.signs)?;
            let filtered = dynamic_filter(&normalized, &dynamic, &l.mlp_bases, d.hidden, 0);
            let gate = linear_rows(&l.gate, &filtered, &self.signs)?;
            let up = linear_rows(&l.up, &filtered, &self.signs)?;
            let projected = linear_rows(&l.down, &glu(&gate, &up), &self.signs)?;
            add(
                &mut hidden,
                &dynamic_filter(&projected, &dynamic, &l.mlp_bases, d.hidden, 1),
            )?;
        }
        norm_rows(&hidden[d.hidden..], &self.final_norm, d.epsilon)
    }
    fn correct(
        &self,
        x: &[f32],
        predecessor: u32,
        candidates: &[Candidate],
    ) -> Result<Vec<Candidate>> {
        let r = self.dimensions.correction_rank;
        let feature = correction_feature(x, &self.correction_projection, r)?;
        let pred =
            &self.predecessor_vectors[predecessor as usize * r..(predecessor as usize + 1) * r];
        let succ: Vec<_> = candidates
            .iter()
            .map(|c| &self.successor_vectors[c.token as usize * r..(c.token as usize + 1) * r])
            .collect();
        correct_candidates(&feature, pred, &succ, candidates)
    }
}

/// The block-draft operations used by `DraftProposer`. Implemented by this scalar
/// reference and by fast drafts, which must reproduce it bit for bit.
pub trait BlockDraft {
    fn dimensions(&self) -> &DraftDimensions;
    fn options(&self) -> &DraftOptions;
    fn position(&self) -> usize;
    fn snapshot(&self) -> DraftSnapshot;
    fn restore(&mut self, snapshot: &DraftSnapshot) -> Result<()>;
    fn reset(&mut self);
    /// Zero the ring and continue at `position`; only for reseeding from a complete
    /// ring-capacity suffix of committed rows.
    fn reset_at(&mut self, position: usize);
    /// `push_context` for consecutive rows starting at `position`; rows before a
    /// failing row stay committed.
    fn push_rows(&mut self, position: usize, rows: &[&[f32]]) -> Result<()>;
    fn propose(
        &mut self,
        frontier: u32,
        embedding: &mut dyn FnMut(u32) -> Result<Vec<f32>>,
        choice: DraftChoice,
    ) -> Result<DraftProposal>;
}
impl BlockDraft for Draft {
    fn dimensions(&self) -> &DraftDimensions {
        Draft::dimensions(self)
    }
    fn options(&self) -> &DraftOptions {
        Draft::options(self)
    }
    fn position(&self) -> usize {
        Draft::position(self)
    }
    fn snapshot(&self) -> DraftSnapshot {
        Draft::snapshot(self)
    }
    fn restore(&mut self, snapshot: &DraftSnapshot) -> Result<()> {
        Draft::restore(self, snapshot)
    }
    fn reset(&mut self) {
        Draft::reset(self)
    }
    fn reset_at(&mut self, position: usize) {
        Draft::reset_at(self, position)
    }
    fn push_rows(&mut self, position: usize, rows: &[&[f32]]) -> Result<()> {
        for (i, &values) in rows.iter().enumerate() {
            self.push_context(position + i, ContextFeatures { values })?;
        }
        Ok(())
    }
    fn propose(
        &mut self,
        frontier: u32,
        embedding: &mut dyn FnMut(u32) -> Result<Vec<f32>>,
        choice: DraftChoice,
    ) -> Result<DraftProposal> {
        Draft::propose(self, frontier, embedding, choice)
    }
}

/// Dense BF16 hidden-to-correction projection `hp` ([rank, hidden], row-major).
pub fn correction_feature(x: &[f32], hp: &[f32], rank: usize) -> Result<Vec<f32>> {
    let hidden = x.len();
    let mut feature = vec![0.0; rank];
    for (rank, value) in feature.iter_mut().enumerate() {
        let mut partial = [0.0_f32; 32];
        for (lane, acc) in partial.iter_mut().enumerate() {
            for i in (lane..hidden).step_by(32) {
                // q35_bdraft.cu:303: dense BF16, NOT compressed.
                *acc = x[i].mul_add(hp[rank * hidden + i], *acc);
            }
        }
        *value = xor_sum32(&partial);
    }
    require_finite(&feature, "draft correction feature")?;
    Ok(feature)
}

/// Predecessor correction: `pred` is the predecessor's row, `succ[i]` candidate i's row.
pub fn correct_candidates(
    feature: &[f32],
    pred: &[f32],
    succ: &[&[f32]],
    candidates: &[Candidate],
) -> Result<Vec<Candidate>> {
    let rank = feature.len();
    let product: Vec<_> = feature
        .iter()
        .enumerate()
        .map(|(r, &h)| pred[r] * h)
        .collect();
    require_finite(&product, "draft predecessor product")?;
    let mut output = Vec::with_capacity(16);
    for (c, succ) in candidates.iter().zip(succ) {
        let mut partial = [0.0_f32; 32];
        for (lane, acc) in partial.iter_mut().enumerate() {
            for r in (lane..rank).step_by(32) {
                // q35_bdraft.cu:409 (also legacy selector :334).
                *acc = product[r].mul_add(succ[r], *acc);
            }
        }
        output.push(Candidate {
            token: c.token,
            score: c.score + xor_sum32(&partial),
        });
    }
    if output.iter().any(|c| !c.score.is_finite()) {
        return Err(Error::NonFinite("draft corrected score"));
    }
    Ok(output)
}
#[cfg(test)]
mod tests;
