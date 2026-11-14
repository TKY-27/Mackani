//! [Chapter 01: main-model architecture](../../../docs/spec/01_MODEL_ARCHITECTURE.md).

use std::fs;
use std::io::{Read, Seek};
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use mackani_format::{Container, VectorType};
use serde_json::Value;

mod checkpoint;
mod execution;
mod loading;
use loading::{load_linear, positive_dimension};
pub use loading::{load_weights, load_weights_with_kv, model_location, model_location_with_kv};

use crate::attention::{Attention, AttentionDimensions, AttentionState};
use crate::compressed_linear::{CompressedLinear, Embedding, EmbeddingTensor, quantize_activation};
use crate::gated_deltanet::{
    GatedDeltaNet, GatedDeltaNetDimensions, GatedDeltaNetState, GatedDeltaNetWeights,
};
use crate::kv_profile::KvProfile;
use crate::output_head_sampler::output_head;
use crate::residual_norm_mlp::{gated_product, prepare_normalization, residual_add};
use crate::{Error, KvMode, Result, require_finite};

/// Validated dimensions shared by model loading and execution.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelDimensions {
    pub hidden_width: usize,
    pub feed_forward_width: usize,
    pub vocabulary_size: usize,
    pub layer_count: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub attention_head_width: usize,
    pub epsilon: f32,
}

impl ModelDimensions {
    fn read(model: &Value) -> Result<Self> {
        if model.get("arch").and_then(Value::as_str) != Some("qwen3_5") {
            return Err(Error::InvalidModel(
                "main reference requires arch=qwen3_5".into(),
            ));
        }
        let dimensions = Self {
            hidden_width: positive_dimension(model, "hidden")?,
            feed_forward_width: positive_dimension(model, "ffn")?,
            vocabulary_size: positive_dimension(model, "vocab")?,
            layer_count: positive_dimension(model, "layers")?,
            query_heads: positive_dimension(model, "heads")?,
            kv_heads: positive_dimension(model, "kv_heads")?,
            attention_head_width: positive_dimension(model, "head_dim")?,
            epsilon: model
                .get("eps")
                .and_then(Value::as_f64)
                .ok_or_else(|| Error::InvalidModel("model.eps must be a number".into()))?
                as f32,
        };
        if dimensions.hidden_width % 512 != 0 || dimensions.feed_forward_width % 512 != 0 {
            return Err(Error::InvalidModel(
                "hidden and feed-forward widths must be multiples of 512".into(),
            ));
        }
        if dimensions.vocabulary_size > u32::MAX as usize {
            return Err(Error::InvalidModel(
                "vocabulary exceeds u32 token IDs".into(),
            ));
        }
        if dimensions.query_heads % dimensions.kv_heads != 0 {
            return Err(Error::InvalidModel(
                "query heads must be divisible by KV heads".into(),
            ));
        }
        if !dimensions.epsilon.is_finite() || dimensions.epsilon <= 0.0 {
            return Err(Error::InvalidModel(
                "model.eps must be finite and positive".into(),
            ));
        }
        Ok(dimensions)
    }
}

/// Dense or compressed token embedding storage.
pub enum Embeddings<T = mackani_format::Tensor> {
    Coded(Box<Embedding<T>>),
    Plain {
        values: Vec<f32>,
        hidden_width: usize,
    },
}

impl<T: EmbeddingTensor> Embeddings<T> {
    /// Reconstruct the embedding values of one vocabulary token.
    pub fn row(&self, token: u32) -> Result<Vec<f32>> {
        match self {
            Self::Coded(embedding) => embedding.row(token as usize),
            Self::Plain {
                values,
                hidden_width,
            } => {
                let start = (token as usize)
                    .checked_mul(*hidden_width)
                    .ok_or_else(|| Error::InvalidInput("embedding token offset overflow".into()))?;
                let end = start
                    .checked_add(*hidden_width)
                    .ok_or_else(|| Error::InvalidInput("embedding token extent overflow".into()))?;
                Ok(values
                    .get(start..end)
                    .ok_or_else(|| {
                        Error::InvalidInput("embedding token is outside the vocabulary".into())
                    })?
                    .to_vec())
            }
        }
    }
}

/// Projection weights and arithmetic for a full-attention branch.
pub struct AttentionLayer<L = CompressedLinear> {
    pub query_gate: L,
    pub key: L,
    pub value: L,
    pub output: L,
    pub attention: Attention,
}

/// Projection weights and arithmetic for a recurrent branch.
pub struct RecurrentLayer<L = CompressedLinear> {
    pub query_key_value: L,
    pub output_gate: L,
    pub output: L,
    pub recurrence: GatedDeltaNet,
}

/// The attention or recurrent operation owned by a model layer.
pub enum Branch<L = CompressedLinear> {
    Attention(Box<AttentionLayer<L>>),
    Recurrent(Box<RecurrentLayer<L>>),
}

/// One model layer's attention branch, normalization and MLP weights.
pub struct Layer<L = CompressedLinear> {
    pub branch: Branch<L>,
    pub input_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub gate: L,
    pub up: L,
    pub down: L,
}

/// The persistent numeric state owned by one model layer.
#[derive(Clone)]
pub enum LayerState {
    Attention(AttentionState),
    Recurrent(GatedDeltaNetState),
}

impl LayerState {
    /// Add all persistent state bytes to a backend-independent FNV-1a digest.
    pub fn update_digest(&self, digest: &mut u64) {
        let tag = match self {
            Self::Attention(_) => 0,
            Self::Recurrent(_) => 1,
        };
        digest_bytes(digest, &[tag]);
        match self {
            Self::Attention(state) => state.update_digest(digest),
            Self::Recurrent(state) => state.update_digest(digest),
        }
    }
}

fn digest_bytes(digest: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *digest = (*digest ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
}

impl<L> Layer<L> {
    /// Allocate an empty numeric state matching these layer dimensions.
    pub fn new_state(&self) -> Result<LayerState> {
        match &self.branch {
            Branch::Attention(layer) => Ok(LayerState::Attention(layer.attention.new_state()?)),
            Branch::Recurrent(layer) => Ok(LayerState::Recurrent(layer.recurrence.new_state()?)),
        }
    }
}

/// Validated model weights parameterized by projection and embedding storage.
pub struct Weights<L = CompressedLinear, T = mackani_format::Tensor> {
    pub dimensions: ModelDimensions,
    pub embeddings: Embeddings<T>,
    pub layers: Vec<Layer<L>>,
    pub final_norm: Vec<f32>,
    pub head: L,
}

/// Optional post-MLP residual taps. Empty layers disable observation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CaptureOptions {
    pub layers: Vec<usize>,
    /// Maximum committed rows retained, normally the draft ring capacity.
    pub max_rows: usize,
}
impl CaptureOptions {
    /// Reject inconsistent dimensions or out-of-range policy values.
    pub fn validate(&self, layer_count: usize) -> Result<()> {
        if !self.layers.is_empty()
            && (self.max_rows == 0
                || self.layers.iter().any(|&layer| layer >= layer_count)
                || self.layers.windows(2).any(|pair| pair[0] >= pair[1]))
        {
            return Err(Error::InvalidInput(
                "invalid capture layers or row capacity".into(),
            ));
        }
        Ok(())
    }
}

/// One absolute input position, in [capture_slot, hidden_channel] order.
#[derive(Clone, Debug, PartialEq)]
pub struct CapturedRow {
    pub position: usize,
    pub values: Vec<f32>,
}

/// Allocated capture storage, including row descriptors and FP32 payloads.
pub fn capture_allocated_bytes(rows: &[CapturedRow], capacity: usize) -> usize {
    capacity * std::mem::size_of::<CapturedRow>()
        + rows
            .iter()
            .map(|row| row.values.capacity() * 4)
            .sum::<usize>()
}

/// An owned committed checkpoint. Attention rows stay in the originating session.
///
/// A checkpoint can only restore its own session, before reset, while its exact
/// token prefix is still present. It contains no attention cache copies.
pub struct Snapshot {
    session: Rc<()>,
    position: usize,
    tokens: Vec<u32>,
    recurrent_states: Vec<Option<GatedDeltaNetState>>,
    committed_logits: Option<Vec<f32>>,
    capture_options: CaptureOptions,
    captures: Vec<CapturedRow>,
}

impl Snapshot {
    /// Heap storage retained by the captured residual rows.
    pub fn capture_bytes(&self) -> usize {
        capture_allocated_bytes(&self.captures, self.captures.capacity())
    }
}

/// A single sequence. Immutable weights are shared only through a single-threaded Rc.
pub struct Model {
    weights: Rc<Weights>,
    layer_states: Vec<LayerState>,
    position: usize,
    context_capacity: usize,
    committed_logits: Option<Vec<f32>>,
    last_pass_logits: Vec<Vec<f32>>,
    pending_tokens: Option<Vec<u32>>,
    committed_tokens: Vec<u32>,
    session: Rc<()>,
    context_identity: Arc<()>,
    capture_options: CaptureOptions,
    captures: Vec<CapturedRow>,
    provisional_captures: Vec<CapturedRow>,
}

type ComputedRows = (Vec<LayerState>, Vec<Vec<f32>>, Vec<CapturedRow>);

impl Model {
    /// Load the model's immutable weights with default options.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_container(Container::open(path)?)
    }

    /// Read only the main weights from chapter 10's optional package manifest.
    ///
    /// `folder` may also be a plain `.yskn` weights path (any non-directory):
    /// that file is loaded directly with the default context capacity of 262,144.
    pub fn load_folder(folder: impl AsRef<Path>) -> Result<Self> {
        Self::load_folder_with(folder, KvMode::Int8)
    }

    /// Profile modes require files.kv_profile (default kvq_u5.bin) inside the folder.
    pub fn load_folder_with(folder: impl AsRef<Path>, mode: KvMode) -> Result<Self> {
        let (path, capacity, profile) = model_location_with_kv(folder.as_ref(), mode)?;
        let mut model = Self::from_container_with_kv(Container::open(path)?, mode, profile)?;
        model.context_capacity = capacity;
        Ok(model)
    }

    /// Load validated scalar weights from an existing container.
    pub fn from_container<R: Read + Seek>(container: Container<R>) -> Result<Self> {
        Self::from_container_with_kv(container, KvMode::Int8, None)
    }

    /// Validate and load scalar weights with an explicit KV representation.
    pub fn from_container_with_kv<R: Read + Seek>(
        mut container: Container<R>,
        mode: KvMode,
        profile: Option<Arc<KvProfile>>,
    ) -> Result<Self> {
        let Weights {
            dimensions,
            embeddings,
            layers,
            final_norm,
            head,
        } = load_weights_with_kv(&mut container, mode, profile, load_linear, |c, name| {
            Ok(c.load_tensor(name)?)
        })?;
        let layer_states = layers.iter().map(Layer::new_state).collect::<Result<_>>()?;
        Ok(Self {
            weights: Rc::new(Weights {
                dimensions,
                embeddings,
                layers,
                final_norm,
                head,
            }),
            layer_states,
            position: 0,
            // This is a session limit from chapter 10, not a model tensor dimension.
            context_capacity: 262_144,
            committed_logits: None,
            last_pass_logits: Vec::new(),
            pending_tokens: None,
            committed_tokens: Vec::new(),
            session: Rc::new(()),
            context_identity: Arc::new(()),
            capture_options: CaptureOptions::default(),
            captures: Vec::new(),
            provisional_captures: Vec::new(),
        })
    }

    /// The validated architecture dimensions used by this session.
    pub fn dimensions(&self) -> &ModelDimensions {
        &self.weights.dimensions
    }
    /// The absolute position immediately following committed input.
    pub fn position(&self) -> usize {
        self.position
    }
    /// The maximum number of input positions retained by this session.
    pub fn context_capacity(&self) -> usize {
        self.context_capacity
    }
    /// The last committed or provisional row's vocabulary logits, when available.
    pub fn logits(&self) -> Option<&[f32]> {
        self.last_pass_logits.last().map(Vec::as_slice)
    }
    /// Vocabulary logits retained from the most recent pass.
    pub fn row_logits(&self) -> &[Vec<f32>] {
        &self.last_pass_logits
    }
    /// Whether candidate rows await commit or discard.
    pub fn has_provisional_pass(&self) -> bool {
        self.pending_tokens.is_some()
    }

    /// Configure before consuming any rows; changing taps mid-prefix loses context.
    pub fn set_capture(&mut self, options: CaptureOptions) -> Result<()> {
        self.require_no_pending_pass()?;
        options.validate(self.dimensions().layer_count)?;
        if self.position != 0 {
            return Err(Error::InvalidInput(
                "configure capture before prefill".into(),
            ));
        }
        self.capture_options = options;
        self.captures.clear();
        Ok(())
    }
    /// The residual taps and retained suffix required by the consumer.
    pub fn capture_options(&self) -> &CaptureOptions {
        &self.capture_options
    }
    /// A stable identity for the currently committed context.
    pub fn context_identity(&self) -> Arc<()> {
        Arc::clone(&self.context_identity)
    }
    /// Move undelivered rows to the consumer; snapshots now omit delivered rows.
    pub fn take_captured_rows(&mut self) -> Vec<CapturedRow> {
        std::mem::take(&mut self.captures)
    }
    /// Heap storage retained by the captured residual rows.
    pub fn capture_bytes(&self) -> usize {
        capture_allocated_bytes(&self.captures, self.captures.capacity())
            + capture_allocated_bytes(
                &self.provisional_captures,
                self.provisional_captures.capacity(),
            )
    }
    /// Committed residual rows awaiting delivery to a draft consumer.
    pub fn captured_rows(&self) -> &[CapturedRow] {
        &self.captures
    }
    /// Residual rows from the current uncommitted candidate pass.
    pub fn provisional_captures(&self) -> &[CapturedRow] {
        &self.provisional_captures
    }
    /// Reconstruct one token's embedding from this model's weights.
    pub fn embedding(&self, token: u32) -> Result<Vec<f32>> {
        self.weights.embeddings.row(token)
    }
    fn retain_captures(&mut self, rows: Vec<CapturedRow>) {
        let excess =
            (self.captures.len() + rows.len()).saturating_sub(self.capture_options.max_rows);
        self.captures.drain(..excess.min(self.captures.len()));
        self.captures.extend(rows);
    }

    /// Create an empty independent session sharing the same immutable weights.
    pub fn new_session(&self) -> Result<Self> {
        Ok(Self {
            weights: Rc::clone(&self.weights),
            layer_states: self
                .weights
                .layers
                .iter()
                .map(Layer::new_state)
                .collect::<Result<_>>()?,
            position: 0,
            context_capacity: self.context_capacity,
            committed_logits: None,
            last_pass_logits: Vec::new(),
            pending_tokens: None,
            committed_tokens: Vec::new(),
            session: Rc::new(()),
            context_identity: Arc::new(()),
            capture_options: self.capture_options.clone(),
            captures: Vec::new(),
            provisional_captures: Vec::new(),
        })
    }

    /// Clear committed and provisional context while retaining model weights.
    pub fn reset(&mut self) -> Result<()> {
        let states = self
            .weights
            .layers
            .iter()
            .map(Layer::new_state)
            .collect::<Result<_>>()?;
        self.layer_states = states;
        self.position = 0;
        self.committed_logits = None;
        self.last_pass_logits.clear();
        self.pending_tokens = None;
        self.committed_tokens.clear();
        self.captures.clear();
        self.provisional_captures.clear();
        self.session = Rc::new(());
        self.context_identity = Arc::new(());
        Ok(())
    }

    /// Consume all tokens atomically. Every row's logits remain available until the next pass.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        self.require_no_pending_pass()?;
        if tokens.is_empty() {
            return Ok(());
        }
        let (states, logits, captures) =
            self.compute_rows(tokens, self.capture_options.max_rows)?;
        self.layer_states = states;
        self.position += tokens.len();
        self.committed_tokens.extend_from_slice(tokens);
        self.committed_logits = logits.last().cloned();
        self.last_pass_logits = logits;
        self.retain_captures(captures);
        Ok(())
    }

    /// Compute a straight causal candidate chain without changing committed state or position.
    pub fn run_provisional(&mut self, tokens: &[u32]) -> Result<&[Vec<f32>]> {
        self.require_no_pending_pass()?;
        if tokens.is_empty() {
            return Err(Error::InvalidInput(
                "a provisional pass must contain at least one row".into(),
            ));
        }
        let (_, logits, captures) = self.compute_rows(tokens, tokens.len())?;
        self.pending_tokens = Some(tokens.to_vec());
        self.provisional_captures = captures;
        self.last_pass_logits = logits;
        Ok(&self.last_pass_logits)
    }

    /// Retain exactly `accepted_rows` input rows, including row zero. No implicit +1.
    pub fn commit(&mut self, accepted_rows: usize) -> Result<()> {
        let tokens = self
            .pending_tokens
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("there is no provisional pass to commit".into()))?;
        if accepted_rows > tokens.len() {
            return Err(Error::InvalidInput(
                "accepted rows exceed the provisional pass".into(),
            ));
        }
        if accepted_rows == 0 {
            self.discard_provisional();
            return Ok(());
        }
        // Replay from the still-committed state. The exact same scalar row path
        // makes commit independent of candidate count and avoids M state snapshots.
        let (states, logits, captures) =
            self.compute_rows(&tokens[..accepted_rows], self.capture_options.max_rows)?;
        self.layer_states = states;
        self.position += accepted_rows;
        self.committed_tokens
            .extend_from_slice(&tokens[..accepted_rows]);
        self.committed_logits = logits.last().cloned();
        self.last_pass_logits = logits;
        self.pending_tokens = None;
        self.provisional_captures.clear();
        self.retain_captures(captures);
        Ok(())
    }

    /// Discard candidate rows and expose the committed logits again.
    pub fn discard_provisional(&mut self) {
        if self.pending_tokens.take().is_some() {
            self.provisional_captures.clear();
            self.last_pass_logits = self.committed_logits.iter().cloned().collect();
        }
    }

    /// Digest committed numeric state independently of provisional rows.
    pub fn state_digest(&self) -> u64 {
        let mut digest = 0xcbf29ce484222325;
        digest_bytes(&mut digest, &(self.position as u64).to_le_bytes());
        for state in &self.layer_states {
            state.update_digest(&mut digest);
        }
        if let Some(logits) = &self.committed_logits {
            for value in logits {
                digest_bytes(&mut digest, &value.to_bits().to_le_bytes());
            }
        }
        digest
    }

    fn require_no_pending_pass(&self) -> Result<()> {
        if self.pending_tokens.is_some() {
            Err(Error::InvalidInput(
                "commit or discard the current provisional pass first".into(),
            ))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod snapshot_capacity_tests;

#[cfg(test)]
mod capture_tests;
