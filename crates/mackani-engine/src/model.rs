//! Layer-major inference over mapped code streams and the reference's scalar primitives.
#[cfg(target_os = "macos")]
use crate::MetalEngine;
use crate::cpu_stages::rest::{
    gated_product, prepare_normalization, quantize_activation, residual_add, restore_projection,
};
use crate::{Engine, Error, MappedModel, Matrix, Result, mapped::MappedEmbeddingTensor};
use mackani_ref::{
    KvMode, ModelDimensions,
    compressed_linear::PreparedActivation,
    gated_deltanet::{GatedDeltaNet, GatedDeltaNetRow, GatedDeltaNetState, round_projection},
    model::{
        Branch, CaptureOptions, CapturedRow, Layer, LayerState, Weights, load_weights_with_kv,
        model_location_with_kv,
    },
    residual_norm_mlp::NormalizedInput,
};
use std::{
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

mod checkpoint;
mod execution;
mod loading;
use execution::replay_recurrence;
mod compute;
pub(crate) use compute::Compute;
pub use compute::{Backend, CpuShare, ExecutionOptions, MetalOptions};

struct Linear {
    matrix: Matrix,
    output_rotation: bool,
}
impl Linear {
    fn input_signs(&self) -> &[i8] {
        self.matrix.input_signs()
    }
    /// The signs of the inverse output rotation, when the consumer applies one.
    fn output_signs(&self) -> Option<&[i8]> {
        self.matrix.output_signs().filter(|_| self.output_rotation)
    }
}

/// Wall times cover nonoverlapping stages of a complete layer-major pass.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct PassTiming {
    pub position: usize,
    pub rows: usize,
    pub total: Duration,
    pub kernel: Duration,
    pub attention: Duration,
    /// Attention row preparation, including KV encoding (also norm/RoPE/gating).
    pub kv_encode: Duration,
    pub recurrence: Duration,
    /// Optional subset of recurrence: rounded history, convolution, normalization and dense gates.
    pub recurrence_prepare: Duration,
    /// Optional sum of dense callback elapsed times (overlaps across rows when M > 1).
    pub dense_gate_work: Duration,
    /// Optional nonoverlapping diagnostic subsets of rest().
    pub normalization: Duration,
    pub activation_encode: Duration,
    pub projection_restore: Duration,
    pub mlp_gate: Duration,
    pub residual: Duration,
    detailed: bool,
}
impl PassTiming {
    fn detailed_start(&self) -> Option<Instant> {
        self.detailed.then(Instant::now)
    }

    /// Elapsed time outside the separately accounted numerical stages.
    pub fn rest(&self) -> Duration {
        self.total
            .saturating_sub(self.kernel + self.attention + self.kv_encode + self.recurrence)
    }
}

/// Vocabulary-head work requested by a committed prefill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogitRows {
    All,
    Last,
    None,
}

/// An owned prompt checkpoint. Attention remains in the owning session.
pub struct Snapshot {
    owner: Arc<()>,
    position: usize,
    tokens: Vec<u32>,
    recurrence: Vec<Option<GatedDeltaNetState>>,
    committed_logits: Option<Vec<f32>>,
    captures: Vec<CapturedRow>,
    capture_options: CaptureOptions,
}

impl Snapshot {
    /// Heap storage retained by the captured residual rows.
    pub fn capture_bytes(&self) -> usize {
        mackani_ref::model::capture_allocated_bytes(&self.captures, self.captures.capacity())
    }
}

/// Persistent storage only; transient projection/attention work is not included.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionMemory {
    pub capture_bytes: usize,
    pub recurrent_state_bytes: usize,
    pub replay_bytes: usize,
    pub replay_capacity_rows: usize,
    pub attention_row_capacities: Vec<usize>,
    pub pending_token_capacity: usize,
    pub history_capacity: usize,
}

/// A main-model session with committed and provisional numeric state.
pub struct Model {
    weights: Weights<Linear, MappedEmbeddingTensor>,
    engine: Compute,
    states: Vec<LayerState>,
    working: Vec<LayerState>,
    replay: Vec<Vec<GatedDeltaNetRow>>,
    pending_tokens: Vec<u32>,
    pending: bool,
    working_complete: bool,
    context_identity: Arc<()>,
    capture_options: CaptureOptions,
    captures: Vec<CapturedRow>,
    pending_captures: Vec<CapturedRow>,
    committed_logits: Option<Vec<f32>>,
    tokens: Vec<u32>,
    owner: Arc<()>,
    position: usize,
    context_capacity: usize,
    rows_per_pass: usize,
    kv_mode: KvMode,
    logits: Vec<Vec<f32>>,
    timings: Vec<PassTiming>,
    detailed_timing: bool,
}
impl Model {
    /// Set the CPU share of Metal projections; CPU-only models ignore it.
    pub fn set_metal_options(&mut self, options: MetalOptions) {
        self.engine.set_metal_options(options);
    }

    /// Select vectorized or reference CPU stages between committed passes.
    /// A pending provisional pass retains the choice that prepared its working state.
    pub fn set_cpu_stages(&mut self, stages: crate::CpuStages) -> Result<()> {
        if self.pending {
            return Err(Error::InvalidInput(
                "CPU stages cannot change during a provisional pass".into(),
            ));
        }
        self.engine.cpu_stages = stages.fast();
        Ok(())
    }

    /// Whether the exact CPU fast paths run.
    pub fn cpu_stages(&self) -> bool {
        self.engine.cpu_stages
    }

    /// The Metal CPU share in use; `None` on the CPU backend.
    pub fn metal_options(&self) -> Option<MetalOptions> {
        self.engine.metal_options()
    }

    /// The CPU kernel and the worker count of this model's pool.
    pub fn kernel(&self) -> crate::Kernel {
        self.engine.kernel()
    }
    /// The number of persistent workers in this CPU pool.
    pub fn threads(&self) -> usize {
        self.engine.threads()
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
    /// The maximum committed or provisional pass width for this model.
    pub fn rows_per_pass(&self) -> usize {
        self.rows_per_pass
    }
    /// The cache representation selected when loading the model.
    pub fn kv_mode(&self) -> KvMode {
        self.kv_mode
    }
    /// The device selected for compressed projections.
    pub fn backend(&self) -> Backend {
        self.engine.backend()
    }
    /// Metal command buffers committed by this model so far; `None` on the CPU backend.
    pub fn metal_command_buffers(&self) -> Option<u64> {
        self.engine.metal_command_buffers()
    }
    /// The projection device: the Metal device name, or the CPU kernel and threads.
    pub fn device(&self) -> String {
        self.engine.device()
    }
    /// Vocabulary logits retained from the most recent pass.
    pub fn row_logits(&self) -> &[Vec<f32>] {
        &self.logits
    }
    /// The last committed or provisional row's vocabulary logits, when available.
    pub fn logits(&self) -> Option<&[f32]> {
        self.logits.last().map(Vec::as_slice)
    }
    /// Collect diagnostic timing subsets on future passes (off by default).
    /// Coarse stage times are always collected; disabled subsets remain zero.
    pub fn set_detailed_timing(&mut self, enabled: bool) {
        self.detailed_timing = enabled;
    }

    /// Nonoverlapping stage timings for the most recent model passes.
    pub fn pass_timings(&self) -> &[PassTiming] {
        &self.timings
    }
    /// Whether candidate rows await commit or discard.
    pub fn has_provisional_pass(&self) -> bool {
        self.pending
    }

    /// Configure observation before any input has been committed.
    pub fn set_capture(&mut self, options: CaptureOptions) -> Result<()> {
        self.require_no_pending_pass()?;
        if self.position != 0 {
            return Err(Error::InvalidInput(
                "capture must be configured before prefill".into(),
            ));
        }
        options.validate(self.weights.layers.len())?;
        self.capture_options = options;
        self.captures.clear();
        self.pending_captures.clear();
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
    /// Deliver retained committed residual rows and release them from the session.
    pub fn take_captured_rows(&mut self) -> Vec<CapturedRow> {
        std::mem::take(&mut self.captures)
    }
    /// Committed residual rows awaiting delivery to a draft consumer.
    pub fn captured_rows(&self) -> &[CapturedRow] {
        &self.captures
    }
    /// Residual rows from the current uncommitted candidate pass.
    pub fn provisional_captures(&self) -> &[CapturedRow] {
        &self.pending_captures
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

    /// Persistent session storage without transient projection scratch space.
    pub fn session_memory(&self) -> SessionMemory {
        SessionMemory {
            capture_bytes: mackani_ref::model::capture_allocated_bytes(
                &self.captures,
                self.captures.capacity(),
            ) + mackani_ref::model::capture_allocated_bytes(
                &self.pending_captures,
                self.pending_captures.capacity(),
            ),
            recurrent_state_bytes: self
                .states
                .iter()
                .chain(&self.working)
                .map(|state| match state {
                    LayerState::Recurrent(s) => s.allocated_bytes(),
                    LayerState::Attention(_) => 0,
                })
                .sum(),
            replay_bytes: self
                .replay
                .iter()
                .flatten()
                .map(GatedDeltaNetRow::allocated_bytes)
                .sum(),
            replay_capacity_rows: self.rows_per_pass,
            attention_row_capacities: self
                .states
                .iter()
                .filter_map(|state| match state {
                    LayerState::Attention(s) => Some(s.capacity()),
                    _ => None,
                })
                .collect(),
            pending_token_capacity: self.pending_tokens.capacity(),
            history_capacity: self.tokens.capacity(),
        }
    }

    /// Digest the committed state, including cache codes, history and logit bits.
    pub fn state_digest(&self) -> u64 {
        let mut digest = 0xcbf29ce484222325;
        for byte in (self.position as u64).to_le_bytes() {
            digest = (digest ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
        for state in &self.states {
            match state {
                LayerState::Attention(state) => {
                    digest = digest.wrapping_mul(0x100000001b3);
                    state.update_prefix_digest(&mut digest, self.position);
                }
                LayerState::Recurrent(_) => state.update_digest(&mut digest),
            }
        }
        for byte in self
            .committed_logits
            .iter()
            .flatten()
            .flat_map(|v| v.to_bits().to_le_bytes())
        {
            digest = (digest ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
        digest
    }

    fn require_no_pending_pass(&self) -> Result<()> {
        if self.pending {
            Err(Error::InvalidInput(
                "commit or discard the current provisional pass first".into(),
            ))
        } else {
            Ok(())
        }
    }

    fn validate_tokens(&self, tokens: &[u32]) -> Result<()> {
        if self
            .position
            .checked_add(tokens.len())
            .is_none_or(|end| end > self.context_capacity)
        {
            return Err(Error::InvalidInput(
                "pass exceeds the session context capacity".into(),
            ));
        }
        if tokens
            .iter()
            .any(|&t| t as usize >= self.dimensions().vocabulary_size)
        {
            return Err(Error::InvalidInput("token ID exceeds vocabulary".into()));
        }
        Ok(())
    }

    fn prepare_working(&mut self) {
        for (work, committed) in self.working.iter_mut().zip(&self.states) {
            if let (LayerState::Recurrent(work), LayerState::Recurrent(committed)) =
                (work, committed)
            {
                if self.engine.cpu_stages {
                    work.copy_history_from(committed);
                } else {
                    work.copy_from(committed);
                }
            }
        }
    }

    fn publish_recurrence(&mut self) {
        for (work, committed) in self.working.iter_mut().zip(&mut self.states) {
            if matches!(committed, LayerState::Recurrent(_)) {
                std::mem::swap(work, committed);
            }
        }
    }

    fn truncate_attention(&mut self, position: usize) {
        for state in &mut self.states {
            if let LayerState::Attention(state) = state {
                state
                    .truncate(position)
                    .expect("rollback only removes appended rows");
            }
        }
    }

    /// Scoring retains all rows; prompt prefill can instead request Last or None.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        self.prefill_with_logits(tokens, LogitRows::All)
    }

    /// Atomically commit the complete prompt, keeping only requested vocabulary heads.
    /// None clears predictions; Last computes only the final row's vocabulary head.
    pub fn prefill_with_logits(&mut self, tokens: &[u32], requested: LogitRows) -> Result<()> {
        self.require_no_pending_pass()?;
        self.validate_tokens(tokens)?;
        if tokens.is_empty() {
            return Ok(());
        }
        let copy_start = Instant::now();
        self.prepare_working();
        let copy_time = copy_start.elapsed();
        let result = (|| {
            let mut logits = Vec::new();
            let mut timings = Vec::new();
            let mut captures = Vec::new();
            for (pass, chunk) in tokens.chunks(self.rows_per_pass).enumerate() {
                let position = self.position + pass * self.rows_per_pass;
                let rows = match requested {
                    LogitRows::Last if position + chunk.len() != self.position + tokens.len() => {
                        LogitRows::None
                    }
                    rows => rows,
                };
                let (head, mut timing, captured) = self.compute_pass(
                    chunk,
                    position,
                    rows,
                    false,
                    (self.position + tokens.len()).saturating_sub(self.capture_options.max_rows),
                )?;
                if pass == 0 {
                    timing.recurrence += copy_time;
                    timing.total += copy_time;
                }
                captures.extend(captured);
                let excess = captures.len().saturating_sub(self.capture_options.max_rows);
                captures.drain(..excess);
                logits.extend(head);
                timings.push(timing);
            }
            Ok::<_, Error>((logits, timings, captures))
        })();
        match result {
            Ok((logits, timings, captures)) => {
                self.retain_captures(captures);
                self.publish_recurrence();
                self.position += tokens.len();
                self.tokens.extend_from_slice(tokens);
                self.committed_logits = logits.last().cloned();
                self.logits = logits;
                self.timings = timings;
                Ok(())
            }
            Err(error) => {
                self.truncate_attention(self.position);
                Err(error)
            }
        }
    }

    /// A single bounded causal pass. Attention is appended in place; committed
    /// recurrence stays untouched while the preallocated working state advances.
    pub fn run_provisional(&mut self, tokens: &[u32]) -> Result<&[Vec<f32>]> {
        self.require_no_pending_pass()?;
        if tokens.is_empty() || tokens.len() > self.rows_per_pass {
            return Err(Error::InvalidInput(
                "provisional width must be in 1..=rows_per_pass".into(),
            ));
        }
        self.validate_tokens(tokens)?;
        let copy_start = Instant::now();
        self.prepare_working();
        let copy_time = copy_start.elapsed();
        match self.compute_pass(tokens, self.position, LogitRows::All, true, 0) {
            Ok((logits, mut timing, captures)) => {
                timing.recurrence += copy_time;
                timing.total += copy_time;
                self.pending_captures = captures;
                self.working_complete = true;
                self.pending_tokens.clear();
                self.pending_tokens.extend_from_slice(tokens);
                self.pending = true;
                self.logits = logits;
                self.timings = vec![timing];
                Ok(&self.logits)
            }
            Err(error) => {
                self.truncate_attention(self.position);
                Err(error)
            }
        }
    }

    /// Commit exactly k input rows. Partial acceptance replays recurrence only;
    /// full acceptance simply publishes the already computed working state.
    pub fn commit(&mut self, k: usize) -> Result<()> {
        if !self.pending || k > self.pending_tokens.len() {
            return Err(Error::InvalidInput(
                "accepted rows exceed the pending pass or no pass exists".into(),
            ));
        }
        if k == 0 {
            self.discard_provisional();
            return Ok(());
        }
        if k < self.pending_tokens.len() || !self.working_complete {
            self.working_complete = false;
            self.prepare_working();
            let result = self
                .weights
                .layers
                .iter()
                .zip(&mut self.working)
                .zip(&self.replay)
                .zip(&self.states)
                .try_for_each(|(((layer, state), rows), source)| {
                    if let (Branch::Recurrent(layer), LayerState::Recurrent(state)) =
                        (&layer.branch, state)
                    {
                        let source = if self.engine.cpu_stages {
                            if let LayerState::Recurrent(source) = source {
                                Some(source)
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        replay_recurrence(
                            &mut self.engine,
                            &layer.recurrence,
                            state,
                            source,
                            &rows[..k],
                        )?;
                    }
                    Ok::<_, Error>(())
                });
            result?;
        }
        let accepted = self
            .pending_captures
            .drain(..k.min(self.pending_captures.len()))
            .collect::<Vec<_>>();
        self.retain_captures(accepted);
        self.pending_captures.clear();
        self.publish_recurrence();
        self.position += k;
        self.tokens.extend_from_slice(&self.pending_tokens[..k]);
        self.truncate_attention(self.position);
        self.logits.truncate(k);
        self.committed_logits = self.logits.last().cloned();
        self.pending_tokens.clear();
        self.pending = false;
        Ok(())
    }

    /// Discard candidate rows and expose the committed logits again.
    pub fn discard_provisional(&mut self) {
        if self.pending {
            self.truncate_attention(self.position);
            self.pending_tokens.clear();
            self.pending_captures.clear();
            self.pending = false;
            self.logits = self.committed_logits.iter().cloned().collect();
        }
    }
}

impl mackani_ref::quality::ScoringModel for Model {
    fn vocabulary_size(&self) -> usize {
        self.dimensions().vocabulary_size
    }
    fn reset(&mut self) -> Result<()> {
        self.reset()
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        self.prefill(tokens)
    }
    fn row_logits(&self) -> &[Vec<f32>] {
        self.row_logits()
    }
}

impl mackani_ref::speculation::Verifier for Model {
    type Error = Error;
    fn context_identity(&self) -> Option<Arc<()>> {
        Some(self.context_identity())
    }
    fn take_captured_rows(&mut self) -> Vec<CapturedRow> {
        self.take_captured_rows()
    }

    fn dimensions(&self) -> Option<&ModelDimensions> {
        Some(self.dimensions())
    }
    fn position(&self) -> usize {
        self.position()
    }
    fn context_capacity(&self) -> usize {
        self.context_capacity()
    }
    fn max_pass_width(&self) -> usize {
        self.rows_per_pass()
    }
    fn run_provisional(&mut self, tokens: &[u32]) -> Result<()> {
        self.run_provisional(tokens).map(|_| ())
    }
    fn row_logits(&self, row: usize) -> Option<&[f32]> {
        self.row_logits().get(row).map(Vec::as_slice)
    }
    fn logits(&self) -> Option<&[f32]> {
        self.logits()
    }
    fn commit(&mut self, rows: usize) -> Result<()> {
        self.commit(rows)
    }
    fn capture_options(&self) -> Option<&CaptureOptions> {
        Some(self.capture_options())
    }
    fn captured_rows(&self) -> &[CapturedRow] {
        self.captured_rows()
    }
    fn provisional_captures(&self) -> &[CapturedRow] {
        self.provisional_captures()
    }
    fn embedding(&self, token: u32) -> mackani_ref::Result<Vec<f32>> {
        self.embedding(token)
            .map_err(|error| mackani_ref::Error::InvalidInput(error.to_string()))
    }
    fn discard(&mut self) {
        self.discard_provisional();
    }
}

impl mackani_ref::speculation::CheckpointVerifier for Model {
    type Snapshot = Snapshot;
    fn snapshot(&self) -> Result<Snapshot> {
        self.snapshot()
    }
    fn restore(&mut self, snapshot: &Snapshot) -> Result<()> {
        self.restore(snapshot)
    }
}

#[cfg(test)]
#[allow(dead_code, clippy::manual_is_multiple_of)]
#[path = "../../mackani-ref/tests/support/mod.rs"]
pub(crate) mod session_test_support;

#[cfg(test)]
mod session_tests;
