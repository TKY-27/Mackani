//! Lossless, straight-chain speculative generation shared by both model implementations.
//!
//! The last token in `prompt_tail` is pending: the verifier has consumed every
//! earlier token, but has not consumed that last token. The final selected token
//! is likewise returned without being consumed. This convention makes stopping
//! after any token (including EOS or cancellation) identical to one-row decoding.
use std::fmt;
use std::time::{Duration, Instant};

use crate::model::{CaptureOptions, CapturedRow, ModelDimensions};
use crate::{Error, Model, Result, Sampler};

pub use crate::draft_depth::{AdaptiveDepth, CostModel, DepthController, DepthPolicy};
pub use crate::draft_proposer::{
    CombinedProposer, DraftProposer, DraftProposerSnapshot, DraftWork, ProposalOrder,
    SessionSnapshot,
};

/// A causal verifier. Failed passes are atomic; commit retains exactly its first
/// `rows` input rows, and discard restores the committed prediction.
pub trait Verifier {
    type Error;
    fn position(&self) -> usize;
    fn context_capacity(&self) -> usize;
    fn max_pass_width(&self) -> usize {
        usize::MAX
    }
    fn run_provisional(&mut self, tokens: &[u32]) -> std::result::Result<(), Self::Error>;
    fn row_logits(&self, row: usize) -> Option<&[f32]>;
    fn logits(&self) -> Option<&[f32]>;
    fn commit(&mut self, rows: usize) -> std::result::Result<(), Self::Error>;
    fn discard(&mut self);
    /// Changes on every successful main-only restore/reset. Drafts require it.
    fn context_identity(&self) -> Option<std::sync::Arc<()>> {
        None
    }
    fn take_captured_rows(&mut self) -> Vec<CapturedRow> {
        Vec::new()
    }
    fn captured_rows(&self) -> &[CapturedRow] {
        &[]
    }
    fn provisional_captures(&self) -> &[CapturedRow] {
        &[]
    }
    fn capture_options(&self) -> Option<&CaptureOptions> {
        None
    }
    fn dimensions(&self) -> Option<&ModelDimensions> {
        None
    }
    fn embedding(&self, _token: u32) -> Result<Vec<f32>> {
        Err(Error::InvalidInput(
            "verifier does not expose embeddings".into(),
        ))
    }
}

impl Verifier for Model {
    type Error = Error;
    fn context_identity(&self) -> Option<std::sync::Arc<()>> {
        Some(self.context_identity())
    }
    fn take_captured_rows(&mut self) -> Vec<CapturedRow> {
        self.take_captured_rows()
    }
    fn position(&self) -> usize {
        self.position()
    }
    fn context_capacity(&self) -> usize {
        self.context_capacity()
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
    fn discard(&mut self) {
        self.discard_provisional();
    }
    fn captured_rows(&self) -> &[CapturedRow] {
        self.captured_rows()
    }
    fn provisional_captures(&self) -> &[CapturedRow] {
        self.provisional_captures()
    }
    fn capture_options(&self) -> Option<&CaptureOptions> {
        Some(self.capture_options())
    }
    fn dimensions(&self) -> Option<&ModelDimensions> {
        Some(self.dimensions())
    }
    fn embedding(&self, token: u32) -> Result<Vec<f32>> {
        self.embedding(token)
    }
}

/// Verifiers whose owned checkpoints can be saved together with a draft ring.
pub trait CheckpointVerifier: Verifier {
    type Snapshot;
    fn snapshot(&self) -> std::result::Result<Self::Snapshot, Self::Error>;
    fn restore(&mut self, snapshot: &Self::Snapshot) -> std::result::Result<(), Self::Error>;
}
impl CheckpointVerifier for Model {
    type Snapshot = crate::Snapshot;
    fn snapshot(&self) -> Result<Self::Snapshot> {
        self.snapshot()
    }
    fn restore(&mut self, snapshot: &Self::Snapshot) -> Result<()> {
        self.restore(snapshot)
    }
}

/// Selection must depend only on logits and absolute prediction position, not
/// on pass grouping. A prediction position is its input row's position plus one.
pub trait Selector {
    fn select(&mut self, logits: &[f32], prediction_position: u64) -> Result<u32>;
}

impl Selector for Sampler {
    fn select(&mut self, logits: &[f32], prediction_position: u64) -> Result<u32> {
        self.sample(logits, prediction_position)
    }
}

/// Select the highest logit with deterministic token-ID tie breaking.
#[derive(Clone, Copy, Debug, Default)]
pub struct Greedy;
impl Selector for Greedy {
    fn select(&mut self, logits: &[f32], _prediction_position: u64) -> Result<u32> {
        crate::output_head_sampler::greedy(logits)
    }
}

/// Diagnostic attribution of the most recent proposal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProposalSource {
    #[default]
    Unspecified,
    Neural,
    Ngram,
}

/// Guesses follow `last_selected`; `committed_history` excludes that pending
/// token. History can be a known suffix of the committed session's full history.
/// Return only valid vocabulary IDs, and at most `max_guesses` tokens.
pub trait Proposer {
    fn propose(
        &mut self,
        committed_history: &[u32],
        last_selected: u32,
        max_guesses: usize,
    ) -> Vec<u32>;
    /// Source of the most recent proposal, including `free_guesses`.
    /// Diagnostic only; unspecified proposers still contribute to total offers.
    fn proposal_source(&self) -> ProposalSource {
        ProposalSource::Unspecified
    }
    /// Observe the retained prompt suffix before generation or a direct proposal.
    fn synchronize<V: Verifier>(&mut self, _verifier: &mut V) -> Result<()> {
        Ok(())
    }
    /// Called after commit with its undelivered captures.
    fn observe_committed(&mut self, _rows: &[CapturedRow]) -> Result<()> {
        Ok(())
    }
    /// Guesses that cost nothing worth counting to make (an exact n-gram copy).
    /// An adaptive depth policy offers these, up to its maximum, without
    /// consulting or updating its estimate; empty defers to the policy.
    fn free_guesses(&mut self, _history: &[u32], _last: u32, _max: usize) -> Vec<u32> {
        Vec::new()
    }
    /// Fallible, verifier-aware entry point; token-only proposers keep their API.
    fn propose_with<V: Verifier>(
        &mut self,
        verifier: &mut V,
        history: &[u32],
        last: u32,
        max: usize,
    ) -> Result<Vec<u32>> {
        self.synchronize(verifier)?;
        Ok(self.propose(history, last, max))
    }
}

/// Generate tokens without offering a speculative continuation.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoProposer;
impl Proposer for NoProposer {
    fn propose(&mut self, _history: &[u32], _last: u32, _max: usize) -> Vec<u32> {
        Vec::new()
    }
}

/// Offers the same fixed script from its beginning on each pass.
#[derive(Clone, Debug)]
pub struct ScriptProposer {
    script: Vec<u32>,
}
impl ScriptProposer {
    /// Create a proposer from its explicit continuation policy.
    pub fn new(script: Vec<u32>) -> Self {
        Self { script }
    }
}
impl Proposer for ScriptProposer {
    fn propose(&mut self, _history: &[u32], _last: u32, max: usize) -> Vec<u32> {
        self.script.iter().copied().take(max).collect()
    }
}

/// Copies the continuation of the most recent earlier occurrence of the last
/// `n` known tokens. Only already-known tokens are copied (no recursive guesses).
#[derive(Clone, Debug)]
pub struct NgramProposer {
    n: usize,
    k: usize,
}
impl NgramProposer {
    /// Create a proposer from its explicit continuation policy.
    pub fn new(n: usize, k: usize) -> Result<Self> {
        if n == 0 {
            return Err(Error::InvalidInput("ngram length must be positive".into()));
        }
        Ok(Self { n, k })
    }
}
impl NgramProposer {
    /// As `propose`, but only when the most recent match of the last `n` known
    /// tokens extends backwards to at least `min_match` equal tokens.
    pub fn propose_long(
        &self,
        history: &[u32],
        last: u32,
        max: usize,
        min_match: usize,
    ) -> Vec<u32> {
        self.continuation(history, last, max, min_match)
    }
    fn continuation(&self, history: &[u32], last: u32, max: usize, min_match: usize) -> Vec<u32> {
        let max = max.min(self.k);
        if max == 0 || history.len() < self.n {
            return Vec::new();
        }
        // Virtual concatenation avoids allocating a copy of the growing history.
        let known_len = history.len() + 1;
        let token = |index: usize| {
            if index == history.len() {
                last
            } else {
                history[index]
            }
        };
        let suffix_start = known_len - self.n;
        for start in (0..suffix_start).rev() {
            if (0..self.n).all(|offset| token(start + offset) == token(suffix_start + offset)) {
                let earlier = (1..=start)
                    .take_while(|&i| token(start - i) == token(suffix_start - i))
                    .take(min_match.saturating_sub(self.n))
                    .count();
                if self.n + earlier < min_match {
                    return Vec::new();
                }
                let continuation = start + self.n;
                return (continuation..known_len).take(max).map(token).collect();
            }
        }
        Vec::new()
    }
}
impl Proposer for NgramProposer {
    fn proposal_source(&self) -> ProposalSource {
        ProposalSource::Ngram
    }
    fn propose(&mut self, history: &[u32], last: u32, max: usize) -> Vec<u32> {
        self.continuation(history, last, max, 0)
    }
}

/// Output, context and proposal limits for one generation call.
pub struct GenerationLimits<'a> {
    pub max_new_tokens: usize,
    /// Fixed depth, or the maximum depth of an adaptive policy.
    pub max_draft_tokens: usize,
    /// Changes only how many guesses are requested, never the output.
    pub depth: DepthPolicy,
    /// An additional caller limit, clamped to the verifier's own capacity.
    pub context_capacity: usize,
    pub eos_ids: &'a [u32],
    /// Called for every returned token, including EOS. Return false to cancel.
    pub on_token: Option<&'a mut dyn FnMut(u32) -> bool>,
}
impl GenerationLimits<'_> {
    /// Set output and context limits with the default proposal policy.
    pub fn new(max_new_tokens: usize, context_capacity: usize) -> Self {
        Self {
            max_new_tokens,
            max_draft_tokens: 0,
            depth: DepthPolicy::Fixed,
            context_capacity,
            eos_ids: &[],
            on_token: None,
        }
    }
}

/// The condition that ended a generation call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    MaxNewTokens,
    ContextCapacity,
    EndOfSequence,
    Cancelled,
}

/// Proposal acceptance and pass timing measured during generation.
#[derive(Clone, Debug, Default)]
pub struct GenerationStatistics {
    pub passes: usize,
    pub rows_verified: usize,
    pub tokens_confirmed_per_pass: Vec<usize>,
    pub guesses_offered: usize,
    /// Actual verified guesses after depth truncation, attributed by the proposer.
    pub neural_guesses_offered: usize,
    pub ngram_guesses_offered: usize,
    /// Proposer calls with a positive guess budget.
    pub proposals: usize,
    /// Guesses requested before each pass; zero when no proposal was made.
    pub requested_depths: Vec<usize>,
    /// Guesses equal to selected tokens, including a matching terminal token.
    pub guesses_accepted: usize,
    /// End-to-end pass time, including selection, streaming callback and commit.
    pub pass_durations: Vec<Duration>,
}

/// Generated tokens, stopping condition and acceptance statistics.
#[derive(Clone, Debug)]
pub struct GenerationResult {
    pub tokens: Vec<u32>,
    pub statistics: GenerationStatistics,
    pub stop_reason: StopReason,
}

/// A verifier, selector or proposal failure encountered during generation.
#[derive(Debug)]
pub enum GenerationError<E> {
    Verifier(E),
    Selector(Error),
    Proposer(Error),
    InvalidInput(&'static str),
}
impl<E: fmt::Display> fmt::Display for GenerationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Verifier(error) => write!(f, "verifier: {error}"),
            Self::Selector(error) => write!(f, "selector: {error}"),
            Self::Proposer(error) => write!(f, "proposer: {error}"),
            Self::InvalidInput(message) => write!(f, "generation: {message}"),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for GenerationError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Verifier(error) => Some(error),
            Self::Selector(error) | Self::Proposer(error) => Some(error),
            Self::InvalidInput(_) => None,
        }
    }
}

/// Generate from a known history tail ending in one unconsumed input token.
///
/// The caller prefills the prompt except its final token before calling. A pass
/// verifies `[pending, guesses..]`, selecting each output from the preceding row.
/// Only selected matches allow the next row to be read. Commit consumes one row
/// per returned output, leaving the final output pending even at a stop boundary.
pub fn generate<V: Verifier, P: Proposer, S: Selector>(
    verifier: &mut V,
    proposer: &mut P,
    selector: &mut S,
    prompt_tail: &[u32],
    mut limits: GenerationLimits<'_>,
) -> std::result::Result<GenerationResult, GenerationError<V::Error>> {
    let (&initial_pending, history) =
        prompt_tail
            .split_last()
            .ok_or(GenerationError::InvalidInput(
                "prompt tail must not be empty",
            ))?;
    let mut pending = initial_pending;
    if history.len() > verifier.position() {
        return Err(GenerationError::InvalidInput(
            "prompt history exceeds committed position",
        ));
    }
    if verifier.max_pass_width() == 0 {
        return Err(GenerationError::InvalidInput(
            "maximum pass width must be positive",
        ));
    }
    let mut history = history.to_vec();
    let capacity = limits.context_capacity.min(verifier.context_capacity());
    let mut result = GenerationResult {
        tokens: Vec::new(),
        statistics: GenerationStatistics::default(),
        stop_reason: StopReason::MaxNewTokens,
    };
    let mut controller = match &limits.depth {
        DepthPolicy::Fixed => None,
        DepthPolicy::Adaptive(config) => {
            config.validate().map_err(GenerationError::InvalidInput)?;
            let budget = limits.max_draft_tokens.min(
                verifier
                    .max_pass_width()
                    .min(capacity.saturating_sub(verifier.position()))
                    .min(limits.max_new_tokens)
                    .saturating_sub(1),
            );
            Some(DepthController::new(config, budget))
        }
    };
    proposer
        .synchronize(verifier)
        .map_err(GenerationError::Proposer)?;
    let mut drafting = true;
    while result.tokens.len() < limits.max_new_tokens {
        let position = verifier.position();
        let room = capacity.saturating_sub(position);
        if room == 0 {
            result.stop_reason = StopReason::ContextCapacity;
            break;
        }
        let remaining = limits.max_new_tokens - result.tokens.len();
        let width = room.min(remaining).min(verifier.max_pass_width());
        let limit = limits.max_draft_tokens.min(width - 1);
        let free = if drafting && controller.is_some() {
            proposer.free_guesses(&history, pending, limit)
        } else {
            Vec::new()
        };
        // The proposer is asked for its whole budget; an adaptive depth only
        // truncates the offer, and zero skips the call (and its cost).
        let depth = match &mut controller {
            Some(controller) if drafting && free.is_empty() => controller.depth(limit),
            _ => limit,
        };
        let call = drafting && free.is_empty() && (controller.is_none() || depth > 0);
        result.statistics.proposals += usize::from(call && limit > 0);
        let mut guesses = if !free.is_empty() {
            free
        } else if call {
            match proposer.propose_with(verifier, &history, pending, limit) {
                Ok(guesses) => guesses,
                Err(error) if result.tokens.is_empty() => {
                    return Err(GenerationError::Proposer(error));
                }
                Err(_) => {
                    drafting = false;
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };
        let vocabulary = verifier
            .dimensions()
            .map(|dimensions| dimensions.vocabulary_size)
            .or_else(|| verifier.logits().map(<[f32]>::len));
        let invalid_guesses = if guesses.len() > limit {
            Some("proposer exceeded the requested guess limit")
        } else if vocabulary.is_some_and(|size| guesses.iter().any(|&id| id as usize >= size)) {
            Some("proposer returned a token outside the vocabulary")
        } else {
            None
        };
        if let Some(message) = invalid_guesses {
            if result.tokens.is_empty() {
                return Err(GenerationError::InvalidInput(message));
            }
            drafting = false;
            guesses.clear();
        }
        if let Some(controller) = &mut controller {
            if call && drafting {
                controller.proposed(result.tokens.len(), &guesses);
            }
            guesses.truncate(depth);
        }
        result
            .statistics
            .requested_depths
            .push(if call { depth } else { guesses.len() });
        let mut rows = Vec::with_capacity(guesses.len() + 1);
        rows.push(pending);
        rows.extend_from_slice(&guesses);
        let started = Instant::now();
        verifier
            .run_provisional(&rows)
            .map_err(GenerationError::Verifier)?;
        result.statistics.passes += 1;
        result.statistics.rows_verified += rows.len();
        result.statistics.guesses_offered += guesses.len();
        match proposer.proposal_source() {
            ProposalSource::Neural => result.statistics.neural_guesses_offered += guesses.len(),
            ProposalSource::Ngram => result.statistics.ngram_guesses_offered += guesses.len(),
            ProposalSource::Unspecified => {}
        }
        let before = result.tokens.len();
        let mut stopped = false;
        for row in 0..rows.len() {
            let Some(logits) = verifier.row_logits(row) else {
                verifier.discard();
                return Err(GenerationError::InvalidInput(
                    "verifier omitted a requested logit row",
                ));
            };
            let prediction_position = u64::try_from(position)
                .ok()
                .and_then(|value| value.checked_add(row as u64 + 1));
            let Some(prediction_position) = prediction_position else {
                verifier.discard();
                return Err(GenerationError::InvalidInput(
                    "prediction position overflow",
                ));
            };
            let selected = match selector.select(logits, prediction_position) {
                Ok(token) => token,
                Err(error) => {
                    verifier.discard();
                    return Err(GenerationError::Selector(error));
                }
            };
            result.tokens.push(selected);
            let matched = guesses.get(row).is_some_and(|&guess| guess == selected);
            result.statistics.guesses_accepted += usize::from(matched);
            let keep_going = limits
                .on_token
                .as_mut()
                .is_none_or(|callback| callback(selected));
            if limits.eos_ids.contains(&selected) {
                result.stop_reason = StopReason::EndOfSequence;
                stopped = true;
            } else if !keep_going {
                result.stop_reason = StopReason::Cancelled;
                stopped = true;
            }
            if stopped || !matched {
                break;
            }
        }
        let confirmed = result.tokens.len() - before;
        let captures = verifier.provisional_captures();
        if !captures.is_empty() && captures.len() < confirmed {
            verifier.discard();
            return Err(GenerationError::InvalidInput(
                "verifier omitted captured rows",
            ));
        }
        if let Err(error) = verifier.commit(confirmed) {
            verifier.discard();
            return Err(GenerationError::Verifier(error));
        }
        let committed_captures = verifier.take_captured_rows();
        if drafting && proposer.observe_committed(&committed_captures).is_err() {
            // A committed output cannot be rolled back or lost to a draft failure.
            drafting = false;
        }
        result.statistics.tokens_confirmed_per_pass.push(confirmed);
        result.statistics.pass_durations.push(started.elapsed());
        history.push(pending);
        let outputs = &result.tokens[before..];
        history.extend_from_slice(&outputs[..outputs.len() - 1]);
        pending = outputs[outputs.len() - 1];
        if let Some(controller) = &mut controller {
            controller.observe(&result.tokens);
        }
        if stopped {
            break;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests;
