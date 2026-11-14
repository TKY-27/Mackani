//! Product session over the mapped main model and shared lossless generator.
//! The caller must keep every model file unchanged for the engine's lifetime.
use std::{fs, time::Instant};

mod config;
mod proposer;

pub use config::{
    DepthMode, DraftMode, LONG_NGRAM_MATCH, MODEL_FILE_STABILITY, ModelInfo, RealEngineConfig,
};
use proposer::{Drafts, Saved};

pub use crate::memory::MemoryEstimate;
use crate::{
    Error, Result,
    engine::{CONTEXT_MARGIN, Engine, Finish, Generation, Sampling},
};
use mackani_engine::{Backend, FastDraft, LogitRows, Model, SessionMemory};
use mackani_ref::{
    Sampler, SamplingParameters, SamplingVersion,
    draft::DraftOptions,
    draft_head::DraftChoice,
    speculation::{
        self, CombinedProposer, DraftProposer, GenerationLimits, NgramProposer, NoProposer,
        ProposalOrder, ProposalSource, Proposer, StopReason,
    },
};

/// Python round() is ties-to-even; the C ABI then narrows to signed 32 bits.
/// engine.py:166-176; engine35.cpp:944-959. Version 3 is the upstream default.
pub fn sampler(sampling: &Sampling, vocabulary: usize) -> Result<Sampler> {
    let s = sampling.normalized();
    if !s.temperature.is_finite() || !s.top_p.is_finite() || !s.min_p.is_finite() {
        return Err(Error::request("sampling parameters must be finite"));
    }
    let fixed = |n: f64| -> Result<i32> {
        let v = (n * 1000.0).round_ties_even();
        if v < i32::MIN as f64 || v > i32::MAX as f64 {
            return Err(Error::request(
                "sampling parameter exceeds native fixed-point range",
            ));
        }
        Ok(v as i32)
    };
    let temp = fixed(s.temperature)?;
    let top_p = fixed(s.top_p)?;
    let min_p = fixed(s.min_p)?;
    if !(1..=1000).contains(&top_p) || !(0..=1000).contains(&min_p) {
        return Err(Error::request(
            "rounded top_p must be in 0.001..=1 and min_p in 0..=1",
        ));
    }
    if temp <= 0 {
        return Ok(Sampler::Greedy);
    }
    let mut p = SamplingParameters::from_thousandths(temp as u32, top_p as u32, min_p as u32)
        .map_err(Error::engine)?;
    p.top_k = (s.top_k as usize).min(vocabulary);
    p.seed = u64::from(s.seed as u32);
    p.version = SamplingVersion::Gumbel;
    Ok(Sampler::Sampled(p))
}

/// This type is intentionally not Send on macOS; construct it on the server's
/// existing engine thread, never before spawning that thread.
pub struct RealEngine {
    model: Model,
    proposer: Drafts,
    config: RealEngineConfig,
    info: ModelInfo,
    defaults: Sampling,
    history: Vec<u32>,
    saved: Option<Saved>,
}
impl RealEngine {
    pub fn load(config: RealEngineConfig) -> Result<Self> {
        Self::load_with_draft_options(config, DraftOptions::default())
    }
    pub fn load_with_draft_options(
        config: RealEngineConfig,
        options: DraftOptions,
    ) -> Result<Self> {
        let info = config.check(None)?;
        Self::load_checked(config, info, options)
    }
    /// Loads the weights after [`RealEngineConfig::check`] returned `info`
    /// for this `config`.
    pub fn load_checked(
        config: RealEngineConfig,
        mut info: ModelInfo,
        options: DraftOptions,
    ) -> Result<Self> {
        // Fail closed also for an `info` from `inspect`, which refuses nothing.
        info.memory.enforce(info.memory.physical_bytes)?;
        let defaults = config.sampling_defaults()?;
        let (backend, kernel, threads) = (info.run.backend, info.run.kernel, info.run.threads);
        let pool = mackani_engine::Engine::new(kernel, threads).map_err(Error::engine)?;
        let width = config.prefill_width.max(
            config
                .k
                .checked_add(1)
                .ok_or_else(|| Error::engine("pass width overflow"))?,
        );
        // The fast draft gets its own pool with the main model's kernel,
        // worker count and projection backend; the two never run concurrently.
        let draft_pool = if config.draft.neural() {
            Some(mackani_engine::Engine::new(kernel, threads).map_err(Error::engine)?)
        } else {
            None
        };
        let (mut model, mut draft) = map_files(&config, pool, draft_pool, backend, width, options)?;
        if let Some(draft) = &mut draft {
            // Both projectors must run what `info` reports for the model.
            if draft.metal_options() != model.metal_options() {
                return Err(Error::engine(
                    "the draft's Metal options differ from the main model's",
                ));
            }
        }
        // `info` shows what the engine reports it uses, not what was requested.
        info.run.read_back(&model);
        info.threads = info.run.threads;
        let mut proposer = match (config.draft, draft) {
            (DraftMode::None, _) => Drafts::None(NoProposer),
            (DraftMode::Ngram, _) => {
                Drafts::Ngram(NgramProposer::new(3, config.k).map_err(Error::engine)?)
            }
            (DraftMode::Neural | DraftMode::Both, Some(draft)) => {
                // A caller may select a smaller shortlist/capture set for a
                // compatible model. Report the actual configured head reads.
                let (_, head_path) =
                    mackani_ref::draft::model_paths(&config.model_dir).map_err(Error::engine)?;
                let layout = mackani_ref::draft_head::HeadLayout::read(
                    &mut fs::File::open(head_path).map_err(Error::engine)?,
                )
                .map_err(Error::engine)?;
                let default_head =
                    crate::memory::reduced_head_code_reads(&layout, 512.min(layout.vocabulary))?;
                let actual_head =
                    crate::memory::reduced_head_code_reads(&layout, draft.options().shortlist)?;
                info.draft_proposal_code_bytes = info
                    .draft_proposal_code_bytes
                    .checked_sub(default_head)
                    .and_then(|n| n.checked_add(actual_head))
                    .ok_or_else(|| Error::engine("draft code byte accounting overflow"))?;
                let neural = DraftProposer::new(draft, config.k, DraftChoice::Greedy)
                    .map_err(Error::engine)?;
                model
                    .set_capture(neural.capture_options())
                    .map_err(Error::engine)?;
                if config.draft == DraftMode::Both {
                    Drafts::Both(
                        Box::new(CombinedProposer::new(
                            NgramProposer::new(3, config.k).map_err(Error::engine)?,
                            neural,
                            ProposalOrder::LongNgramFirst(LONG_NGRAM_MATCH),
                        )),
                        ProposalSource::Unspecified,
                    )
                } else {
                    Drafts::Neural(Box::new(neural))
                }
            }
            (DraftMode::Neural | DraftMode::Both, None) => {
                return Err(Error::engine("neural draft was not loaded"));
            }
        };
        proposer.synchronize(&mut model).map_err(Error::engine)?;
        info.backend = model.backend().name().into();
        info.device = model.device();
        Ok(Self {
            model,
            proposer,
            config,
            info,
            defaults,
            history: Vec::new(),
            saved: None,
        })
    }
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }
    pub fn default_sampling(&self) -> &Sampling {
        &self.defaults
    }
    pub fn session_memory(&self) -> SessionMemory {
        self.model.session_memory()
    }
    /// At most one checkpoint is retained, independent of request count.
    pub fn snapshot_bytes(&self) -> u64 {
        if self.saved.is_some() {
            self.info.memory.snapshot_bytes
        } else {
            0
        }
    }
    fn empty_generation(&self, cached_tokens: usize, snap_mode: u8, start: Instant) -> Generation {
        Generation {
            ids: vec![],
            finish: Finish::Cancelled,
            cached_tokens,
            snap_mode,
            passes: 0,
            rows_verified: 0,
            guesses_offered: 0,
            neural_guesses_offered: 0,
            ngram_guesses_offered: 0,
            guesses_accepted: 0,
            draft_proposals: 0,
            draft_updates: 0,
            prefill_ms: start.elapsed().as_secs_f64() * 1000.0,
            gen_ms: 0.0,
        }
    }
}
/// The engine and tokenizer of `chat`, `serve` and `selfcheck`, in start-up
/// order: [`RealEngineConfig::check`] (metadata only), then the tokenizer and
/// vocabulary, then the weights. `physical` as in `check`.
pub fn load_with_tokenizer(
    config: RealEngineConfig,
    physical: Option<u64>,
) -> Result<(RealEngine, mackani_text::ChatFormatter)> {
    let info = config.check(physical)?;
    let formatter = mackani_text::ChatFormatter::load(&config.model_dir)?;
    let engine = RealEngine::load_checked(config, info, DraftOptions::default())?;
    Ok((engine, formatter))
}

/// The single mapping boundary in this crate: main weights and the fast draft.
#[allow(unsafe_code)]
fn map_files(
    config: &RealEngineConfig,
    pool: mackani_engine::Engine,
    draft_pool: Option<mackani_engine::Engine>,
    backend: Backend,
    width: usize,
    options: DraftOptions,
) -> Result<(Model, Option<FastDraft>)> {
    let execution = mackani_engine::ExecutionOptions {
        backend,
        cpu_stages: config.cpu_stages,
        metal: config.metal,
    };
    // SAFETY: the product's documented file-stability contract requires all
    // model files to remain unmodified and untruncated until this engine drops.
    let model = unsafe {
        Model::with_execution_options(&config.model_dir, pool, execution, width, config.kv)
    }
    .map_err(Error::engine)?;
    let draft = draft_pool
        .map(|pool| {
            // SAFETY: as above, for both draft files.
            unsafe {
                FastDraft::with_execution_options(&config.model_dir, pool, execution, options)
            }
            .map_err(Error::engine)
        })
        .transpose()?;
    Ok((model, draft))
}
fn strict_prefix(prefix: &[u32], tokens: &[u32]) -> bool {
    !prefix.is_empty() && prefix.len() < tokens.len() && tokens.starts_with(prefix)
}
impl Engine for RealEngine {
    fn context_capacity(&self) -> usize {
        self.config.context
    }
    fn reset(&mut self) -> Result<()> {
        self.saved = None;
        self.history.clear();
        self.model.reset().map_err(Error::engine)?;
        self.proposer.reset();
        Ok(())
    }
    fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        sampling: &Sampling,
        stop_ids: Option<&[u32]>,
        on_tokens: &mut dyn FnMut(&[u32]) -> Result<bool>,
    ) -> Result<Generation> {
        let room = self
            .config
            .context
            .checked_sub(prompt.len())
            .and_then(|n| n.checked_sub(CONTEXT_MARGIN))
            .filter(|&n| n > 0);
        if prompt.is_empty() || room.is_none() {
            return Err(Error::request("the prompt leaves no room in the context"));
        }
        if prompt
            .iter()
            .any(|&id| id as usize >= self.model.dimensions().vocabulary_size)
        {
            return Err(Error::request("prompt token exceeds model vocabulary"));
        }
        let mut selector = sampler(sampling, self.model.dimensions().vocabulary_size)?;
        let choice = match &selector {
            Sampler::Greedy => DraftChoice::Greedy,
            Sampler::Sampled(p) => DraftChoice::sampled(p.seed),
        };
        self.proposer.choice(choice);
        let begin = Instant::now();
        if !on_tokens(&[])? {
            return Ok(self.empty_generation(0, 0, begin));
        }
        let (cached_tokens, snap_mode) = if strict_prefix(&self.history, prompt) {
            (self.history.len(), 1)
        } else if self
            .saved
            .as_ref()
            .is_some_and(|s| strict_prefix(&s.tokens, prompt))
        {
            let saved = self.saved.as_ref().expect("checked saved checkpoint");
            self.proposer.restore(&mut self.model, &saved.checkpoint)?;
            self.history.clone_from(&saved.tokens);
            (saved.tokens.len(), 2)
        } else {
            self.reset()?;
            (0, 0)
        };
        // src/specx.cpp:103-108: last marker in the newly processed suffix;
        // absent marker falls back to n-1, a final marker gives no new snapshot.
        let cut = prompt[cached_tokens..]
            .iter()
            .rposition(|&id| id == 248045)
            .map_or(prompt.len() - 1, |p| cached_tokens + p + 1);
        let cut = (cut > cached_tokens && cut < prompt.len()).then_some(cut);
        let end = prompt.len() - 1; // Shared loop consumes the final prompt row.
        let mut position = cached_tokens;
        while position < end {
            let mut next = (position + self.config.prefill_width.min(end - position)).min(end);
            if let Some(cut) = cut {
                if position < cut && cut < next {
                    next = cut;
                }
            }
            let logits = if next == end {
                LogitRows::Last
            } else {
                LogitRows::None
            };
            self.model
                .prefill_with_logits(&prompt[position..next], logits)
                .map_err(Error::engine)?;
            self.history.extend_from_slice(&prompt[position..next]);
            self.proposer
                .synchronize(&mut self.model)
                .map_err(Error::engine)?;
            position = next;
            if cut == Some(position) {
                // Drop the previous slot before allocating its replacement.
                self.saved = None;
                let checkpoint = self.proposer.snapshot(&mut self.model)?;
                self.saved = Some(Saved {
                    tokens: prompt[..position].to_vec(),
                    checkpoint,
                });
            }
            if !on_tokens(&[])? {
                return Ok(self.empty_generation(cached_tokens, snap_mode, begin));
            }
        }
        let prefill_ms = begin.elapsed().as_secs_f64() * 1000.0;
        let generation_start = Instant::now();
        let mut emitted = Vec::new();
        let mut callback_error = None;
        let mut callback = |token| {
            emitted.push(token);
            match on_tokens(&emitted) {
                Ok(go) => go,
                Err(error) => {
                    callback_error = Some(error);
                    false
                }
            }
        };
        let mut limits = GenerationLimits::new(
            max_new.max(1).min(room.expect("checked room") + 1),
            self.config.context - CONTEXT_MARGIN + 1,
        );
        limits.max_draft_tokens = if self.config.draft == DraftMode::None {
            0
        } else {
            self.config.k
        };
        limits.depth = self.config.depth_policy();
        limits.eos_ids = stop_ids.unwrap_or(&[]);
        limits.on_token = Some(&mut callback);
        let work = self.proposer.work();
        let result = speculation::generate(
            &mut self.model,
            &mut self.proposer,
            &mut selector,
            prompt,
            limits,
        );
        let gen_ms = generation_start.elapsed().as_secs_f64() * 1000.0;
        let after = self.proposer.work();
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                self.reset()?;
                return Err(Error::engine(error));
            }
        };
        self.history.clear();
        self.history.extend_from_slice(prompt);
        self.history
            .extend_from_slice(&result.tokens[..result.tokens.len().saturating_sub(1)]);
        debug_assert_eq!(self.history.len(), self.model.position());
        if let Some(error) = callback_error {
            return Err(error);
        }
        let finish = match result.stop_reason {
            StopReason::EndOfSequence => Finish::Stop,
            StopReason::Cancelled => Finish::Cancelled,
            _ => Finish::Length,
        };
        Ok(Generation {
            ids: result.tokens,
            finish,
            cached_tokens,
            snap_mode,
            passes: result.statistics.passes,
            rows_verified: result.statistics.rows_verified,
            guesses_offered: result.statistics.guesses_offered,
            neural_guesses_offered: result.statistics.neural_guesses_offered,
            ngram_guesses_offered: result.statistics.ngram_guesses_offered,
            guesses_accepted: result.statistics.guesses_accepted,
            draft_proposals: after.proposals - work.proposals,
            draft_updates: after.updates - work.updates,
            prefill_ms,
            gen_ms,
        })
    }
}
