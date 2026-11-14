use std::{fs, path::PathBuf, str::FromStr};

use mackani_engine::{Backend, CpuStages, Kernel, KvMode, MetalOptions};
use mackani_ref::speculation::{AdaptiveDepth, DepthPolicy};
use serde_json::{Value, json};

use super::sampler;
use crate::{Error, Result, engine::Sampling, memory::MemoryEstimate};

/// `--draft both` copies an n-gram continuation only when the match spans at
/// least this many tokens; shorter matches lose to the neural draft.
pub const LONG_NGRAM_MATCH: usize = 6;

pub const MODEL_FILE_STABILITY: &str = "Model files must not change while mackani is running.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftMode {
    None,
    Ngram,
    Neural,
    Both,
}
impl DraftMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Ngram => "ngram",
            Self::Neural => "neural",
            Self::Both => "both",
        }
    }
    pub(crate) fn neural(self) -> bool {
        matches!(self, Self::Neural | Self::Both)
    }
}
impl FromStr for DraftMode {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "none" => Ok(Self::None),
            "ngram" => Ok(Self::Ngram),
            "neural" => Ok(Self::Neural),
            "both" => Ok(Self::Both),
            _ => Err(Error::engine("--draft expects none|ngram|neural|both")),
        }
    }
}
/// How many guesses each pass requests: an adaptive choice up to `k`, or `k`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthMode {
    Adaptive,
    Fixed,
}
impl DepthMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Adaptive => "adaptive",
            Self::Fixed => "fixed",
        }
    }
}
impl FromStr for DepthMode {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "adaptive" => Ok(Self::Adaptive),
            "fixed" => Ok(Self::Fixed),
            _ => Err(Error::request("--depth must be adaptive or fixed")),
        }
    }
}
#[derive(Debug, Clone)]
pub struct RealEngineConfig {
    pub model_dir: PathBuf,
    pub kv: KvMode,
    /// `None`: Metal when it works on this machine, else the CPU.
    pub backend: Option<Backend>,
    /// `None`: the fastest kernel this CPU supports.
    pub kernel: Option<Kernel>,
    pub cpu_stages: CpuStages,
    pub metal: MetalOptions,
    pub draft: DraftMode,
    /// Maximum guesses per pass; every pass with `DepthMode::Fixed`.
    pub k: usize,
    pub depth: DepthMode,
    pub context: usize,
    pub threads: Option<usize>,
    pub prefill_width: usize,
}
impl Default for RealEngineConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            kv: KvMode::Kv5,
            backend: None,
            kernel: None,
            cpu_stages: CpuStages::Auto,
            metal: MetalOptions::default(),
            draft: DraftMode::Both,
            k: 7,
            depth: DepthMode::Adaptive,
            context: 32_768,
            threads: None,
            prefill_width: 16,
        }
    }
}
impl RealEngineConfig {
    /// Start-up checks: Apple silicon, macOS 26 or later, the option
    /// values and every model file this configuration reads, before anything
    /// is opened or mapped.
    pub(crate) fn validate(&self) -> Result<()> {
        crate::platform::check_this_platform()?;
        if !self.model_dir.is_dir() {
            return Err(Error::engine("--model-dir must name a model folder"));
        }
        if self.context == 0 || self.prefill_width == 0 || self.threads == Some(0) {
            return Err(Error::engine(
                "--ctx, --threads and prefill width must be positive",
            ));
        }
        if self.k > self.context || self.k.checked_add(1).is_none() {
            return Err(Error::engine("--k must not exceed --ctx"));
        }
        if self.draft.neural() && !(1..8).contains(&self.k) {
            return Err(Error::engine("neural --k must be in 1..8"));
        }
        crate::platform::check_folder(self)
    }
    pub(crate) fn kv_name(&self) -> &'static str {
        match self.kv {
            KvMode::Int8 => "int8",
            KvMode::Int8Profile => "int8-profile",
            KvMode::Kv5 => "kv5",
        }
    }
    /// The adaptive policy decides when and how deep to run the neural draft;
    /// n-gram guesses cost nothing to make and are always offered up to `k`.
    /// Its cost model is a fixed table measured on an Apple M5, so the
    /// depths never depend on wall-clock time and repeat for a given input.
    pub fn depth_policy(&self) -> DepthPolicy {
        if self.depth == DepthMode::Adaptive && self.draft.neural() {
            DepthPolicy::Adaptive(AdaptiveDepth::neural())
        } else {
            DepthPolicy::Fixed
        }
    }
    /// Start-up checks and the memory estimate from metadata only; refuses
    /// nothing for lack of memory (`mackani info`).
    pub fn inspect(&self) -> Result<ModelInfo> {
        crate::memory::inspect(self, None)
    }
    /// What every command that loads a model checks before anything large is
    /// read: the start-up checks, the metadata inspection and the memory
    /// check. `physical` is the physical memory the estimate must fit; `None`
    /// reads this Mac's.
    pub fn check(&self, physical: Option<u64>) -> Result<ModelInfo> {
        let info = crate::memory::inspect(self, physical)?;
        info.memory.enforce(info.memory.physical_bytes)?;
        Ok(info)
    }
    /// Qwen model defaults, overridden by the selected folder's generation config.
    pub fn sampling_defaults(&self) -> Result<Sampling> {
        let value = match fs::read(self.model_dir.join("generation_config.json")) {
            Ok(bytes) => serde_json::from_slice::<Value>(&bytes).map_err(Error::engine)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Value::Null,
            Err(e) => return Err(Error::engine(e)),
        };
        let number = |key: &str, default: f64| -> Result<f64> {
            match value.get(key) {
                None => Ok(default),
                Some(v) => v.as_f64().filter(|n| n.is_finite()).ok_or_else(|| {
                    Error::engine(format!("generation_config.{key} must be a finite number"))
                }),
            }
        };
        let sampling = Sampling {
            temperature: number("temperature", 1.0)?,
            top_p: number("top_p", 0.95)?,
            top_k: match value.get("top_k") {
                None => 20,
                Some(v) => v
                    .as_i64()
                    .ok_or_else(|| Error::engine("generation_config.top_k must be an integer"))?,
            },
            min_p: number("min_p", 0.0)?,
            seed: 0,
        };
        // Validate with the same fixed-point conversions as a real request.
        sampler(&sampling, 20)?;
        Ok(sampling)
    }
}
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_file: PathBuf,
    pub model_bytes: u64,
    pub code_bytes_per_pass: u64,
    /// Logical draft code read by one block proposal / one history update.
    pub draft_proposal_code_bytes: u64,
    pub draft_update_code_bytes: u64,
    pub layers: usize,
    pub hidden_width: usize,
    pub feed_forward_width: usize,
    pub vocabulary: usize,
    pub context: usize,
    pub model_max_context: usize,
    pub threads: usize,
    pub kv: String,
    pub backend: String,
    pub device: String,
    pub draft: String,
    pub k: usize,
    pub depth: String,
    pub memory: MemoryEstimate,
    /// Effective backend, kernel, CPU stages, Metal options and threads.
    pub run: crate::platform::RunChoice,
}
impl ModelInfo {
    pub fn to_json(&self) -> Value {
        let mut value = self.json_fields();
        if let (Some(value), Value::Object(run)) = (value.as_object_mut(), self.run.to_json()) {
            value.extend(run);
        }
        value
    }
    fn json_fields(&self) -> Value {
        json!({
            "model_file": self.model_file,
            "model_bytes": self.model_bytes,
            "code_bytes_per_pass": self.code_bytes_per_pass,
            "draft_proposal_code_bytes": self.draft_proposal_code_bytes,
            "draft_update_code_bytes": self.draft_update_code_bytes,
            "layers": self.layers,
            "hidden_width": self.hidden_width,
            "feed_forward_width": self.feed_forward_width,
            "vocabulary": self.vocabulary,
            "context": self.context,
            "model_max_context": self.model_max_context,
            "threads": self.threads,
            "kv": self.kv,
            "backend": self.backend,
            "device": self.device,
            "draft": self.draft,
            "k": self.k,
            "depth": self.depth,
            "memory": self.memory.to_json(),
            "file_stability": MODEL_FILE_STABILITY,
        })
    }
}
