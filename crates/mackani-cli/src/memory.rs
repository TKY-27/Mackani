//! Metadata-only start-up accounting; never maps or reads a weight payload.
mod inspect;
mod physical;

pub(crate) use inspect::{inspect, reduced_head_code_reads};
pub use physical::physical_memory;

use serde_json::{Value, json};

use crate::{Error, Result};

const GIB: f64 = 1_073_741_824.0;
/// The smallest context the option list of `enforce`'s message prices.
const SMALLEST_SUGGESTED_CONTEXT: usize = 4096;
/// The smallest `--ctx` the product accepts; the largest context that fits is
/// searched down to it.
pub const SMALLEST_CONTEXT: usize = 66;

/// Header-derived sizes behind the start-up estimate. They do not depend
/// on the context length, pass width or physical memory, so one set of facts
/// prices every `--ctx` and, with or without `draft`, `--draft none`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryFacts {
    pub weights_bytes: u64,
    /// One committed recurrent state, including convolution history.
    pub recurrent_bytes: u64,
    pub replay_bytes_per_row: u64,
    /// KV cache bytes per token in the chosen mode, and in `int8`.
    pub cache_bytes_per_token: u64,
    pub int8_cache_bytes_per_token: u64,
    /// All dense vectors expanded to f32.
    pub dense_bytes: u64,
    pub hidden: u64,
    pub feed_forward: u64,
    pub vocabulary: u64,
    /// The fast neural draft (`--draft neural|both`), if the configuration runs it.
    pub draft: Option<DraftFacts>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftFacts {
    /// Mapped draft files plus the fast draft's owned head and vectors.
    pub bytes: u64,
    /// The ring of past key/value rows.
    pub ring_bytes: u64,
    /// Captured main-model features per row.
    pub features: u64,
}

#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    pub weights_bytes: u64,
    pub draft_bytes: u64,
    /// One committed recurrent state, including convolution history.
    pub recurrent_bytes: u64,
    pub cache_bytes: u64,
    /// One recurrent checkpoint, prefix IDs, logits and optional draft ring.
    pub snapshot_bytes: u64,
    /// Working recurrence, fixed replay, draft ring/captures and operator scratch.
    pub working_bytes: u64,
    /// 15% of accounted allocations plus 1 GiB for the process, allocator and OS.
    pub margin_bytes: u64,
    pub needed_bytes: u64,
    pub physical_bytes: u64,
    /// What this estimate was computed from, to price the alternatives.
    pub facts: MemoryFacts,
    pub context: usize,
    pub width: usize,
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB
}

impl MemoryFacts {
    /// The memory formula for `context` tokens and pass width `width` (max of the
    /// prefill width and K + 1):
    ///
    /// ```text
    /// subtotal = weights + draft + recurrent + cache + snapshot + working
    /// cache    = cache_per_token * next_power_of_two(context)
    /// snapshot = recurrent + 4*context + 4*vocabulary + draft_ring
    /// working  = recurrent + width*replay_row + dense + 256*width*(hidden+ffn)
    ///          + draft_ring + 8*width*features + 8*context + 4*width*vocabulary
    /// needed   = subtotal + ceil(0.15 * subtotal) + 1 GiB
    /// ```
    pub fn estimate(&self, context: usize, width: usize, physical: u64) -> Result<MemoryEstimate> {
        let (draft_bytes, ring, features) = self
            .draft
            .as_ref()
            .map_or((0, 0, 0), |d| (d.bytes, d.ring_bytes, d.features));
        let width_u = width as u64;
        // Cache Vecs grow geometrically. Account for their largest capacity, not
        // merely configured length, at non-power-of-two context limits.
        let cache_capacity = context.checked_next_power_of_two().ok_or_else(overflow)? as u64;
        let cache = mul(self.cache_bytes_per_token, cache_capacity)?;
        let captures = mul(4, mul(width_u, mul(features, 2)?)?)?;
        let checkpoint = add(
            add(self.recurrent_bytes, mul(context as u64, 4)?)?,
            add(mul(self.vocabulary, 4)?, ring)?,
        )?;
        // Conservative scratch: 64 f32 hidden/FFN vectors per row, plus all row
        // logits, dense loaded vectors, token history, replay and the working state.
        let scratch = mul(256, mul(width_u, add(self.hidden, self.feed_forward)?)?)?;
        let working = add(
            add(
                add(
                    self.recurrent_bytes,
                    mul(width_u, self.replay_bytes_per_row)?,
                )?,
                add(self.dense_bytes, scratch)?,
            )?,
            add(
                add(ring, captures)?,
                add(
                    mul(context as u64, 8)?,
                    mul(width_u, mul(self.vocabulary, 4)?)?,
                )?,
            )?,
        )?;
        let subtotal = [
            self.weights_bytes,
            draft_bytes,
            self.recurrent_bytes,
            cache,
            checkpoint,
            working,
        ]
        .into_iter()
        .try_fold(0, add)?;
        let margin = add(mul(subtotal, 15)?.div_ceil(100), 1_073_741_824)?;
        Ok(MemoryEstimate {
            weights_bytes: self.weights_bytes,
            draft_bytes,
            recurrent_bytes: self.recurrent_bytes,
            cache_bytes: cache,
            snapshot_bytes: checkpoint,
            working_bytes: working,
            margin_bytes: margin,
            needed_bytes: add(subtotal, margin)?,
            physical_bytes: physical,
            facts: self.clone(),
            context,
            width,
        })
    }
}

impl MemoryEstimate {
    pub fn fits(&self, physical_bytes: u64) -> bool {
        self.needed_bytes <= physical_bytes
    }

    /// Fails closed when the estimate exceeds `physical_bytes`, naming what the
    /// configuration needs, what each part costs, and what each option that
    /// reduces it saves.
    pub fn enforce(&self, physical_bytes: u64) -> Result<()> {
        if self.fits(physical_bytes) {
            return Ok(());
        }
        Err(Error::engine(self.shortfall(physical_bytes)?))
    }

    fn shortfall(&self, physical: u64) -> Result<String> {
        let mut text = format!(
            "this configuration needs {} bytes ({:.2} GiB), including {} bytes margin; physical memory is {} bytes ({:.2} GiB), {} bytes short.\nneeded: model weights {}, draft {}, KV cache for --ctx {} {}, recurrent state {}, prompt checkpoint {}, working memory {}, margin {}.\noptions that lower the need (bytes saved, bytes then needed):",
            self.needed_bytes,
            gib(self.needed_bytes),
            self.margin_bytes,
            physical,
            gib(physical),
            self.needed_bytes.saturating_sub(physical),
            self.weights_bytes,
            self.draft_bytes,
            self.context,
            self.cache_bytes,
            self.recurrent_bytes,
            self.snapshot_bytes,
            self.working_bytes,
            self.margin_bytes,
        );
        let mut option = |flag: String, other: &MemoryEstimate| {
            text += &format!(
                "\n  {flag}: saves {} bytes, needs {} bytes ({:.2} GiB){}",
                self.needed_bytes.saturating_sub(other.needed_bytes),
                other.needed_bytes,
                gib(other.needed_bytes),
                if other.fits(physical) { ", fits" } else { "" },
            );
        };
        let without_draft = MemoryFacts {
            draft: None,
            ..self.facts.clone()
        };
        if self.facts.draft.is_some() {
            option(
                "--draft none".into(),
                &without_draft.estimate(self.context, self.width, physical)?,
            );
        }
        for context in smaller_contexts(self.context, SMALLEST_SUGGESTED_CONTEXT) {
            option(
                format!("--ctx {context}"),
                &self.facts.estimate(context, self.width, physical)?,
            );
        }
        let int8 = MemoryFacts {
            cache_bytes_per_token: self.facts.int8_cache_bytes_per_token,
            ..self.facts.clone()
        };
        if int8 != self.facts {
            option(
                "--kv int8".into(),
                &int8.estimate(self.context, self.width, physical)?,
            );
        }
        // The largest context that fits for each distinct draft/cache choice.
        let least_facts = MemoryFacts {
            draft: None,
            ..int8.clone()
        };
        let mut variants: Vec<(&MemoryFacts, &str)> = Vec::new();
        for (facts, flag) in [
            (&self.facts, ""),
            (&without_draft, " --draft none"),
            (&int8, " --kv int8"),
            (&least_facts, " --draft none --kv int8"),
        ] {
            if variants.iter().all(|(seen, _)| *seen != facts) {
                variants.push((facts, flag));
            }
        }
        let mut fitting = Vec::new();
        for (facts, flag) in variants {
            // The context itself, smaller powers of two, the smallest accepted.
            let candidates = std::iter::once(self.context)
                .chain(smaller_contexts(self.context, SMALLEST_CONTEXT))
                .chain((SMALLEST_CONTEXT < self.context).then_some(SMALLEST_CONTEXT));
            for context in candidates {
                if facts
                    .estimate(context, self.width, physical)?
                    .fits(physical)
                {
                    fitting.push(format!("--ctx {context}{flag}"));
                    break;
                }
            }
        }
        if fitting.is_empty() {
            let least =
                least_facts.estimate(SMALLEST_CONTEXT.min(self.context), self.width, physical)?;
            text += &format!(
                "\nnothing fits down to the smallest context mackani accepts: even --ctx {} --draft none --kv int8 needs {} bytes ({:.2} GiB), so this Mac does not have enough memory for this model.",
                least.context,
                least.needed_bytes,
                gib(least.needed_bytes)
            );
        } else {
            text += &format!("\nlargest that fits: {}", fitting.join("; "));
        }
        Ok(text)
    }

    pub fn to_json(&self) -> Value {
        let f = &self.facts;
        json!({
            "weights_bytes": self.weights_bytes,
            "draft_bytes": self.draft_bytes,
            "recurrent_bytes": self.recurrent_bytes,
            "cache_bytes": self.cache_bytes,
            "snapshot_bytes": self.snapshot_bytes,
            "working_bytes": self.working_bytes,
            "margin_bytes": self.margin_bytes,
            "needed_bytes": self.needed_bytes,
            "physical_bytes": self.physical_bytes,
            "fits": self.fits(self.physical_bytes),
            "check": (!self.fits(self.physical_bytes))
                .then(|| self.shortfall(self.physical_bytes).ok())
                .flatten(),
            "facts": {
                "replay_bytes_per_row": f.replay_bytes_per_row,
                "cache_bytes_per_token": f.cache_bytes_per_token,
                "int8_cache_bytes_per_token": f.int8_cache_bytes_per_token,
                "dense_bytes": f.dense_bytes,
                "hidden": f.hidden,
                "feed_forward": f.feed_forward,
                "vocabulary": f.vocabulary,
                "width": self.width,
                "draft_ring_bytes": f.draft.as_ref().map(|d| d.ring_bytes),
                "draft_features": f.draft.as_ref().map(|d| d.features),
            },
        })
    }
}

/// Powers of two below `context`, from the largest down to `floor` (positive).
fn smaller_contexts(context: usize, floor: usize) -> impl Iterator<Item = usize> {
    // The largest power of two below `context`: 65,536 for 65,537 and 32,768 for 65,536.
    let largest = context.checked_next_power_of_two().unwrap_or(0) / 2;
    std::iter::successors(Some(largest), |&n| Some(n / 2)).take_while(move |&n| n >= floor)
}

fn overflow() -> Error {
    Error::engine("memory estimate arithmetic overflow")
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(overflow)
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(overflow)
}
