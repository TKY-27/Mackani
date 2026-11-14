use std::fs;

use mackani_format::{Container, Header, Role};
use mackani_ref::{KvMode, draft::DraftDimensions};
use serde_json::Value;

use super::{DraftFacts, MemoryFacts, add, mul, overflow, physical_memory};
use crate::{
    Error, Result,
    real_engine::{ModelInfo, RealEngineConfig},
};

fn dim(model: &Value, key: &str) -> Result<usize> {
    model
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| Error::engine(format!("model.{key} must be positive")))
}
fn vector(header: &Header, name: &str) -> Result<u64> {
    header
        .vectors
        .iter()
        .find(|v| v.name == name)
        .map(|v| v.len as u64)
        .ok_or_else(|| Error::engine(format!("missing {name}")))
}
fn codes(header: &Header) -> Result<u64> {
    header
        .tensors
        .iter()
        .filter(|t| t.role != Role::Embedding)
        .try_fold(0, |n, t| add(n, t.stream_bytes))
}
/// `physical`: the physical memory to price the estimate against; `None`
/// reads this Mac's.
pub(crate) fn inspect(config: &RealEngineConfig, physical: Option<u64>) -> Result<ModelInfo> {
    config.validate()?;
    let (path, rounded_max, profile) =
        mackani_ref::model::model_location_with_kv(&config.model_dir, config.kv)
            .map_err(Error::engine)?;
    let manifest: Value = serde_json::from_slice(
        &fs::read(config.model_dir.join("yasakani.json")).map_err(Error::engine)?,
    )
    .map_err(Error::engine)?;
    let max_context = manifest
        .get("max_ctx")
        .map(|_| dim(&manifest, "max_ctx"))
        .transpose()?
        .unwrap_or(rounded_max);
    if config.context > max_context {
        return Err(Error::engine(format!(
            "--ctx {} exceeds model max_ctx {max_context}",
            config.context
        )));
    }
    let container = Container::open(&path).map_err(Error::engine)?;
    let h = container.header();
    let hidden = dim(&h.model, "hidden")?;
    let ffn = dim(&h.model, "ffn")?;
    let vocab = dim(&h.model, "vocab")?;
    let layers = dim(&h.model, "layers")?;
    let kv_heads = dim(&h.model, "kv_heads")? as u64;
    let head_width = dim(&h.model, "head_dim")? as u64;
    let kinds = h
        .model
        .get("layer_types")
        .and_then(Value::as_array)
        .filter(|v| v.len() == layers)
        .ok_or_else(|| Error::engine("invalid model.layer_types"))?;
    let mut recurrent = 0;
    let mut replay_per_row = 0;
    let mut cache_per_token = 0;
    let mut int8_cache_per_token = 0;
    let mut attention = 0;
    for (index, kind) in kinds.iter().enumerate() {
        if kind.as_str() == Some("full_attention") {
            let int8_row = mul(2, mul(kv_heads, add(32, head_width)?)?)?;
            let mut row_bytes = int8_row;
            if config.kv == KvMode::Kv5 {
                let p = profile
                    .as_ref()
                    .ok_or_else(|| Error::engine("missing KV profile"))?;
                p.check_dimensions(
                    kinds
                        .iter()
                        .filter(|k| k.as_str() == Some("full_attention"))
                        .count(),
                    kv_heads as usize,
                    head_width as usize,
                )
                .map_err(Error::engine)?;
                for side in 0..2 {
                    if p.packable(attention, side) {
                        // Packed rows include 32-byte enum + 544-byte packed row + packed u32 words.
                        let words = mul(4, mul(head_width, p.packed_bits() as u64)?.div_ceil(32))?;
                        let packed = add(576, words)?;
                        row_bytes = add(row_bytes, mul(kv_heads, packed - (32 + head_width))?)?;
                    }
                }
            }
            cache_per_token = add(cache_per_token, add(48, row_bytes)?)?;
            int8_cache_per_token = add(int8_cache_per_token, add(48, int8_row)?)?;
            attention += 1;
        } else {
            let prefix = format!("layers.{index}.linear_attn");
            let hd = vector(h, &format!("{prefix}.norm"))?;
            let vh = vector(h, &format!("{prefix}.A_log"))?;
            let qkv = h
                .tensors
                .iter()
                .find(|t| t.name == format!("{prefix}.in_proj_qkv"))
                .ok_or_else(|| Error::engine("missing recurrent qkv"))?
                .rows as u64;
            let conv = vector(h, &format!("{prefix}.conv1d"))?;
            let out = mul(vh, hd)?;
            let qk = qkv.checked_sub(out).ok_or_else(overflow)?;
            let hist = conv.checked_sub(qkv).ok_or_else(overflow)?;
            recurrent = add(recurrent, mul(4, add(mul(vh, mul(hd, hd)?)?, hist)?)?)?;
            replay_per_row = add(
                replay_per_row,
                mul(4, add(add(mul(2, qkv)?, out)?, add(qk, mul(2, vh)?)?)?)?,
            )?;
        }
    }
    let width = config
        .prefill_width
        .max(config.k.checked_add(1).ok_or_else(overflow)?);
    let weights = h.file_bytes;
    let dense = h
        .vectors
        .iter()
        .try_fold(0, |n, v| add(n, mul(v.len as u64, 4)?))?;
    let mut draft = None;
    let mut proposal_codes = 0;
    let mut update_codes = 0;
    if config.draft.neural() {
        let (draft_path, head_path) =
            mackani_ref::draft::model_paths(&config.model_dir).map_err(Error::engine)?;
        let d = Container::open(draft_path).map_err(Error::engine)?;
        let dims = DraftDimensions::read(d.header()).map_err(Error::engine)?;
        let layout = mackani_ref::draft_head::HeadLayout::read(
            &mut fs::File::open(&head_path).map_err(Error::engine)?,
        )
        .map_err(Error::engine)?;
        // Code streams and selector pred/succ stay mapped. Only the norms,
        // layer vectors and selector hp are expanded to f32 (FastDraft).
        let len = |range: &std::ops::Range<u64>| range.end - range.start;
        let owned_head = [0, 1, 2, 4]
            .into_iter()
            .try_fold(0, |n, i| add(n, len(&layout.arrays[i])))?;
        let mut vector_elements = 0;
        for name in ["bdraft2.hidden_norm", "bdraft2.norm", "bdraft2.sel.hp"] {
            vector_elements = add(vector_elements, vector(d.header(), name)?)?;
        }
        for i in 0..dims.layers {
            for name in ["ln1", "ln2", "q_norm", "k_norm", "abase", "mbase"] {
                vector_elements = add(
                    vector_elements,
                    vector(d.header(), &format!("bdraft2.layers.{i}.{name}"))?,
                )?;
            }
        }
        let vectors = mul(vector_elements, 4)?;
        let bytes = [
            d.header().file_bytes,
            layout.file_bytes,
            owned_head,
            vectors,
        ]
        .into_iter()
        .try_fold(0, add)?;
        // Logical code reads, not a hardware/DRAM bandwidth measurement. A
        // proposal runs every block matrix once for its eight rows, and seven
        // generated rows each scan approximate V*rank int8 weights and rescore
        // shortlist*hidden int8 weights. A history update (one push of
        // consecutive committed rows, at most 256 per read) runs the FC input
        // projection and every layer's key/value projections.
        let mut fc = None;
        for t in &d.header().tensors {
            let kv = t.name.starts_with("bdraft2.layers.")
                && (t.name.ends_with(".k") || t.name.ends_with(".v"));
            if t.name == "bdraft2.fc" {
                fc = Some(t.stream_bytes);
            }
            if t.name == "bdraft2.fc" || kv {
                update_codes = add(update_codes, t.stream_bytes)?;
            }
        }
        let fc = fc.ok_or_else(|| Error::engine("missing bdraft2.fc"))?;
        proposal_codes = add(
            codes(d.header())?.checked_sub(fc).ok_or_else(overflow)?,
            reduced_head_code_reads(&layout, 512.min(layout.vocabulary))?,
        )?;
        let ring_bytes = mul(
            4,
            mul(
                dims.layers as u64,
                mul(dims.kv_heads as u64, mul(dims.window as u64, 128)?)?,
            )?,
        )?;
        draft = Some(DraftFacts {
            bytes,
            ring_bytes,
            features: dims.features as u64,
        });
    }
    let facts = MemoryFacts {
        weights_bytes: weights,
        recurrent_bytes: recurrent,
        replay_bytes_per_row: replay_per_row,
        cache_bytes_per_token: cache_per_token,
        int8_cache_bytes_per_token: int8_cache_per_token,
        dense_bytes: dense,
        hidden: hidden as u64,
        feed_forward: ffn as u64,
        vocabulary: vocab as u64,
        draft,
    };
    let physical = match physical {
        Some(bytes) => bytes,
        None => physical_memory()?,
    };
    let memory = facts.estimate(config.context, width, physical)?;
    let run = crate::platform::RunChoice::select(config)?;
    Ok(ModelInfo {
        model_file: path,
        model_bytes: weights,
        code_bytes_per_pass: codes(h)?,
        draft_proposal_code_bytes: proposal_codes,
        draft_update_code_bytes: update_codes,
        layers,
        hidden_width: hidden,
        feed_forward_width: ffn,
        vocabulary: vocab,
        context: config.context,
        model_max_context: max_context,
        threads: run.threads,
        kv: config.kv_name().into(),
        backend: run.backend.name().into(),
        device: run.device.clone(),
        draft: config.draft.name().into(),
        k: config.k,
        depth: match config.depth_policy() {
            mackani_ref::speculation::DepthPolicy::Fixed => "fixed",
            mackani_ref::speculation::DepthPolicy::Adaptive(_) => "adaptive",
        }
        .into(),
        memory,
        run,
    })
}

pub(crate) fn reduced_head_code_reads(
    layout: &mackani_ref::draft_head::HeadLayout,
    shortlist: usize,
) -> Result<u64> {
    mul(
        (mackani_ref::draft::BLOCK_ROWS - 1) as u64,
        add(
            mul(layout.vocabulary as u64, layout.rank as u64)?,
            mul(shortlist as u64, layout.hidden as u64)?,
        )?,
    )
}
