//! Mapped draft loading and descriptor validation.
use super::*;

impl FastDraft {
    /// Load with the detected hardware and default draft options.
    /// # Safety
    /// Both draft files must remain unmodified and untruncated until this draft is dropped.
    pub unsafe fn load(folder: impl AsRef<Path>) -> Result<Self> {
        // SAFETY: the caller guarantees file stability for both draft files.
        unsafe {
            Self::with_options(
                folder,
                Engine::detected()?,
                Backend::detected(),
                DraftOptions::default(),
            )
        }
    }

    /// Explicit pool, projection backend and options. Never reads the environment.
    /// The float stages use NEON unless `engine` uses the portable kernel.
    /// # Safety
    /// Both draft files must remain unmodified and untruncated until this draft is dropped.
    pub unsafe fn with_options(
        folder: impl AsRef<Path>,
        engine: Engine,
        backend: Backend,
        options: DraftOptions,
    ) -> Result<Self> {
        // SAFETY: the caller guarantees file stability for both returned mappings.
        unsafe {
            Self::with_execution_options(
                folder,
                engine,
                crate::ExecutionOptions {
                    backend,
                    ..crate::ExecutionOptions::default()
                },
                options,
            )
        }
    }

    /// Load mapped draft weights with explicit execution and proposal options.
    /// # Safety
    /// Both draft files must remain unmodified and untruncated until this draft is dropped.
    pub unsafe fn with_execution_options(
        folder: impl AsRef<Path>,
        engine: Engine,
        execution: crate::ExecutionOptions,
        options: DraftOptions,
    ) -> Result<Self> {
        let (draft_path, head_path) = model_paths(folder)?;
        // SAFETY: the caller guarantees file stability for both mappings.
        let mapping = unsafe { MappedModel::open(draft_path)? };
        // SAFETY: as above.
        let head = unsafe { Head::open(&head_path)? };
        let d = DraftDimensions::read(mapping.header())?;
        validate_options(&d, &options, &head.layout)?;
        let fc = matrix(&mapping, "bdraft2.fc", d.hidden, d.features)?;
        let signs = fc.input_signs().to_vec();
        validate_signs(&d, &signs)?;
        let mut reader = mapping.reader()?;
        let mut vector = |name: &str, n: usize| -> Result<Vec<f32>> {
            vector_descriptor(&mapping, name, n)?;
            let values = reader.load_vector(name)?.values;
            if values.iter().any(|v| !v.is_finite()) {
                return Err(Error::NonFinite("draft vector"));
            }
            Ok(values)
        };
        let hidden_norm = vector("bdraft2.hidden_norm", d.hidden)?;
        let final_norm = vector("bdraft2.norm", d.hidden)?;
        let (qn, kvn) = (d.query_heads * HEAD_DIM, d.kv_heads * HEAD_DIM);
        let mut layers = Vec::new();
        for i in 0..d.layers {
            let p = format!("bdraft2.layers.{i}.");
            let m = |name: &str, rows, cols| matrix(&mapping, &format!("{p}{name}"), rows, cols);
            let (q, k, v, o) = (
                m("q", qn, d.hidden)?,
                m("k", kvn, d.hidden)?,
                m("v", kvn, d.hidden)?,
                m("o", d.hidden, qn)?,
            );
            let (gate, up, down) = (
                m("gate", d.feed_forward, d.hidden)?,
                m("up", d.feed_forward, d.hidden)?,
                m("down", d.hidden, d.feed_forward)?,
            );
            let (akp, mkp) = (
                m("akp", d.hidden / 4, d.hidden)?,
                m("mkp", d.hidden / 4, d.hidden)?,
            );
            layers.push(Layer {
                query: q,
                key: k,
                value: v,
                output: o,
                gate,
                up,
                down,
                attention_filter: akp,
                mlp_filter: mkp,
                input_norm: vector(&format!("{p}ln1"), d.hidden)?,
                post_attention_norm: vector(&format!("{p}ln2"), d.hidden)?,
                query_norm: vector(&format!("{p}q_norm"), HEAD_DIM)?,
                key_norm: vector(&format!("{p}k_norm"), HEAD_DIM)?,
                attention_bases: vector(&format!("{p}abase"), 4 * d.hidden)?,
                mlp_bases: vector(&format!("{p}mbase"), 4 * d.hidden)?,
            });
        }
        let hp = vector("bdraft2.sel.hp", d.correction_rank * d.hidden)?;
        let pred = bf16_rows(
            &mapping,
            "bdraft2.sel.pred",
            d.vocabulary,
            d.correction_rank,
        )?;
        let succ = bf16_rows(
            &mapping,
            "bdraft2.sel.succ",
            d.vocabulary,
            d.correction_rank,
        )?;
        let mut compute = Compute::new(engine, execution.backend)?;
        compute.cpu_stages = execution.cpu_stages.fast();
        compute.set_metal_options(execution.metal);
        Ok(Self {
            ring: DraftSnapshot::empty(d.layers, d.kv_heads, d.window),
            simd: execution.cpu_stages.fast() && compute.kernel() != Kernel::Portable,
            dimensions: d,
            options,
            signs,
            feature_projection: fc,
            hidden_norm,
            final_norm,
            layers,
            correction_projection: hp,
            predecessor_vectors: pred,
            successor_vectors: succ,
            head,
            compute,
            timing: DraftTiming::default(),
        })
    }
}

pub(super) fn matrix(
    mapping: &MappedModel,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Matrix> {
    let t = mapping
        .header()
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| invalid(format!("missing {name}")))?;
    if t.rows != rows || t.cols != cols {
        return Err(invalid(format!("{name}: draft matrix shape mismatch")));
    }
    // Quantizing the table checks the table and scales are finite; common
    // input signs, no output transform (engine35.cpp:1638, q35_bdraft.cu:474).
    mapping.matrix(name)
}

pub(super) fn vector_descriptor<'a>(
    mapping: &'a MappedModel,
    name: &str,
    n: usize,
) -> Result<&'a mackani_format::VectorDescriptor> {
    let v = mapping
        .header()
        .vectors
        .iter()
        .find(|v| v.name == name)
        .ok_or_else(|| invalid(format!("missing {name}")))?;
    if v.len != n || v.dtype != VectorType::Bf16 {
        return Err(invalid(format!("{name}: draft BF16 vector shape mismatch")));
    }
    Ok(v)
}

/// A large BF16 vector kept mapped, after the reference's finite check.
pub(super) fn bf16_rows(
    mapping: &MappedModel,
    name: &str,
    rows: usize,
    width: usize,
) -> Result<Bf16Rows> {
    let n = rows
        .checked_mul(width)
        .ok_or_else(|| invalid("draft dimension overflow"))?;
    let v = vector_descriptor(mapping, name, n)?;
    let bytes = mapping.mapping();
    let range = crate::mapped::region(bytes, v.offset, 2 * n as u64)?;
    if bytes[range.clone()]
        .as_chunks::<2>()
        .0
        .iter()
        .any(|b| u16::from_le_bytes(*b) & 0x7f80 == 0x7f80)
    {
        return Err(Error::NonFinite("draft vector"));
    }
    Ok(Bf16Rows {
        mapping: Arc::clone(bytes),
        range,
        width,
    })
}
