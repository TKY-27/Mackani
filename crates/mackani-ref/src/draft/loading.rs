//! Draft container and reduced-head loading.
use super::*;

impl Draft {
    /// Load the model's immutable weights with default options.
    pub fn load(folder: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_options(folder, DraftOptions::default())
    }
    /// Load draft weights and a reduced head with explicit proposal options.
    pub fn load_with_options(folder: impl AsRef<Path>, options: DraftOptions) -> Result<Self> {
        let (draft, head) = model_paths(folder)?;
        Self::from_container(
            &mut Container::open(draft)?,
            ReducedHead::open(head)?,
            options,
        )
    }
    /// Load validated scalar weights from an existing container.
    pub fn from_container<R: Read + Seek>(
        container: &mut Container<R>,
        head: ReducedHead,
        options: DraftOptions,
    ) -> Result<Self> {
        let d = DraftDimensions::read(container.header())?;
        validate_options(&d, &options, head.layout())?;
        let fc = matrix(container, "bdraft2.fc", d.hidden, d.features)?;
        let signs = fc.input_signs().to_vec();
        validate_signs(&d, &signs)?;
        let hidden_norm = vector(container, "bdraft2.hidden_norm", d.hidden)?;
        let final_norm = vector(container, "bdraft2.norm", d.hidden)?;
        let mut layers = Vec::new();
        for i in 0..d.layers {
            let p = format!("bdraft2.layers.{i}.");
            layers.push(Layer {
                query: matrix(
                    container,
                    &format!("{p}q"),
                    product(d.query_heads, HEAD_DIM)?,
                    d.hidden,
                )?,
                key: matrix(
                    container,
                    &format!("{p}k"),
                    product(d.kv_heads, HEAD_DIM)?,
                    d.hidden,
                )?,
                value: matrix(
                    container,
                    &format!("{p}v"),
                    product(d.kv_heads, HEAD_DIM)?,
                    d.hidden,
                )?,
                output: matrix(
                    container,
                    &format!("{p}o"),
                    d.hidden,
                    product(d.query_heads, HEAD_DIM)?,
                )?,
                gate: matrix(container, &format!("{p}gate"), d.feed_forward, d.hidden)?,
                up: matrix(container, &format!("{p}up"), d.feed_forward, d.hidden)?,
                down: matrix(container, &format!("{p}down"), d.hidden, d.feed_forward)?,
                attention_filter: matrix(container, &format!("{p}akp"), d.hidden / 4, d.hidden)?,
                mlp_filter: matrix(container, &format!("{p}mkp"), d.hidden / 4, d.hidden)?,
                input_norm: vector(container, &format!("{p}ln1"), d.hidden)?,
                post_attention_norm: vector(container, &format!("{p}ln2"), d.hidden)?,
                query_norm: vector(container, &format!("{p}q_norm"), HEAD_DIM)?,
                key_norm: vector(container, &format!("{p}k_norm"), HEAD_DIM)?,
                attention_bases: vector(container, &format!("{p}abase"), 4 * d.hidden)?,
                mlp_bases: vector(container, &format!("{p}mbase"), 4 * d.hidden)?,
            });
        }
        let hp = vector(
            container,
            "bdraft2.sel.hp",
            product(d.correction_rank, d.hidden)?,
        )?;
        let pred = vector(
            container,
            "bdraft2.sel.pred",
            product(d.vocabulary, d.correction_rank)?,
        )?;
        let succ = vector(
            container,
            "bdraft2.sel.succ",
            product(d.vocabulary, d.correction_rank)?,
        )?;
        product(product(d.kv_heads, d.window)?, HEAD_DIM)?;
        let ring = DraftSnapshot::empty(d.layers, d.kv_heads, d.window);
        Ok(Self {
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
            ring,
        })
    }
}

pub(super) fn matrix<R: Read + Seek>(
    c: &mut Container<R>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<CompressedLinear> {
    let t = c
        .header()
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| invalid(format!("missing {name}")))?;
    if t.rows != rows || t.cols != cols {
        return Err(invalid(format!("{name}: draft matrix shape mismatch")));
    }
    let t = c.load_tensor(name)?;
    require_finite(t.table(), "draft weight table")?;
    require_finite(t.scales(), "draft weight scales")?;
    // engine35.cpp:1638 and q35_bdraft.cu:474: common input signs, no output transform.
    CompressedLinear::new(t, false)
}
pub(super) fn vector<R: Read + Seek>(
    c: &mut Container<R>,
    name: &str,
    n: usize,
) -> Result<Vec<f32>> {
    let v = c
        .header()
        .vectors
        .iter()
        .find(|v| v.name == name)
        .ok_or_else(|| invalid(format!("missing {name}")))?;
    if v.len != n || v.dtype != VectorType::Bf16 {
        return Err(invalid(format!("{name}: draft BF16 vector shape mismatch")));
    }
    let v = c.load_vector(name)?.values;
    require_finite(&v, "draft vector")?;
    Ok(v)
}
