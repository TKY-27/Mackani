//! Container validation, model layout and manifest resolution.
use super::*;

/// Load validated weights with backend-specific tensor storage.
pub fn load_weights<R: Read + Seek, L, T: EmbeddingTensor>(
    container: &mut Container<R>,
    linear: impl FnMut(&mut Container<R>, &str, usize, usize, bool) -> Result<L>,
    tensor: impl FnMut(&mut Container<R>, &str) -> Result<T>,
) -> Result<Weights<L, T>> {
    load_weights_with_kv(container, KvMode::Int8, None, linear, tensor)
}

/// Backend-independent loader; the profile is shared by immutable attention layers.
pub fn load_weights_with_kv<R: Read + Seek, L, T: EmbeddingTensor>(
    container: &mut Container<R>,
    mode: KvMode,
    profile: Option<Arc<KvProfile>>,
    mut linear: impl FnMut(&mut Container<R>, &str, usize, usize, bool) -> Result<L>,
    mut tensor: impl FnMut(&mut Container<R>, &str) -> Result<T>,
) -> Result<Weights<L, T>> {
    let header = container.header().model.clone();
    let dimensions = ModelDimensions::read(&header)?;
    let rotary_dim = rotary_width_from_header(&header)?;
    let layer_types = header
        .get("layer_types")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::InvalidModel("model.layer_types must be an array".into()))?;
    if layer_types.len() != dimensions.layer_count
        || layer_types.iter().any(|value| !value.is_string())
    {
        return Err(Error::InvalidModel(
            "layer_types must contain one string per layer".into(),
        ));
    }
    if mode != KvMode::Int8 {
        profile
            .as_ref()
            .ok_or_else(|| Error::InvalidModel("profile KV mode requires a KVQP profile".into()))?
            .check_dimensions(
                layer_types
                    .iter()
                    .filter(|kind| kind.as_str() == Some("full_attention"))
                    .count(),
                dimensions.kv_heads,
                dimensions.attention_head_width,
            )?;
    }
    let embeddings = if container
        .header()
        .tensors
        .iter()
        .any(|tensor| tensor.name == "embed_tokens")
    {
        let main = tensor(container, "embed_tokens")?;
        if main.rows() != dimensions.vocabulary_size || main.cols() != dimensions.hidden_width {
            return Err(Error::InvalidModel(
                "embedding dimensions do not match model header".into(),
            ));
        }
        let residual = optional_tensor(container, "embed_tokens.res", &mut tensor)?;
        let residual2 = optional_tensor(container, "embed_tokens.res2", &mut tensor)?;
        Embeddings::Coded(Box::new(Embedding::new(main, residual, residual2)?))
    } else {
        let count = product(dimensions.vocabulary_size, dimensions.hidden_width)?;
        Embeddings::Plain {
            values: load_vector(container, "embed_tokens", count)?,
            hidden_width: dimensions.hidden_width,
        }
    };

    let mut layers = Vec::new();
    let mut attention_ordinal = 0;
    for (index, kind) in layer_types.iter().enumerate() {
        let prefix = format!("layers.{index}");
        let branch = if kind.as_str() == Some("full_attention") {
            let query_width = product(dimensions.query_heads, dimensions.attention_head_width)?;
            let key_width = product(dimensions.kv_heads, dimensions.attention_head_width)?;
            let attention = Attention::new(
                AttentionDimensions {
                    query_heads: dimensions.query_heads,
                    kv_heads: dimensions.kv_heads,
                    head_dim: dimensions.attention_head_width,
                    rotary_dim,
                },
                load_vector(
                    container,
                    &format!("{prefix}.self_attn.q_norm"),
                    dimensions.attention_head_width,
                )?,
                load_vector(
                    container,
                    &format!("{prefix}.self_attn.k_norm"),
                    dimensions.attention_head_width,
                )?,
                dimensions.epsilon,
            )?
            .with_kv(mode, profile.clone(), attention_ordinal)?;
            attention_ordinal += 1;
            Branch::Attention(Box::new(AttentionLayer {
                query_gate: linear(
                    container,
                    &format!("{prefix}.self_attn.q_proj"),
                    product(2, query_width)?,
                    dimensions.hidden_width,
                    true,
                )?,
                key: linear(
                    container,
                    &format!("{prefix}.self_attn.k_proj"),
                    key_width,
                    dimensions.hidden_width,
                    true,
                )?,
                value: linear(
                    container,
                    &format!("{prefix}.self_attn.v_proj"),
                    key_width,
                    dimensions.hidden_width,
                    true,
                )?,
                output: linear(
                    container,
                    &format!("{prefix}.self_attn.o_proj"),
                    dimensions.hidden_width,
                    query_width,
                    true,
                )?,
                attention,
            }))
        } else {
            // The upstream dispatch treats every other layer-type string as GDN.
            Branch::Recurrent(Box::new(load_recurrent_layer(
                container,
                &prefix,
                &dimensions,
                &mut linear,
            )?))
        };
        layers.push(Layer {
            branch,
            input_norm: load_vector(
                container,
                &format!("{prefix}.input_layernorm"),
                dimensions.hidden_width,
            )?,
            post_attention_norm: load_vector(
                container,
                &format!("{prefix}.post_attention_layernorm"),
                dimensions.hidden_width,
            )?,
            gate: linear(
                container,
                &format!("{prefix}.mlp.gate_proj"),
                dimensions.feed_forward_width,
                dimensions.hidden_width,
                true,
            )?,
            up: linear(
                container,
                &format!("{prefix}.mlp.up_proj"),
                dimensions.feed_forward_width,
                dimensions.hidden_width,
                true,
            )?,
            down: linear(
                container,
                &format!("{prefix}.mlp.down_proj"),
                dimensions.hidden_width,
                dimensions.feed_forward_width,
                true,
            )?,
        });
    }
    let final_norm = load_vector(container, "norm", dimensions.hidden_width)?;
    let head = linear(
        container,
        "lm_head",
        dimensions.vocabulary_size,
        dimensions.hidden_width,
        false,
    )?;
    Ok(Weights {
        dimensions,
        embeddings,
        layers,
        final_norm,
        head,
    })
}

pub(super) fn load_recurrent_layer<R: Read + Seek, L>(
    container: &mut Container<R>,
    prefix: &str,
    model: &ModelDimensions,
    linear: &mut impl FnMut(&mut Container<R>, &str, usize, usize, bool) -> Result<L>,
) -> Result<RecurrentLayer<L>> {
    let name = |suffix: &str| format!("{prefix}.linear_attn.{suffix}");
    let head_dim = vector_length(container, &name("norm"))?;
    let value_heads = vector_length(container, &name("A_log"))?;
    let qkv_rows = container
        .header()
        .tensors
        .iter()
        .find(|tensor| tensor.name == name("in_proj_qkv"))
        .ok_or_else(|| Error::InvalidModel(format!("missing {}", name("in_proj_qkv"))))?
        .rows;
    let output_width = product(value_heads, head_dim)?;
    let query_key_width = qkv_rows.checked_sub(output_width).ok_or_else(|| {
        Error::InvalidModel("GDN qkv rows are smaller than its value width".into())
    })?;
    let two_heads = product(2, head_dim)?;
    if query_key_width % two_heads != 0 || query_key_width == 0 {
        return Err(Error::InvalidModel(
            "GDN qkv shape does not give integral query/key heads".into(),
        ));
    }
    let convolution_elements = vector_length(container, &name("conv1d"))?;
    if convolution_elements % qkv_rows != 0 {
        return Err(Error::InvalidModel(
            "GDN convolution shape does not give integral taps".into(),
        ));
    }
    // Chapter 03's shape equations determine these dimensions from header descriptors.
    let dimensions = GatedDeltaNetDimensions {
        hidden_width: model.hidden_width,
        query_key_heads: query_key_width / two_heads,
        value_heads,
        head_dim,
        convolution_width: convolution_elements / qkv_rows,
    };
    let dense_elements = product(value_heads, model.hidden_width)?;
    let recurrence = GatedDeltaNet::new(
        dimensions,
        GatedDeltaNetWeights {
            projection_a: load_vector(container, &name("in_proj_a"), dense_elements)?,
            projection_b: load_vector(container, &name("in_proj_b"), dense_elements)?,
            convolution_weight: load_vector(container, &name("conv1d"), convolution_elements)?,
            log_decay_weight: load_vector(container, &name("A_log"), value_heads)?,
            time_bias: load_vector(container, &name("dt_bias"), value_heads)?,
            normalization_weight: load_vector(container, &name("norm"), head_dim)?,
        },
        model.epsilon,
    )?;
    Ok(RecurrentLayer {
        query_key_value: linear(
            container,
            &name("in_proj_qkv"),
            qkv_rows,
            model.hidden_width,
            true,
        )?,
        output_gate: linear(
            container,
            &name("in_proj_z"),
            output_width,
            model.hidden_width,
            true,
        )?,
        output: linear(
            container,
            &name("out_proj"),
            model.hidden_width,
            output_width,
            true,
        )?,
        recurrence,
    })
}

pub(super) fn load_linear<R: Read + Seek>(
    container: &mut Container<R>,
    name: &str,
    rows: usize,
    cols: usize,
    output_rotation: bool,
) -> Result<CompressedLinear> {
    let tensor = container.load_tensor(name)?;
    if tensor.rows() != rows || tensor.cols() != cols {
        return Err(Error::InvalidModel(format!(
            "{name} must have shape [{rows}, {cols}]"
        )));
    }
    CompressedLinear::new(tensor, output_rotation)
}

pub(super) fn optional_tensor<R: Read + Seek, T>(
    container: &mut Container<R>,
    name: &str,
    tensor: &mut impl FnMut(&mut Container<R>, &str) -> Result<T>,
) -> Result<Option<T>> {
    if container
        .header()
        .tensors
        .iter()
        .any(|tensor| tensor.name == name)
    {
        Ok(Some(tensor(container, name)?))
    } else {
        Ok(None)
    }
}

pub(super) fn vector_length<R: Read + Seek>(container: &Container<R>, name: &str) -> Result<usize> {
    container
        .header()
        .vectors
        .iter()
        .find(|vector| vector.name == name)
        .map(|vector| vector.len)
        .filter(|&len| len > 0)
        .ok_or_else(|| Error::InvalidModel(format!("missing or empty vector {name}")))
}

pub(super) fn load_vector<R: Read + Seek>(
    container: &mut Container<R>,
    name: &str,
    length: usize,
) -> Result<Vec<f32>> {
    let descriptor = container
        .header()
        .vectors
        .iter()
        .find(|vector| vector.name == name)
        .ok_or_else(|| Error::InvalidModel(format!("missing vector {name}")))?;
    if descriptor.dtype != VectorType::Bf16 || descriptor.len != length {
        return Err(Error::InvalidModel(format!(
            "{name} must contain {length} BF16 values"
        )));
    }
    let vector = container.load_vector(name)?;
    require_finite(&vector.values, "model vector")?;
    Ok(vector.values)
}

pub(super) fn positive_dimension(model: &Value, name: &str) -> Result<usize> {
    model
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|&value| value > 0)
        .ok_or_else(|| {
            Error::InvalidModel(format!(
                "model.{name} must be a positive addressable integer"
            ))
        })
}

pub(super) fn product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| Error::InvalidModel("model dimension product overflow".into()))
}

pub(super) fn rotary_width_from_header(header: &Value) -> Result<usize> {
    if header.get("arch").and_then(Value::as_str) != Some("qwen3_5") {
        return Err(Error::InvalidModel(
            "main reference requires arch=qwen3_5".into(),
        ));
    }
    // This architecture fixes the attention width independently of the header.
    if header.get("head_dim").and_then(Value::as_u64) != Some(256) {
        return Err(Error::InvalidModel(
            "qwen3_5 requires model.head_dim=256".into(),
        ));
    }
    Ok(64)
}

/// Resolve a folder manifest (or a direct weights path) without opening model weights.
pub fn model_location(folder: &Path) -> Result<(PathBuf, usize)> {
    let (path, capacity, _) = model_location_with_kv(folder, KvMode::Int8)?;
    Ok((path, capacity))
}

/// Resolve weights/capacity and load only the optional profile, never weight payloads.
/// Mapped backends can reuse this with load_weights_with_kv.
pub fn model_location_with_kv(
    folder: &Path,
    mode: KvMode,
) -> Result<(PathBuf, usize, Option<Arc<KvProfile>>)> {
    if !folder.is_dir() {
        if mode != KvMode::Int8 {
            return Err(Error::InvalidModel(
                "profile KV mode requires a model folder or an explicit profile".into(),
            ));
        }
        return Ok((folder.to_path_buf(), 262_144, None));
    }
    let manifest_path = folder.join("yasakani.json");
    let manifest: Value = match fs::read(manifest_path) {
        Ok(bytes) => serde_json::from_slice(&bytes)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Value::Null,
        Err(error) => return Err(error.into()),
    };
    let weights = manifest_file(folder, &manifest, "weights", "qwen38_27b_e3x.yskn")?;
    let mut context_capacity = 262_144;
    if let Some(capacity) = manifest.get("max_ctx") {
        let capacity = capacity
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value > 0)
            .ok_or_else(|| {
                Error::InvalidModel("max_ctx must be a positive addressable integer".into())
            })?;
        context_capacity = capacity
            .checked_add(63)
            .map(|rounded| rounded / 64 * 64)
            .ok_or_else(|| Error::InvalidModel("rounded context capacity overflows".into()))?;
    }
    let profile = if mode == KvMode::Int8 {
        None
    } else {
        let path = manifest_file(folder, &manifest, "kv_profile", "kvq_u5.bin")?;
        Some(Arc::new(KvProfile::parse(&fs::read(path)?)?))
    };
    Ok((weights, context_capacity, profile))
}

pub(super) fn manifest_file(
    folder: &Path,
    manifest: &Value,
    key: &str,
    default: &str,
) -> Result<PathBuf> {
    let filename = match manifest.get("files").and_then(|files| files.get(key)) {
        Some(Value::String(filename)) => filename.as_str(),
        None => default,
        _ => return Err(Error::InvalidModel(format!("files.{key} must be a string"))),
    };
    if filename.is_empty()
        || Path::new(filename)
            .components()
            .any(|part| !matches!(part, Component::Normal(_) | Component::CurDir))
    {
        return Err(Error::InvalidModel(format!(
            "files.{key} must be relative to the model folder"
        )));
    }
    let path = folder.join(filename);
    // Apply the same containment rule to both files, including existing symlink targets.
    if path.exists() && !fs::canonicalize(&path)?.starts_with(fs::canonicalize(folder)?) {
        return Err(Error::InvalidModel(format!(
            "files.{key} must stay inside the model folder"
        )));
    }
    Ok(path)
}
