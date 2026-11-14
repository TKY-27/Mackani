//! Validate the shared synthetic writer once, independently of its consumers.
#[allow(dead_code, clippy::manual_is_multiple_of)]
mod support;
use support::*;

#[test]
fn synthetic_writer_builds_reproducible_mixed_model() {
    let layer_types = [
        "linear_attention",
        "full_attention",
        "linear_attention",
        "full_attention",
    ];
    let bytes = synthetic_model(19, &layer_types);
    assert_eq!(bytes, synthetic_model(19, &layer_types));
    assert_ne!(bytes, synthetic_model(20, &layer_types));
    let mut container = mackani_format::Container::from_bytes(bytes).unwrap();
    assert_eq!(container.header().model["layers"], 4);
    let tensor_names: Vec<_> = container
        .header()
        .tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect();
    let vector_names: Vec<_> = container
        .header()
        .vectors
        .iter()
        .map(|vector| vector.name.clone())
        .collect();
    for name in tensor_names {
        let tensor = container.load_tensor(&name).unwrap();
        assert!(
            tensor
                .table()
                .iter()
                .chain(tensor.scales())
                .all(|value| value.is_finite())
        );
        assert_eq!(tensor.row_states(0).unwrap().len(), tensor.cols() / 2);
    }
    for name in vector_names {
        assert!(
            container
                .load_vector(&name)
                .unwrap()
                .values
                .iter()
                .all(|value| value.is_finite())
        );
    }
}

#[test]
fn synthetic_writer_supports_other_dimensions_and_plain_embeddings() {
    let layer_types = ["linear_attention"];
    let bytes = synthetic_model_with_dimensions(21, &layer_types, 1024, 512, 65);
    let mut container = mackani_format::Container::from_bytes(bytes).unwrap();
    assert_eq!(container.header().model["hidden"], 1024);
    assert_eq!(container.header().model["ffn"], 512);
    assert_eq!(container.header().model["vocab"], 65);
    let head = container.load_tensor("lm_head").unwrap();
    assert_eq!(
        (head.rows(), head.cols(), head.physical_rows()),
        (65, 1024, 128)
    );
    let bytes = synthetic_model_with_dense_embeddings(21, &layer_types);
    let mut container = mackani_format::Container::from_bytes(bytes).unwrap();
    let embedding = container.load_vector("embed_tokens").unwrap();
    assert_eq!(embedding.values.len(), 128 * 512);
    assert_eq!(embedding.descriptor.dtype, mackani_format::VectorType::Bf16);
    assert!(container.load_tensor("embed_tokens").is_err());
}

#[test]
fn synthetic_writer_released_layout_uses_warp_and_released_bit_widths() {
    let bytes =
        synthetic_model_released_layout(23, &["linear_attention", "full_attention"], 512, 512, 65);
    let container = mackani_format::Container::from_bytes(bytes).unwrap();
    for tensor in &container.header().tensors {
        let (layout, bits) = match tensor.role {
            mackani_format::Role::Embedding => (mackani_format::Layout::Row, tensor.bits),
            mackani_format::Role::Head => (mackani_format::Layout::Warp, 8),
            _ => (mackani_format::Layout::Warp, 6),
        };
        assert_eq!(
            (tensor.layout, tensor.bits),
            (layout, bits),
            "{}",
            tensor.name
        );
    }
}
