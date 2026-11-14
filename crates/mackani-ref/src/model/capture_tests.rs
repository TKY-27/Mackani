use super::*;
use crate::portable_math::bfloat;

#[test]
fn captured_two_layer_residual_is_independent_post_mlp_bf16_sum() {
    let mut bytes =
        crate::test_support::synthetic_model(317, &["full_attention", "full_attention"]);
    let n = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[32..32 + n]).unwrap();
    // Hand-set sizeable MLP scales so both residual boundaries are observable.
    for tensor in header["tensors"].as_array().unwrap() {
        if tensor["name"].as_str().unwrap().contains(".mlp.") {
            let start = tensor["scale_off"].as_u64().unwrap() as usize;
            let count = tensor["physical_rows"].as_u64().unwrap() as usize;
            for pair in bytes[start..start + 2 * count].chunks_exact_mut(2) {
                pair.copy_from_slice(&mackani_format::f32_to_f16(0.25).to_le_bytes());
            }
        }
    }
    let mut model = Model::from_container(Container::from_bytes(bytes).unwrap()).unwrap();
    let weights = Rc::get_mut(&mut model.weights).unwrap();
    weights.embeddings = Embeddings::Plain {
        values: (0..128 * 512)
            .map(|i| bfloat((i % 71) as f32 / 32.0 - 1.0))
            .collect(),
        hidden_width: 512,
    };
    for layer in &mut weights.layers {
        layer.input_norm.fill(2.0);
        layer.post_attention_norm.fill(3.0);
    }
    model
        .set_capture(CaptureOptions {
            layers: vec![0, 1],
            max_rows: 4,
        })
        .unwrap();
    let mut hidden = model.embedding(3).unwrap();
    let mut expected = Vec::new();
    for layer in &model.weights.layers {
        let Branch::Attention(branch) = &layer.branch else {
            unreachable!()
        };
        let mut state = branch.attention.new_state().unwrap();
        let norm = prepare_normalization(
            &hidden,
            &layer.input_norm,
            branch.query_gate.input_signs(),
            model.dimensions().epsilon,
        )
        .unwrap();
        let q = branch
            .query_gate
            .project(&norm.prepared, Some(norm.inverse_rms))
            .unwrap();
        let k = branch
            .key
            .project(&norm.prepared, Some(norm.inverse_rms))
            .unwrap();
        let v = branch
            .value
            .project(&norm.prepared, Some(norm.inverse_rms))
            .unwrap();
        let output = branch.attention.step(&mut state, 0, &q, &k, &v).unwrap();
        let prepared = quantize_activation(&output, branch.output.input_signs()).unwrap();
        let projected = branch.output.project(&prepared, None).unwrap();
        let old: Vec<_> = hidden
            .iter()
            .zip(projected)
            .map(|(&h, p)| bfloat(h + bfloat(p)))
            .collect();
        let norm = prepare_normalization(
            &old,
            &layer.post_attention_norm,
            layer.gate.input_signs(),
            model.dimensions().epsilon,
        )
        .unwrap();
        let gate = layer
            .gate
            .project(&norm.prepared, Some(norm.inverse_rms))
            .unwrap();
        let up = layer
            .up
            .project(&norm.prepared, Some(norm.inverse_rms))
            .unwrap();
        let product = gated_product(&gate, &up).unwrap();
        let prepared = quantize_activation(&product, layer.down.input_signs()).unwrap();
        let down = layer.down.project(&prepared, None).unwrap();
        hidden = old
            .iter()
            .zip(down)
            .map(|(&h, p)| bfloat(h + bfloat(p)))
            .collect();
        assert_ne!(hidden, old, "fixture must detect a tap before the MLP add");
        assert_ne!(hidden, norm.weighted_hidden_bf16);
        expected.extend_from_slice(&hidden);
    }
    model.prefill(&[3]).unwrap();
    assert_eq!(model.captured_rows()[0].position, 0);
    assert_eq!(
        model.captured_rows()[0]
            .values
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert_ne!(&expected[..512], &expected[512..]);
}

#[test]
fn capture_chunks_commit_restore_and_bounded_prefill_observe_without_changing_logits() {
    let bytes = crate::test_support::synthetic_model(319, &["linear_attention", "full_attention"]);
    let build = || Model::from_container(Container::from_bytes(bytes.clone()).unwrap()).unwrap();
    let mut whole = build();
    let mut single = build();
    let mut uneven = build();
    let mut plain = build();
    for model in [&mut whole, &mut single, &mut uneven] {
        model
            .set_capture(CaptureOptions {
                layers: vec![0, 1],
                max_rows: 16,
            })
            .unwrap();
    }
    let tokens = [1, 4, 2, 9, 3, 8, 7, 6, 5];
    whole.prefill(&tokens).unwrap();
    plain.prefill(&tokens).unwrap();
    assert_eq!(whole.row_logits(), plain.row_logits());
    assert_eq!(whole.state_digest(), plain.state_digest());
    for token in tokens {
        single.prefill(&[token]).unwrap();
    }
    for chunk in [&tokens[..2], &tokens[2..7], &tokens[7..]] {
        uneven.prefill(chunk).unwrap();
    }
    assert_eq!(whole.captured_rows(), single.captured_rows());
    assert_eq!(whole.captured_rows(), uneven.captured_rows());
    let snapshot = whole.snapshot().unwrap();
    whole.run_provisional(&[10, 11, 12]).unwrap();
    let provisional = whole.provisional_captures().to_vec();
    assert_eq!(
        provisional.iter().map(|r| r.position).collect::<Vec<_>>(),
        [9, 10, 11]
    );
    whole.commit(2).unwrap();
    assert_eq!(&whole.captured_rows()[9..], &provisional[..2]);
    whole.restore(&snapshot).unwrap();
    assert_eq!(whole.captured_rows(), single.captured_rows());
    whole.run_provisional(&[11]).unwrap();
    whole.discard_provisional();
    assert!(whole.provisional_captures().is_empty());
    let mut bounded = build();
    bounded
        .set_capture(CaptureOptions {
            layers: vec![0, 1],
            max_rows: 3,
        })
        .unwrap();
    bounded.prefill(&tokens).unwrap();
    assert_eq!(bounded.captured_rows(), &single.captured_rows()[6..]);
}
