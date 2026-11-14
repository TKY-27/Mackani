use super::*;

#[test]
fn repeated_owned_checkpoints_do_not_grow_session_buffers() {
    let mut model = Model::from_container(
        Container::from_bytes(crate::test_support::synthetic_model(
            139,
            &["linear_attention", "full_attention"],
        ))
        .unwrap(),
    )
    .unwrap();
    model.prefill(&[1, 2]).unwrap();
    let capacities = |model: &Model| {
        (
            model.layer_states.capacity(),
            model.committed_tokens.capacity(),
            model.committed_logits.as_ref().unwrap().capacity(),
            model.last_pass_logits.capacity(),
            model
                .layer_states
                .iter()
                .map(|state| match state {
                    LayerState::Attention(state) => state.capacity(),
                    LayerState::Recurrent(state) => state.allocated_bytes(),
                })
                .collect::<Vec<_>>(),
        )
    };
    let expected = capacities(&model);
    let saved = model.snapshot().unwrap();
    let snapshot_capacities = (
        saved.tokens.capacity(),
        saved.recurrent_states.capacity(),
        saved.committed_logits.as_ref().unwrap().capacity(),
    );
    for _ in 0..64 {
        let snapshot = model.snapshot().unwrap();
        assert_eq!(capacities(&model), expected);
        assert_eq!(
            (
                snapshot.tokens.capacity(),
                snapshot.recurrent_states.capacity(),
                snapshot.committed_logits.as_ref().unwrap().capacity(),
            ),
            snapshot_capacities
        );
        model.restore(&saved).unwrap();
        assert_eq!(capacities(&model), expected);
    }
}
