use super::*;

#[test]
fn closing_think_token_alone_is_not_an_answer() {
    assert!(!answer_stage_reached(true, 0, &[1]));
    assert!(!answer_stage_reached(true, 4, &[5]));
    assert!(!answer_stage_reached(true, 5, &[5]));
    assert!(!answer_stage_reached(true, 4, &[5, 0]));
    assert!(answer_stage_reached(true, 4, &[6]));
    assert!(answer_stage_reached(true, 5, &[5, 1]));
    assert!(answer_stage_reached(false, 0, &[1]));
}

#[test]
fn reporting_reset_only_or_zero_cached_tokens_cannot_cover_reuse() {
    let mut failures = vec![];
    reuse_exercised(
        "none",
        &[json!({"snapshot_modes":[0,1,2],"cached_tokens":0})],
        &mut failures,
    );
    assert_eq!(failures.len(), 2);
    failures.clear();
    reuse_exercised(
        "none",
        &[json!({"snapshot_modes":[0,1,2],"cached_tokens":32})],
        &mut failures,
    );
    assert!(failures.is_empty());
}

#[test]
fn acceptance_under_greedy_and_probes_cannot_cover_inactive_seeded_runs() {
    let mut rows = vec![json!({"sampler":"greedy","guesses_offered":10,"guesses_accepted":5})];
    for sampler in &SAMPLERS[1..] {
        rows.push(json!({"sampler":sampler,"guesses_offered":10,"guesses_accepted":0}));
        rows.push(json!({"sampler":sampler,"probe":true,"guesses_offered":10,"guesses_accepted":5,"neural_guesses_offered":5,"ngram_guesses_offered":5}));
    }
    let mut failures = vec![];
    speculation_exercised("both-k7-adaptive", &[0, 1, 2, 3], &rows, &mut failures);
    assert_eq!(
        failures
            .iter()
            .filter(|f| f["kind"] == "speculation not exercised")
            .count(),
        1
    );
    assert_eq!(
        failures
            .iter()
            .filter(|f| f["kind"] == "draft source not exercised")
            .count(),
        8
    );
}

#[test]
fn both_sources_are_required_per_sampler_but_hot_acceptance_may_be_zero() {
    let rows: Vec<_> = SAMPLERS.iter().map(|sampler| json!({"sampler":sampler,"guesses_offered":8,"guesses_accepted":2,"neural_guesses_offered":5,"ngram_guesses_offered":3})).collect();
    let mut failures = vec![];
    speculation_exercised("both-k7-adaptive", &[0, 1, 2, 3], &rows, &mut failures);
    assert!(failures.is_empty());
    let mut missing = rows;
    missing[3]["guesses_accepted"] = json!(0);
    speculation_exercised("both-k7-adaptive", &[0, 1, 2, 3], &missing, &mut failures);
    assert!(failures.is_empty());
    missing[1]["ngram_guesses_offered"] = json!(0);
    speculation_exercised("both-k7-adaptive", &[0, 1, 2, 3], &missing, &mut failures);
    assert_eq!(
        failures,
        vec![
            json!({"kind":"draft source not exercised","configuration":"both-k7-adaptive","sampler":"seed-42","source":"ngram_guesses_offered"})
        ]
    );
}

#[test]
fn two_empty_generations_are_not_a_lossless_pass() {
    for stages in [vec![], vec![0]] {
        let empty = Observation {
            ids: vec![],
            stages,
            finish: "length".into(),
            answer: String::new(),
        };
        let mut failures = vec![];
        compare(&empty, &empty, &json!({}), "repeat_mismatch", &mut failures);
        assert_eq!(failures[0]["kind"], "generation not exercised");
    }
}

#[test]
fn explicit_sampling_overrides_greedy_and_single_choice_defaults() {
    let prompt = prompts("test", r#"[{"id":"a","request":{"messages":[{"role":"user","content":"x"}]}},{"id":"b","request":{"messages":[{"role":"user","content":"y"}]}}]"#).unwrap().remove(0);
    let mut defaults = RequestDefaults::default();
    defaults.sampling.temperature = 0.0;
    defaults.sampling.top_k = 1;
    for sampler in 1..4 {
        let p = parameters(&prompt, &defaults, sampler, 32).unwrap();
        assert!(p.temperature > 0.0 && p.top_k > 1 && p.top_p >= 0.9);
    }
    defaults.sampling.temperature = 0.8;
    defaults.sampling.top_k = 11;
    let p = parameters(&prompt, &defaults, 2, 32).unwrap();
    assert_eq!((p.temperature, p.top_k), (0.8, 11));
}

#[test]
fn identical_samplers_and_identical_seeds_fail_the_diversity_gate() {
    let mut failures = vec![];
    sampling_exercised(
        "none",
        &[0, 1, 2, 3],
        &vec![vec![vec![1]]; 4],
        &mut failures,
    );
    assert_eq!(failures.len(), 6);
    assert!(
        failures
            .iter()
            .all(|f| f["kind"] == "sampling not exercised")
    );
    failures.clear();
    sampling_exercised(
        "none",
        &[0, 1, 2, 3],
        &[vec![vec![0]], vec![vec![1]], vec![vec![1]], vec![vec![3]]],
        &mut failures,
    );
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0]["samplers"], json!(["seed-42", "seed-2027"]));
}
