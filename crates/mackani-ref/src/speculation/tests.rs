use super::*;
use crate::SamplingParameters;

// This verifier is a deterministic sequence-to-logits function. The oracle
// below drives that function directly, without provisional/commit or generate.
fn prediction(prefix: &[u32]) -> Vec<f32> {
    let state = prefix.iter().fold(19_u64, |state, &token| {
        state.wrapping_mul(31).wrapping_add(u64::from(token) + 7)
    });
    (0..8)
        .map(|token| ((state.rotate_right(token * 7) % 113) as f32 - 50.0) / 23.0)
        .collect()
}

#[derive(Clone, Debug)]
struct SequenceVerifier {
    committed: Vec<u32>,
    pending: Option<Vec<u32>>,
    rows: Vec<Vec<f32>>,
    last: Option<Vec<f32>>,
    capacity: usize,
    width: usize,
}
impl SequenceVerifier {
    fn new(prefix: &[u32], capacity: usize, width: usize) -> Self {
        Self {
            committed: prefix.to_vec(),
            pending: None,
            rows: Vec::new(),
            last: (!prefix.is_empty()).then(|| prediction(prefix)),
            capacity,
            width,
        }
    }
}
impl Verifier for SequenceVerifier {
    type Error = Error;
    fn position(&self) -> usize {
        self.committed.len()
    }
    fn context_capacity(&self) -> usize {
        self.capacity
    }
    fn max_pass_width(&self) -> usize {
        self.width
    }
    fn run_provisional(&mut self, tokens: &[u32]) -> Result<()> {
        assert!(self.pending.is_none());
        assert!(!tokens.is_empty());
        assert!(tokens.len() <= self.width);
        assert!(self.position() + tokens.len() <= self.capacity);
        let mut prefix = self.committed.clone();
        self.rows = tokens
            .iter()
            .map(|&token| {
                prefix.push(token);
                prediction(&prefix)
            })
            .collect();
        self.pending = Some(tokens.to_vec());
        Ok(())
    }
    fn row_logits(&self, row: usize) -> Option<&[f32]> {
        self.rows.get(row).map(Vec::as_slice)
    }
    fn logits(&self) -> Option<&[f32]> {
        self.last.as_deref()
    }
    fn commit(&mut self, rows: usize) -> Result<()> {
        let pending = self.pending.take().unwrap();
        assert!(rows <= pending.len());
        self.committed.extend_from_slice(&pending[..rows]);
        self.last = Some(prediction(&self.committed));
        self.rows.truncate(rows);
        Ok(())
    }
    fn discard(&mut self) {
        self.pending = None;
        self.rows.clear();
    }
}

#[test]
fn observation_failure_keeps_committed_outputs_and_continues_without_drafting() {
    struct FailsOnObservation {
        calls: usize,
    }
    impl Proposer for FailsOnObservation {
        fn propose(&mut self, _: &[u32], _: u32, _: usize) -> Vec<u32> {
            vec![]
        }
        fn observe_committed(&mut self, _: &[CapturedRow]) -> Result<()> {
            self.calls += 1;
            if self.calls == 2 {
                Err(Error::InvalidInput("observation failed".into()))
            } else {
                Ok(())
            }
        }
    }
    let mut expected = SequenceVerifier::new(&[1, 2], 32, 4);
    let mut actual = expected.clone();
    let mut limits = GenerationLimits::new(7, 32);
    limits.max_draft_tokens = 3;
    let baseline = generate(
        &mut expected,
        &mut NoProposer,
        &mut Greedy,
        &[1, 2, 3],
        GenerationLimits::new(7, 32),
    )
    .unwrap();
    let mut proposer = FailsOnObservation { calls: 0 };
    let generated = generate(&mut actual, &mut proposer, &mut Greedy, &[1, 2, 3], limits).unwrap();
    assert_eq!(generated.tokens, baseline.tokens);
    assert_eq!(actual.committed, expected.committed);
    assert_eq!(actual.last, expected.last);
    assert_eq!(proposer.calls, 2);
}

struct FailsOnProposal {
    calls: usize,
    fail_on: usize,
}
impl Proposer for FailsOnProposal {
    fn propose(&mut self, _: &[u32], _: u32, _: usize) -> Vec<u32> {
        unreachable!()
    }
    fn propose_with<V: Verifier>(
        &mut self,
        _: &mut V,
        _: &[u32],
        _: u32,
        _: usize,
    ) -> Result<Vec<u32>> {
        self.calls += 1;
        if self.calls == self.fail_on {
            Err(Error::InvalidInput("proposal failed".into()))
        } else {
            Ok(vec![])
        }
    }
}

#[test]
fn proposal_failure_after_output_disables_drafting_and_matches_k_zero() {
    let mut expected = SequenceVerifier::new(&[1, 2], 32, 4);
    let mut actual = expected.clone();
    let baseline = generate(
        &mut expected,
        &mut NoProposer,
        &mut Greedy,
        &[1, 2, 3],
        GenerationLimits::new(7, 32),
    )
    .unwrap();
    let mut proposer = FailsOnProposal {
        calls: 0,
        fail_on: 2,
    };
    let mut seen = Vec::new();
    let mut callback = |token| {
        seen.push(token);
        true
    };
    let mut limits = GenerationLimits::new(7, 32);
    limits.max_draft_tokens = 3;
    limits.on_token = Some(&mut callback);
    let generated = generate(&mut actual, &mut proposer, &mut Greedy, &[1, 2, 3], limits).unwrap();
    assert_eq!(generated.tokens, baseline.tokens);
    assert_eq!(seen, baseline.tokens);
    assert_eq!(generated.stop_reason, baseline.stop_reason);
    assert_eq!(actual.committed, expected.committed);
    assert_eq!(actual.last, expected.last);
    assert!(actual.pending.is_none());
    assert_eq!(proposer.calls, 2);
}

#[test]
fn proposal_failure_before_output_returns_named_error_without_changing_verifier() {
    let mut main = SequenceVerifier::new(&[1, 2], 32, 4);
    let before = main.clone();
    let mut proposer = FailsOnProposal {
        calls: 0,
        fail_on: 1,
    };
    let mut limits = GenerationLimits::new(7, 32);
    limits.max_draft_tokens = 3;
    let error = generate(&mut main, &mut proposer, &mut Greedy, &[1, 2, 3], limits).unwrap_err();
    assert!(matches!(
        error,
        GenerationError::Proposer(Error::InvalidInput(message)) if message == "proposal failed"
    ));
    assert_eq!(main.committed, before.committed);
    assert_eq!(main.last, before.last);
    assert_eq!(main.rows, before.rows);
    assert_eq!(main.pending, before.pending);
}

#[test]
fn invalid_guesses_before_output_error_and_after_output_fall_back_losslessly() {
    struct Misbehaving {
        calls: usize,
        fail_on: usize,
        oversized: bool,
    }
    impl Proposer for Misbehaving {
        fn propose(&mut self, _: &[u32], _: u32, max: usize) -> Vec<u32> {
            self.calls += 1;
            if self.calls != self.fail_on {
                vec![]
            } else if self.oversized {
                vec![0; max + 1]
            } else {
                vec![8] // SequenceVerifier has vocabulary IDs 0..8.
            }
        }
    }
    for oversized in [false, true] {
        for fail_on in [1, 2] {
            let mut expected = SequenceVerifier::new(&[1, 2], 32, 4);
            let mut actual = expected.clone();
            let before = actual.clone();
            let baseline = generate(
                &mut expected,
                &mut NoProposer,
                &mut Greedy,
                &[1, 2, 3],
                GenerationLimits::new(7, 32),
            )
            .unwrap();
            let mut proposer = Misbehaving {
                calls: 0,
                fail_on,
                oversized,
            };
            let mut seen = Vec::new();
            let mut callback = |token| {
                seen.push(token);
                true
            };
            let mut limits = GenerationLimits::new(7, 32);
            limits.max_draft_tokens = 3;
            limits.on_token = Some(&mut callback);
            let generated = generate(&mut actual, &mut proposer, &mut Greedy, &[1, 2, 3], limits);
            if fail_on == 1 {
                let expected_error = if oversized {
                    "proposer exceeded the requested guess limit"
                } else {
                    "proposer returned a token outside the vocabulary"
                };
                assert!(matches!(
                    generated,
                    Err(GenerationError::InvalidInput(message)) if message == expected_error
                ));
                assert!(seen.is_empty());
                assert_eq!(actual.committed, before.committed);
                assert_eq!(actual.last, before.last);
                assert_eq!(actual.rows, before.rows);
                assert_eq!(actual.pending, before.pending);
            } else {
                let generated = generated.unwrap();
                assert_eq!(generated.tokens, baseline.tokens);
                assert_eq!(seen, baseline.tokens);
                assert_eq!(generated.stop_reason, baseline.stop_reason);
                assert_eq!(actual.committed, expected.committed);
                assert_eq!(actual.last, expected.last);
                assert!(actual.pending.is_none());
            }
            assert_eq!(proposer.calls, fail_on);
        }
    }
}

#[test]
fn source_offers_count_verified_guesses_after_depth_truncation_even_when_cancelled() {
    struct Sourced {
        source: ProposalSource,
        free: bool,
    }
    impl Proposer for Sourced {
        fn propose(&mut self, _: &[u32], _: u32, max: usize) -> Vec<u32> {
            vec![0; max]
        }
        fn free_guesses(&mut self, _: &[u32], _: u32, max: usize) -> Vec<u32> {
            if self.free { vec![0; max] } else { Vec::new() }
        }
        fn proposal_source(&self) -> ProposalSource {
            self.source
        }
    }
    // This independent cost table favors exactly two paid guesses out of
    // four available; free guesses bypass that depth and offer all four.
    let policy = DepthPolicy::Adaptive(AdaptiveDepth {
        costs: CostModel {
            pass: vec![1.0, 1.0, 1.0, 100.0, 100.0],
            proposal: 0.0,
        },
        prior: vec![1.0, 1.0, 0.0, 0.0],
        rate: 0.25,
        probe_after: 4,
        relax: 0.0,
    });
    for (source, free, depth, expected) in [
        (ProposalSource::Neural, false, policy.clone(), (2, 2, 0)),
        (ProposalSource::Ngram, true, policy, (4, 0, 4)),
        (ProposalSource::Ngram, false, DepthPolicy::Fixed, (4, 0, 4)),
        (
            ProposalSource::Unspecified,
            false,
            DepthPolicy::Fixed,
            (4, 0, 0),
        ),
    ] {
        let mut verifier = SequenceVerifier::new(&[1, 2], 64, 9);
        let mut proposer = Sourced { source, free };
        let mut limits = GenerationLimits::new(5, 64);
        limits.max_draft_tokens = 7;
        limits.depth = depth;
        let mut cancel = |_| false;
        limits.on_token = Some(&mut cancel);
        let result = generate(
            &mut verifier,
            &mut proposer,
            &mut Greedy,
            &[1, 2, 3],
            limits,
        )
        .unwrap();
        assert_eq!(result.stop_reason, StopReason::Cancelled);
        assert_eq!(result.tokens.len(), 1);
        assert_eq!(result.statistics.rows_verified, expected.0 + 1);
        assert_eq!(
            (
                result.statistics.guesses_offered,
                result.statistics.neural_guesses_offered,
                result.statistics.ngram_guesses_offered
            ),
            expected
        );
    }
}

struct GuidedProposer {
    sampler: Sampler,
    // None: oracle; Some(0): always wrong; Some(usize::MAX): varying prefix.
    wrong_after: Option<usize>,
}
impl Proposer for GuidedProposer {
    fn propose(&mut self, history: &[u32], last: u32, max: usize) -> Vec<u32> {
        let mut prefix = history.to_vec();
        prefix.push(last);
        let wrong_after = match self.wrong_after {
            Some(usize::MAX) => Some(prefix.iter().sum::<u32>() as usize % (max + 1)),
            other => other,
        };
        (0..max)
            .map(|depth| {
                let selected = self
                    .sampler
                    .sample(&prediction(&prefix), prefix.len() as u64)
                    .unwrap();
                let guessed = if wrong_after.is_some_and(|after| depth >= after) {
                    (selected + 1) % 8
                } else {
                    selected
                };
                prefix.push(guessed);
                guessed
            })
            .collect()
    }
}

fn samplers() -> Vec<Sampler> {
    let sampled = |seed| {
        Sampler::Sampled(SamplingParameters {
            seed,
            top_k: 8,
            ..SamplingParameters::default()
        })
    };
    vec![Sampler::Greedy, sampled(17), sampled(0xabcdef)]
}

// An intentionally plain token-at-a-time oracle with no generation traits.
fn sequential(
    prompt: &[u32],
    sampler: &Sampler,
    count: usize,
    capacity: usize,
    eos: &[u32],
    cancel_after: Option<usize>,
) -> (Vec<u32>, Vec<u32>, StopReason) {
    let mut prefix = prompt[..prompt.len() - 1].to_vec();
    let mut pending = prompt[prompt.len() - 1];
    let mut output = Vec::new();
    let mut stop = StopReason::MaxNewTokens;
    while output.len() < count {
        if prefix.len() == capacity {
            stop = StopReason::ContextCapacity;
            break;
        }
        prefix.push(pending);
        pending = sampler
            .sample(&prediction(&prefix), prefix.len() as u64)
            .unwrap();
        output.push(pending);
        if eos.contains(&pending) {
            stop = StopReason::EndOfSequence;
            break;
        }
        if cancel_after == Some(output.len()) {
            stop = StopReason::Cancelled;
            break;
        }
    }
    (output, prefix, stop)
}

fn check(
    prompt: &[u32],
    sampler: &Sampler,
    proposer: &mut impl Proposer,
    k: usize,
    sizes: (usize, usize),
    eos: &[u32],
    cancel_after: Option<usize>,
) -> GenerationResult {
    check_with(
        DepthPolicy::Fixed,
        prompt,
        sampler,
        proposer,
        k,
        sizes,
        eos,
        cancel_after,
    )
}

#[allow(clippy::too_many_arguments)]
fn check_with(
    depth: DepthPolicy,
    prompt: &[u32],
    sampler: &Sampler,
    proposer: &mut impl Proposer,
    k: usize,
    sizes: (usize, usize),
    eos: &[u32],
    cancel_after: Option<usize>,
) -> GenerationResult {
    let (max, capacity) = sizes;
    let expected = sequential(prompt, sampler, max, capacity, eos, cancel_after);
    let mut verifier = SequenceVerifier::new(&prompt[..prompt.len() - 1], capacity, 9);
    let mut selector = sampler.clone();
    let mut seen = Vec::new();
    let mut callback = |token| {
        seen.push(token);
        cancel_after != Some(seen.len())
    };
    let mut limits = GenerationLimits::new(max, capacity);
    limits.max_draft_tokens = k;
    limits.depth = depth;
    limits.eos_ids = eos;
    limits.on_token = Some(&mut callback);
    let result = generate(&mut verifier, proposer, &mut selector, prompt, limits).unwrap();
    assert_eq!(result.tokens, expected.0);
    assert_eq!(seen, expected.0);
    assert_eq!(verifier.committed, expected.1);
    assert_eq!(verifier.logits(), Some(prediction(&expected.1).as_slice()));
    assert!(verifier.pending.is_none());
    assert_eq!(result.stop_reason, expected.2);
    assert_eq!(
        result
            .statistics
            .tokens_confirmed_per_pass
            .iter()
            .sum::<usize>(),
        result.tokens.len()
    );
    result
}

#[test]
fn speculation_preserves_sequential_tokens_state_and_seeded_repeatability() {
    let prompt = [1, 2, 3, 1, 2, 3, 1];
    for sampler in samplers() {
        for k in [1, 2, 3, 7] {
            check(&prompt, &sampler, &mut NoProposer, k, (23, 64), &[], None);
            for wrong_after in [None, Some(0), Some(usize::MAX)] {
                let mut proposer = GuidedProposer {
                    sampler: sampler.clone(),
                    wrong_after,
                };
                let result = check(&prompt, &sampler, &mut proposer, k, (23, 64), &[], None);
                let repeated = check(&prompt, &sampler, &mut proposer, k, (23, 64), &[], None);
                assert_eq!(result.tokens, repeated.tokens);
                assert_eq!(
                    result.statistics.tokens_confirmed_per_pass,
                    repeated.statistics.tokens_confirmed_per_pass
                );
                if wrong_after.is_none() {
                    assert!(result.statistics.passes < result.tokens.len());
                    assert_eq!(
                        result.statistics.guesses_offered,
                        result.statistics.guesses_accepted
                    );
                }
                if wrong_after == Some(0) {
                    assert_eq!(result.statistics.passes, result.tokens.len());
                    assert_eq!(result.statistics.guesses_accepted, 0);
                }
            }
            let mut ngram = NgramProposer::new(2, k).unwrap();
            check(&prompt, &sampler, &mut ngram, k, (23, 64), &[], None);
        }
    }
}

#[test]
fn adaptive_depth_preserves_sequential_tokens_and_repeats_its_depths() {
    let prompt = [1, 2, 3, 1, 2, 3, 1];
    let policy = DepthPolicy::Adaptive(AdaptiveDepth::neural());
    for sampler in samplers() {
        for k in [1, 3, 7] {
            for wrong_after in [None, Some(0), Some(2), Some(usize::MAX)] {
                let mut proposer = GuidedProposer {
                    sampler: sampler.clone(),
                    wrong_after,
                };
                for (eos, cancel) in [(vec![], None), (vec![5], None), (vec![], Some(9))] {
                    let run = |proposer: &mut GuidedProposer| {
                        let sizes = (40, 64);
                        let p = policy.clone();
                        check_with(p, &prompt, &sampler, proposer, k, sizes, &eos, cancel)
                    };
                    let first = run(&mut proposer);
                    let again = run(&mut proposer);
                    let stats = (&first.statistics, &again.statistics);
                    assert_eq!(stats.0.requested_depths, stats.1.requested_depths);
                    assert_eq!(stats.0.passes, stats.0.requested_depths.len());
                    assert!(stats.0.requested_depths.iter().all(|&d| d <= k));
                    let calls = stats.0.requested_depths.iter().filter(|&&d| d > 0);
                    assert_eq!(calls.count(), stats.0.proposals);
                    if wrong_after.is_none() && cancel.is_none() && eos.is_empty() {
                        assert!(stats.0.passes < first.tokens.len());
                    }
                    if wrong_after == Some(0) && eos.is_empty() && cancel.is_none() {
                        // Never right: the proposer is skipped on most passes.
                        assert!(stats.0.proposals * 2 < stats.0.passes, "{stats:?}");
                    }
                }
            }
            let mut ngram = NgramProposer::new(2, k).unwrap();
            check_with(
                policy.clone(),
                &prompt,
                &sampler,
                &mut ngram,
                k,
                (23, 64),
                &[],
                None,
            );
        }
    }
}

#[test]
fn invalid_adaptive_configuration_is_rejected_before_any_pass() {
    let mut verifier = SequenceVerifier::new(&[1, 2], 32, 4);
    let mut config = AdaptiveDepth::neural();
    config.rate = f64::NAN;
    let mut limits = GenerationLimits::new(7, 32);
    limits.max_draft_tokens = 3;
    limits.depth = DepthPolicy::Adaptive(config);
    let error = generate(
        &mut verifier,
        &mut NoProposer,
        &mut Greedy,
        &[1, 2, 3],
        limits,
    );
    assert!(matches!(error, Err(GenerationError::InvalidInput(_))));
    assert!(verifier.pending.is_none());
    assert_eq!(verifier.committed, [1, 2]);
}

#[test]
fn eos_cancel_and_capacity_discard_later_rows_and_keep_last_token_pending() {
    let prompt = [1, 2, 3, 1, 2, 3, 1];
    for policy in [
        DepthPolicy::Fixed,
        DepthPolicy::Adaptive(AdaptiveDepth::neural()),
    ] {
        for sampler in samplers() {
            let sequence = sequential(&prompt, &sampler, 16, 64, &[], None).0;
            for k in [1, 2, 3, 7] {
                for eos in [vec![sequence[0]], vec![sequence[3]], vec![sequence[7]]] {
                    let mut oracle = GuidedProposer {
                        sampler: sampler.clone(),
                        wrong_after: None,
                    };
                    check_with(
                        policy.clone(),
                        &prompt,
                        &sampler,
                        &mut oracle,
                        k,
                        (16, 64),
                        &eos,
                        None,
                    );
                    // An EOS guess cannot stop generation until the selector chooses it.
                    check_with(
                        policy.clone(),
                        &prompt,
                        &sampler,
                        &mut ScriptProposer::new(vec![eos[0]; k]),
                        k,
                        (16, 64),
                        &eos,
                        None,
                    );
                }
                for cancel in [1, 3, 6, 11] {
                    let mut oracle = GuidedProposer {
                        sampler: sampler.clone(),
                        wrong_after: None,
                    };
                    check_with(
                        policy.clone(),
                        &prompt,
                        &sampler,
                        &mut oracle,
                        k,
                        (16, 64),
                        &[],
                        Some(cancel),
                    );
                }
                for capacity in [prompt.len() - 1, prompt.len(), prompt.len() + 3] {
                    let mut oracle = GuidedProposer {
                        sampler: sampler.clone(),
                        wrong_after: None,
                    };
                    check_with(
                        policy.clone(),
                        &prompt,
                        &sampler,
                        &mut oracle,
                        k,
                        (16, capacity),
                        &[],
                        None,
                    );
                }
            }
        }
    }
}

#[test]
fn ngram_uses_most_recent_match_and_only_known_continuation() {
    let mut proposer = NgramProposer::new(2, 7).unwrap();
    assert_eq!(proposer.propose(&[1, 2, 3, 1, 2, 4, 1], 2, 7), [4, 1, 2]);
    assert_eq!(proposer.propose(&[1, 2, 3, 1, 2, 4, 1], 2, 1), [4]);
    assert_eq!(proposer.propose(&[1, 1], 1, 7), [1]);
    assert!(proposer.propose(&[1], 2, 7).is_empty());
    assert!(proposer.propose(&[1, 2, 3], 4, 7).is_empty());
    assert!(NgramProposer::new(0, 7).is_err());
    assert!(
        NgramProposer::new(1, 0)
            .unwrap()
            .propose(&[1], 1, 7)
            .is_empty()
    );
}

#[test]
fn long_ngram_requires_the_match_to_extend_backwards() {
    let proposer = NgramProposer::new(2, 7).unwrap();
    let history = [7, 5, 1, 2, 9, 7, 5, 1];
    assert_eq!(proposer.propose_long(&history, 2, 7, 4), [9, 7, 5, 1, 2]);
    assert_eq!(proposer.propose_long(&history, 2, 3, 0), [9, 7, 5]);
    assert!(proposer.propose_long(&history, 2, 7, 5).is_empty());
    // The most recent match decides, as for `propose`.
    assert!(
        proposer
            .propose_long(&[1, 2, 3, 1, 2, 4, 1], 2, 7, 3)
            .is_empty()
    );
}

#[test]
fn free_guesses_bypass_the_adaptive_policy_and_stay_lossless() {
    struct Free {
        oracle: GuidedProposer,
        calls: usize,
    }
    impl Proposer for Free {
        fn propose(&mut self, _: &[u32], _: u32, _: usize) -> Vec<u32> {
            self.calls += 1;
            Vec::new()
        }
        fn free_guesses(&mut self, history: &[u32], last: u32, max: usize) -> Vec<u32> {
            self.oracle.propose(history, last, max)
        }
    }
    let prompt = [1, 2, 3, 1, 2, 3, 1];
    for sampler in samplers() {
        for wrong_after in [None, Some(0)] {
            let mut proposer = Free {
                oracle: GuidedProposer {
                    sampler: sampler.clone(),
                    wrong_after,
                },
                calls: 0,
            };
            let policy = DepthPolicy::Adaptive(AdaptiveDepth::neural());
            let result = check_with(
                policy,
                &prompt,
                &sampler,
                &mut proposer,
                5,
                (30, 64),
                &[],
                None,
            );
            // Free guesses are always offered in full; the paid proposer never runs.
            assert_eq!(proposer.calls, 0);
            assert_eq!(result.statistics.proposals, 0);
            assert_eq!(
                result.statistics.guesses_offered,
                result.statistics.rows_verified - result.statistics.passes
            );
            if wrong_after.is_none() {
                assert_eq!(result.statistics.passes, 5);
            }
        }
    }
}

#[test]
fn caller_capacity_and_pass_width_bound_every_verification() {
    for depth in [
        DepthPolicy::Fixed,
        DepthPolicy::Adaptive(AdaptiveDepth::neural()),
    ] {
        let mut verifier = SequenceVerifier::new(&[1, 2], 64, 2);
        let mut proposer = GuidedProposer {
            sampler: Sampler::Greedy,
            wrong_after: None,
        };
        let mut limits = GenerationLimits::new(30, 7);
        limits.max_draft_tokens = usize::MAX;
        limits.depth = depth;
        let result = generate(
            &mut verifier,
            &mut proposer,
            &mut Greedy,
            &[1, 2, 3],
            limits,
        )
        .unwrap();
        assert_eq!(result.stop_reason, StopReason::ContextCapacity);
        assert_eq!(
            result.tokens,
            sequential(&[1, 2, 3], &Sampler::Greedy, 30, 7, &[], None).0
        );
        assert_eq!(verifier.position(), 7);
        assert_eq!(result.statistics.tokens_confirmed_per_pass, [2, 2, 1]);
        assert_eq!(result.statistics.rows_verified, 5);
        assert_eq!(result.statistics.guesses_offered, 2);
    }
}

#[test]
fn adaptive_controller_allocation_respects_output_and_context_without_width_cap() {
    for (n, capacity) in [(0, usize::MAX), (1, usize::MAX), (30, 3), (30, 2)] {
        let mut verifier = SequenceVerifier::new(&[1, 2], 64, usize::MAX);
        let mut limits = GenerationLimits::new(n, capacity);
        limits.max_draft_tokens = usize::MAX;
        limits.depth = DepthPolicy::Adaptive(AdaptiveDepth::neural());
        let result = generate(
            &mut verifier,
            &mut NoProposer,
            &mut Greedy,
            &[1, 2, 3],
            limits,
        )
        .unwrap();
        let expected = sequential(&[1, 2, 3], &Sampler::Greedy, n, capacity.min(64), &[], None);
        assert_eq!(result.tokens, expected.0);
        assert_eq!(result.stop_reason, expected.2);
        assert!(result.statistics.requested_depths.iter().all(|&d| d == 0));
    }
}

#[test]
fn no_tokens_does_not_run_a_pass_and_selector_failure_discards_candidates() {
    let mut verifier = SequenceVerifier::new(&[1, 2], 64, 9);
    let result = generate(
        &mut verifier,
        &mut NoProposer,
        &mut Greedy,
        &[3],
        GenerationLimits::new(0, 64),
    )
    .unwrap();
    assert!(result.tokens.is_empty());
    assert_eq!(verifier.committed, [1, 2]);
    assert_eq!(result.statistics.passes, 0);
    assert!(
        generate(
            &mut verifier,
            &mut NoProposer,
            &mut Greedy,
            &[],
            GenerationLimits::new(1, 64)
        )
        .is_err()
    );
    struct Failing;
    impl Selector for Failing {
        fn select(&mut self, _logits: &[f32], _position: u64) -> Result<u32> {
            Err(Error::InvalidInput("selection failed".into()))
        }
    }
    let error = generate(
        &mut verifier,
        &mut NoProposer,
        &mut Failing,
        &[3],
        GenerationLimits::new(5, 64),
    )
    .unwrap_err();
    assert!(matches!(error, GenerationError::Selector(_)));
    assert_eq!(verifier.committed, [1, 2]);
    assert!(verifier.pending.is_none());
    assert_eq!(verifier.logits(), Some(prediction(&[1, 2]).as_slice()));
}
