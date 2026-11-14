use mackani_engine::{FastDraft, Model, Snapshot};
use mackani_ref::{
    draft_head::DraftChoice,
    model::CapturedRow,
    speculation::{
        CombinedProposer, DraftProposer, DraftWork, NgramProposer, NoProposer, ProposalSource,
        Proposer, SessionSnapshot, Verifier,
    },
};

use crate::{Error, Result};

type Neural = DraftProposer<FastDraft>;
pub(super) enum Drafts {
    None(NoProposer),
    Ngram(NgramProposer),
    Neural(Box<Neural>),
    Both(Box<CombinedProposer<FastDraft>>, ProposalSource),
}
impl Drafts {
    pub(super) fn work(&self) -> DraftWork {
        match self {
            Self::Neural(p) => p.work(),
            Self::Both(p, _) => p.draft().work(),
            _ => DraftWork::default(),
        }
    }
    pub(super) fn reset(&mut self) {
        match self {
            Self::Neural(p) => p.reset(),
            Self::Both(p, _) => p.reset(),
            _ => {}
        }
    }
    pub(super) fn choice(&mut self, choice: DraftChoice) {
        match self {
            Self::Neural(p) => p.set_choice(choice),
            Self::Both(p, _) => p.set_choice(choice),
            _ => {}
        }
    }
    pub(super) fn snapshot(&mut self, main: &mut Model) -> Result<Checkpoint> {
        match self {
            Self::Neural(p) => p
                .snapshot_session(main)
                .map(Checkpoint::Joint)
                .map_err(Error::engine),
            Self::Both(p, _) => p
                .snapshot_session(main)
                .map(Checkpoint::Joint)
                .map_err(Error::engine),
            _ => main.snapshot().map(Checkpoint::Main).map_err(Error::engine),
        }
    }
    pub(super) fn restore(&mut self, main: &mut Model, snapshot: &Checkpoint) -> Result<()> {
        match (self, snapshot) {
            (Self::Neural(p), Checkpoint::Joint(s)) => {
                p.restore_session(main, s).map_err(Error::engine)
            }
            (Self::Both(p, _), Checkpoint::Joint(s)) => {
                p.restore_session(main, s).map_err(Error::engine)
            }
            (Self::None(_) | Self::Ngram(_), Checkpoint::Main(s)) => {
                main.restore(s).map_err(Error::engine)
            }
            _ => Err(Error::engine("checkpoint proposer mismatch")),
        }
    }
}
impl Proposer for Drafts {
    fn proposal_source(&self) -> ProposalSource {
        match self {
            Self::None(_) => ProposalSource::Unspecified,
            Self::Ngram(_) => ProposalSource::Ngram,
            Self::Neural(_) => ProposalSource::Neural,
            Self::Both(_, source) => *source,
        }
    }
    fn propose(&mut self, h: &[u32], last: u32, max: usize) -> Vec<u32> {
        match self {
            Self::None(p) => p.propose(h, last, max),
            Self::Ngram(p) => p.propose(h, last, max),
            Self::Neural(p) => p.propose(h, last, max),
            Self::Both(p, source) => {
                *source = ProposalSource::Ngram;
                p.propose(h, last, max)
            }
        }
    }
    fn free_guesses(&mut self, h: &[u32], last: u32, max: usize) -> Vec<u32> {
        match self {
            Self::None(p) => p.free_guesses(h, last, max),
            Self::Ngram(p) => p.free_guesses(h, last, max),
            Self::Neural(p) => p.free_guesses(h, last, max),
            Self::Both(p, source) => {
                *source = ProposalSource::Ngram;
                p.free_guesses(h, last, max)
            }
        }
    }
    fn synchronize<V: Verifier>(&mut self, v: &mut V) -> mackani_ref::Result<()> {
        match self {
            Self::None(p) => p.synchronize(v),
            Self::Ngram(p) => p.synchronize(v),
            Self::Neural(p) => p.synchronize(v),
            Self::Both(p, _) => p.synchronize(v),
        }
    }
    fn observe_committed(&mut self, rows: &[CapturedRow]) -> mackani_ref::Result<()> {
        match self {
            Self::None(p) => p.observe_committed(rows),
            Self::Ngram(p) => p.observe_committed(rows),
            Self::Neural(p) => p.observe_committed(rows),
            Self::Both(p, _) => p.observe_committed(rows),
        }
    }
    fn propose_with<V: Verifier>(
        &mut self,
        v: &mut V,
        h: &[u32],
        last: u32,
        max: usize,
    ) -> mackani_ref::Result<Vec<u32>> {
        match self {
            Self::None(p) => p.propose_with(v, h, last, max),
            Self::Ngram(p) => p.propose_with(v, h, last, max),
            Self::Neural(p) => p.propose_with(v, h, last, max),
            Self::Both(p, source) => {
                // Combined proposals contain one source. Attribute the returned
                // guesses by the neural call actually made, not the configured mode.
                let before = p.draft().work().proposals;
                let guesses = p.propose_with(v, h, last, max)?;
                *source = if p.draft().work().proposals > before {
                    ProposalSource::Neural
                } else {
                    ProposalSource::Ngram
                };
                Ok(guesses)
            }
        }
    }
}
pub(super) enum Checkpoint {
    Main(Snapshot),
    Joint(SessionSnapshot<Snapshot>),
}
pub(super) struct Saved {
    pub(super) tokens: Vec<u32>,
    pub(super) checkpoint: Checkpoint,
}
