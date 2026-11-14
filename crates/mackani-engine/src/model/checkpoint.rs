//! Session reset and owned prefix checkpoints.
use super::*;

impl Model {
    /// Clear committed and provisional context while retaining model weights.
    pub fn reset(&mut self) -> Result<()> {
        self.states = self
            .weights
            .layers
            .iter()
            .map(Layer::new_state)
            .collect::<Result<_>>()?;
        self.position = 0;
        self.pending = false;
        self.working_complete = false;
        self.captures.clear();
        self.pending_captures.clear();
        self.pending_tokens.clear();
        self.tokens.clear();
        self.committed_logits = None;
        self.logits.clear();
        self.timings.clear();
        self.owner = Arc::new(());
        self.context_identity = Arc::new(());
        Ok(())
    }

    /// Capture owned state for restoring this session's committed prefix.
    pub fn snapshot(&self) -> Result<Snapshot> {
        self.require_no_pending_pass()?;
        Ok(Snapshot {
            owner: Arc::clone(&self.owner),
            position: self.position,
            tokens: self.tokens.clone(),
            recurrence: self
                .states
                .iter()
                .map(|state| match state {
                    LayerState::Attention(_) => None,
                    LayerState::Recurrent(state) => Some(state.clone()),
                })
                .collect(),
            committed_logits: self.committed_logits.clone(),
            captures: self.captures.clone(),
            capture_options: self.capture_options.clone(),
        })
    }

    /// Validate and restore an owned checkpoint for this session.
    pub fn restore(&mut self, snapshot: &Snapshot) -> Result<()> {
        self.require_no_pending_pass()?;
        if !Arc::ptr_eq(&snapshot.owner, &self.owner)
            || snapshot.position > self.position
            || !self.tokens.starts_with(&snapshot.tokens)
        {
            return Err(Error::InvalidInput(
                "snapshot is not a retained prefix of this session".into(),
            ));
        }
        if snapshot.capture_options != self.capture_options {
            return Err(Error::InvalidInput(
                "snapshot capture configuration changed".into(),
            ));
        }
        // Validate every cache before publishing any part of the checkpoint.
        for state in &self.states {
            if let LayerState::Attention(state) = state
                && state.len() < snapshot.position
            {
                return Err(Error::InvalidInput(
                    "snapshot attention prefix is unavailable".into(),
                ));
            }
        }
        for (state, saved) in self.states.iter_mut().zip(&snapshot.recurrence) {
            match (state, saved) {
                (LayerState::Attention(state), None) => state.truncate(snapshot.position)?,
                (LayerState::Recurrent(state), Some(saved)) => state.copy_from(saved),
                _ => unreachable!("snapshot belongs to the same model"),
            }
        }
        self.context_identity = Arc::new(());
        self.captures.clone_from(&snapshot.captures);
        self.position = snapshot.position;
        self.tokens.truncate(snapshot.position);
        self.committed_logits.clone_from(&snapshot.committed_logits);
        self.logits = self.committed_logits.iter().cloned().collect();
        self.timings.clear();
        Ok(())
    }
}
