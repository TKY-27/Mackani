//! Owned session checkpoints with atomic validation on restore.
use super::*;

impl Model {
    /// Capture a committed checkpoint without copying attention cache rows.
    pub fn snapshot(&self) -> Result<Snapshot> {
        self.require_no_pending_pass()?;
        Ok(Snapshot {
            session: Rc::clone(&self.session),
            position: self.position,
            tokens: self.committed_tokens.clone(),
            recurrent_states: self
                .layer_states
                .iter()
                .map(|state| match state {
                    LayerState::Attention(_) => None,
                    LayerState::Recurrent(state) => Some(state.clone()),
                })
                .collect(),
            committed_logits: self.committed_logits.clone(),
            capture_options: self.capture_options.clone(),
            captures: self.captures.clone(),
        })
    }

    /// Return to a checkpoint on this session's retained causal prefix.
    /// Invalid checkpoints leave all observable state unchanged.
    pub fn restore(&mut self, snapshot: &Snapshot) -> Result<()> {
        self.require_no_pending_pass()?;
        if !Rc::ptr_eq(&self.session, &snapshot.session) {
            return Err(Error::InvalidInput(
                "snapshot belongs to a different session or reset".into(),
            ));
        }
        if snapshot.capture_options != self.capture_options {
            return Err(Error::InvalidInput(
                "snapshot capture configuration changed".into(),
            ));
        }
        if snapshot.position > self.position {
            return Err(Error::InvalidInput(
                "snapshot is beyond the current position".into(),
            ));
        }
        if self.committed_tokens[..snapshot.position] != snapshot.tokens {
            return Err(Error::InvalidInput(
                "snapshot token prefix is no longer retained".into(),
            ));
        }
        // Validate every cache before mutating any layer, so restore remains atomic.
        for state in &self.layer_states {
            if let LayerState::Attention(state) = state {
                if state.len() < snapshot.position {
                    return Err(Error::InvalidInput(
                        "snapshot attention prefix is unavailable".into(),
                    ));
                }
            }
        }
        for (state, recurrent) in self.layer_states.iter_mut().zip(&snapshot.recurrent_states) {
            match (state, recurrent) {
                (LayerState::Attention(state), None) => state.truncate(snapshot.position)?,
                (LayerState::Recurrent(state), Some(saved)) => state.copy_from(saved),
                _ => unreachable!("checkpoint layer kinds are fixed within a session"),
            }
        }
        self.context_identity = Arc::new(());
        self.captures.clone_from(&snapshot.captures);
        self.provisional_captures.clear();
        self.position = snapshot.position;
        self.committed_tokens.truncate(snapshot.position);
        self.committed_logits.clone_from(&snapshot.committed_logits);
        self.last_pass_logits.clear();
        self.last_pass_logits
            .extend(self.committed_logits.iter().cloned());
        Ok(())
    }
}
