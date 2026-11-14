//! Scalar layer execution in the reference evaluation order.
use super::*;

impl Layer {
    pub(super) fn forward(
        &self,
        hidden: &[f32],
        state: &mut LayerState,
        position: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>> {
        let input_signs = match &self.branch {
            Branch::Attention(layer) => layer.query_gate.input_signs(),
            Branch::Recurrent(layer) => layer.query_key_value.input_signs(),
        };
        let input = prepare_normalization(hidden, &self.input_norm, input_signs, epsilon)?;
        let projected = match (&self.branch, state) {
            (Branch::Recurrent(layer), LayerState::Recurrent(state)) => {
                // qkv and z share qkv's prepared signs, even when their metadata differs.
                let query_key_value = layer
                    .query_key_value
                    .project(&input.prepared, Some(input.inverse_rms))?;
                let output_gate = layer
                    .output_gate
                    .project(&input.prepared, Some(input.inverse_rms))?;
                let gated = layer.recurrence.step(
                    state,
                    &query_key_value,
                    &output_gate,
                    &input.weighted_hidden_bf16,
                    input.inverse_rms,
                )?;
                let prepared = quantize_activation(&gated, layer.output.input_signs())?;
                layer.output.project(&prepared, None)?
            }
            (Branch::Attention(layer), LayerState::Attention(state)) => {
                // All three projections reuse q's input signs and the same integer scales.
                let query_gate = layer
                    .query_gate
                    .project(&input.prepared, Some(input.inverse_rms))?;
                let key = layer
                    .key
                    .project(&input.prepared, Some(input.inverse_rms))?;
                let value = layer
                    .value
                    .project(&input.prepared, Some(input.inverse_rms))?;
                let gated = layer
                    .attention
                    .step(state, position, &query_gate, &key, &value)?;
                let prepared = quantize_activation(&gated, layer.output.input_signs())?;
                layer.output.project(&prepared, None)?
            }
            _ => {
                return Err(Error::InvalidModel(
                    "layer and persistent state types differ".into(),
                ));
            }
        };
        let hidden = residual_add(hidden, &projected)?;
        let input = prepare_normalization(
            &hidden,
            &self.post_attention_norm,
            self.gate.input_signs(),
            epsilon,
        )?;
        let gate = self
            .gate
            .project(&input.prepared, Some(input.inverse_rms))?;
        let up = self.up.project(&input.prepared, Some(input.inverse_rms))?;
        let product = gated_product(&gate, &up)?;
        let prepared = quantize_activation(&product, self.down.input_signs())?;
        residual_add(&hidden, &self.down.project(&prepared, None)?)
    }
}

impl Model {
    pub(super) fn compute_rows(
        &self,
        tokens: &[u32],
        capture_limit: usize,
    ) -> Result<ComputedRows> {
        let end_position = self
            .position
            .checked_add(tokens.len())
            .ok_or_else(|| Error::InvalidInput("position overflow".into()))?;
        if end_position > self.context_capacity {
            return Err(Error::InvalidInput(
                "pass exceeds the session context capacity".into(),
            ));
        }
        if tokens
            .iter()
            .any(|&token| token as usize >= self.weights.dimensions.vocabulary_size)
        {
            return Err(Error::InvalidInput("token ID exceeds vocabulary".into()));
        }
        let mut states = self.layer_states.clone();
        let mut logits = Vec::with_capacity(tokens.len());
        let mut captures = Vec::new();
        let capture_start = tokens.len().saturating_sub(capture_limit);
        // A canonical row traversal defines batch/chunk invariance. Kernel dispatch
        // and upstream one-row precommit shortcuts are not part of this reference.
        for (row, &token) in tokens.iter().enumerate() {
            let mut hidden = self.weights.embeddings.row(token)?;
            let mut values = Vec::new();
            for (index, (layer, state)) in self.weights.layers.iter().zip(&mut states).enumerate() {
                hidden = layer.forward(
                    &hidden,
                    state,
                    self.position + row,
                    self.weights.dimensions.epsilon,
                )?;
                if row >= capture_start && self.capture_options.layers.contains(&index) {
                    values.extend_from_slice(&hidden);
                }
            }
            if !values.is_empty() {
                captures.push(CapturedRow {
                    position: self.position + row,
                    values,
                });
            }
            let input = prepare_normalization(
                &hidden,
                &self.weights.final_norm,
                self.weights.head.input_signs(),
                self.weights.dimensions.epsilon,
            )?;
            logits.push(output_head(&self.weights.head, &input)?);
        }
        Ok((states, logits, captures))
    }
}
