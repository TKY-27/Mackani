//! Parallel shortlist scoring and exact full-width candidate selection.
use super::*;

impl FastDraft {
    /// Reduced head for the seven rows, then sequential correction and choice.
    /// Per-row failures are reported in the reference's row/stage order.
    pub(super) fn select(
        &mut self,
        hidden: &[f32],
        frontier: u32,
        choice: DraftChoice,
    ) -> Result<DraftProposal> {
        let (h, v, count) = (
            self.dimensions.hidden,
            self.dimensions.vocabulary,
            self.options.shortlist,
        );
        let rows: Vec<&[f32]> = hidden.chunks_exact(h).collect();
        let n = rows.len();
        let simd = self.simd;
        let mut errors: Vec<Option<Error>> = rows
            .iter()
            .map(|x| {
                x.iter()
                    .any(|v| !v.is_finite())
                    .then_some(Error::NonFinite("draft head input"))
            })
            .collect();
        let mut clock = Instant::now();
        let deinterleaved: Vec<_> = rows.iter().map(|x| draft_math::deinterleave(x)).collect();
        let projection = &self.head.projection;
        let chunks = self.compute.map_range(0..RANK / RANK_TASK, |chunk| {
            let mut out = vec![0.0; n * RANK_TASK];
            for k in 0..RANK_TASK {
                let weights = &projection[(chunk * RANK_TASK + k) * h..][..h];
                for (r, x) in rows.iter().enumerate() {
                    out[r * RANK_TASK + k] =
                        draft_math::low_rank(x, &deinterleaved[r], weights, simd);
                }
            }
            Ok(out)
        })?;
        let mut low = vec![0.0; n * RANK];
        for (chunk, values) in chunks.iter().enumerate() {
            for r in 0..n {
                low[r * RANK + chunk * RANK_TASK..][..RANK_TASK]
                    .copy_from_slice(&values[r * RANK_TASK..(r + 1) * RANK_TASK]);
            }
        }
        for (r, error) in errors.iter_mut().enumerate() {
            if error.is_none() && low[r * RANK..(r + 1) * RANK].iter().any(|v| !v.is_finite()) {
                *error = Some(Error::NonFinite("draft low-rank projection"));
            }
        }
        self.timing.head_projection += lap(&mut clock);
        let tiles = v.div_ceil(TILE);
        let (packed, scales) = (&self.head.approximate, &self.head.approximate_scale);
        let low = &low;
        let parts = self
            .compute
            .map_range(0..tiles.div_ceil(SCORE_TILES), |task| {
                let range = task * SCORE_TILES..((task + 1) * SCORE_TILES).min(tiles);
                Ok(draft_math::approximate_scores(
                    low, packed, scales, range, simd,
                ))
            })?;
        let mut scores = vec![Vec::with_capacity(v); n];
        for part in &parts {
            let tokens = part.len() / n;
            for (r, scores) in scores.iter_mut().enumerate() {
                scores.extend_from_slice(&part[r * tokens..(r + 1) * tokens]);
            }
        }
        drop(parts);
        self.timing.head_scores += lap(&mut clock);
        let scores = &scores;
        let shortlists = self
            .compute
            .map_range(0..n, |r| Ok(shortlist(&scores[r], count)))?;
        let mut ids = Vec::with_capacity(n);
        for (r, list) in shortlists.into_iter().enumerate() {
            match list {
                Ok(list) if errors[r].is_none() => ids.push(list),
                Ok(_) => ids.push(Vec::new()),
                Err(error) => {
                    errors[r].get_or_insert(error);
                    ids.push(Vec::new());
                }
            }
        }
        self.timing.head_shortlist += lap(&mut clock);
        let (head, ids) = (&self.head, &ids);
        let tasks = count.div_ceil(RESCORE_TASK);
        let rescored = self.compute.map_range(0..n * tasks, |index| {
            let (r, task) = (index / tasks, index % tasks);
            let list = ids[r].get(task * RESCORE_TASK..).unwrap_or(&[]);
            Ok(list
                .iter()
                .take(RESCORE_TASK)
                .map(|&token| Candidate {
                    token,
                    score: draft_math::rescore(
                        rows[r],
                        &deinterleaved[r],
                        head.rescore_row(token as usize),
                        simd,
                    ) * head.rescore_scale[token as usize],
                })
                .collect::<Vec<_>>())
        })?;
        let mut top = Vec::with_capacity(n);
        for (r, chunk) in rescored.chunks(tasks).enumerate() {
            if errors[r].is_some() {
                top.push(Vec::new());
                continue;
            }
            match top16(chunk.concat()) {
                Ok(list) => top.push(list),
                Err(error) => {
                    errors[r] = Some(error);
                    top.push(Vec::new());
                }
            }
        }
        self.timing.head_rescore += lap(&mut clock);
        let (hp, rank) = (&self.correction_projection, self.dimensions.correction_rank);
        let features = self.compute.map_range(0..n, |r| {
            Ok(draft_math::correction_feature(rows[r], hp, rank))
        })?;
        let mut result = DraftProposal {
            tokens: Vec::new(),
            candidates: Vec::new(),
            corrected: Vec::new(),
        };
        let mut predecessor = frontier;
        let position = self.position();
        for (j, ((error, candidates), feature)) in
            errors.into_iter().zip(top).zip(features).enumerate()
        {
            if let Some(error) = error {
                return Err(error);
            }
            let feature = feature?;
            let pred = self.predecessor_vectors.row(predecessor as usize);
            let succ: Vec<_> = candidates
                .iter()
                .map(|c| self.successor_vectors.row(c.token as usize))
                .collect();
            let succ: Vec<&[f32]> = succ.iter().map(Vec::as_slice).collect();
            let corrected = correct_candidates(&feature, &pred, &succ, &candidates)?;
            let token = choose(&corrected, choice, (position + 1 + j) as u64)?;
            result.tokens.push(token);
            result.candidates.push(candidates);
            result.corrected.push(corrected);
            predecessor = token;
        }
        self.timing.head_select += lap(&mut clock);
        Ok(result)
    }
}
