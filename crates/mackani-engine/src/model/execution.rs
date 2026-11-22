//! Ordered layer arithmetic and parallel row preparation.
use super::*;

fn normalize(
    engine: &mut Compute,
    rows: &[Vec<f32>],
    norm: &[f32],
    linear: &Linear,
    epsilon: f32,
    timing: &mut PassTiming,
) -> Result<Vec<NormalizedInput>> {
    let start = timing.detailed_start();
    let prepare_normalization = if engine.cpu_stages {
        prepare_normalization
    } else {
        mackani_ref::residual_norm_mlp::prepare_normalization
    };
    let output = engine.map_range(0..rows.len(), |i| {
        prepare_normalization(&rows[i], norm, linear.input_signs(), epsilon)
    })?;
    if let Some(start) = start {
        timing.normalization += start.elapsed();
    }
    Ok(output)
}
/// Projections that share one normalized input (and its first matrix's signs).
fn project_normalized<const N: usize>(
    engine: &mut Compute,
    linears: [&Linear; N],
    input: &[NormalizedInput],
    timing: &mut PassTiming,
) -> Result<[Vec<Vec<f32>>; N]> {
    let input: Vec<_> = input
        .iter()
        .map(|x| (&x.prepared, Some(x.inverse_rms)))
        .collect();
    project(engine, linears, &input, timing)
}
fn project<const N: usize>(
    engine: &mut Compute,
    linears: [&Linear; N],
    inputs: &[(&PreparedActivation, Option<f32>)],
    timing: &mut PassTiming,
) -> Result<[Vec<Vec<f32>>; N]> {
    let activation: Vec<_> = inputs
        .iter()
        .flat_map(|(p, _)| p.activation8().iter().copied())
        .collect();
    let scales: Vec<_> = inputs
        .iter()
        .flat_map(|(p, _)| p.integer_scale().iter().copied())
        .collect();
    let matrices = linears.map(|linear| &linear.matrix);
    let start = Instant::now();
    let sums = engine.weighted_sums(&matrices, &activation, &scales)?;
    timing.kernel += start.elapsed();
    let start = timing.detailed_start();
    let m = inputs.len();
    let restore_projection = if engine.cpu_stages {
        restore_projection
    } else {
        mackani_ref::compressed_linear::restore_projection
    };
    let mut rows = engine
        .map_range(0..N * m, |index| {
            let (linear, sums, i) = (linears[index / m], &sums[index / m], index % m);
            let n = linear.matrix.descriptor().physical_rows;
            restore_projection(
                &sums[i * n..(i + 1) * n],
                inputs[i].0,
                inputs[i].1,
                linear.matrix.effective_row_scale(),
                if linear.output_rotation {
                    linear.matrix.output_signs()
                } else {
                    None
                },
                linear.matrix.descriptor().rows,
            )
        })?
        .into_iter();
    if let Some(start) = start {
        timing.projection_restore += start.elapsed();
    }
    Ok(std::array::from_fn(|_| rows.by_ref().take(m).collect()))
}
fn project_rows(
    engine: &mut Compute,
    linear: &Linear,
    rows: &[Vec<f32>],
    timing: &mut PassTiming,
) -> Result<Vec<Vec<f32>>> {
    let start = timing.detailed_start();
    let prepared = engine.map_range(0..rows.len(), |i| {
        quantize_activation(&rows[i], linear.input_signs())
    })?;
    if let Some(start) = start {
        timing.activation_encode += start.elapsed();
    }
    let inputs: Vec<_> = prepared.iter().map(|p| (p, None)).collect();
    let [projected] = project(engine, [linear], &inputs, timing)?;
    Ok(projected)
}
/// The reference's `GatedDeltaNet::step` stages for a pass of rows: rows are
/// prepared in parallel (they read only earlier rows' rounded projections), then
/// each value head runs its rows in ascending order on one worker. On error the
/// state is partly updated; the committed recurrent state remains untouched.
#[allow(clippy::too_many_arguments)]
pub(super) fn recurrence(
    engine: &mut Compute,
    layer: &GatedDeltaNet,
    state: &mut GatedDeltaNetState,
    source: Option<&GatedDeltaNetState>,
    qkv: &[Vec<f32>],
    z: &[Vec<f32>],
    input: &[NormalizedInput],
    cache: Option<&mut [GatedDeltaNetRow]>,
    timing: &mut PassTiming,
) -> Result<Vec<Vec<f32>>> {
    if engine.cpu_stages {
        prepare_recurrence::<crate::cpu_stages::math::FastMath>(
            engine, layer, state, source, qkv, z, input, cache, timing,
        )
    } else {
        prepare_recurrence::<mackani_ref::portable_math::ScalarMath>(
            engine, layer, state, source, qkv, z, input, cache, timing,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_recurrence<M: mackani_ref::portable_math::Elementwise>(
    engine: &mut Compute,
    layer: &GatedDeltaNet,
    state: &mut GatedDeltaNetState,
    source: Option<&GatedDeltaNetState>,
    qkv: &[Vec<f32>],
    z: &[Vec<f32>],
    input: &[NormalizedInput],
    cache: Option<&mut [GatedDeltaNetRow]>,
    timing: &mut PassTiming,
) -> Result<Vec<Vec<f32>>> {
    let dense_gate = if engine.cpu_stages {
        crate::cpu_stages::recurrence::dense_gate
    } else {
        mackani_ref::gated_deltanet::dense_gate
    };
    let prepare_start = timing.detailed_start();
    let detailed = timing.detailed;
    let gate_nanos = std::sync::atomic::AtomicU64::new(0);
    let rounded: Vec<_> = qkv.iter().map(|row| round_projection(row)).collect();
    let rows = {
        let taps: Vec<&[f32]> = state
            .history_rows()
            .chain(rounded.iter().map(Vec::as_slice))
            .collect();
        let width = taps.len() - rounded.len();
        if engine.cpu_stages && qkv.len() == 1 {
            vec![layer.prepare_with_math::<M>(
                &taps[..width],
                &qkv[0],
                &z[0],
                &input[0].weighted_hidden_bf16,
                input[0].inverse_rms,
                |weights, hidden, rms| {
                    let start = detailed.then(Instant::now);
                    let gates = engine.map_range(0..weights.len() / hidden.len(), |head| {
                        dense_gate(
                            &weights[head * hidden.len()..(head + 1) * hidden.len()],
                            hidden,
                            rms,
                        )
                    });
                    if let Some(start) = start {
                        gate_nanos.fetch_add(
                            start.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                    gates
                },
            )?]
        } else {
            engine.map_range(0..qkv.len(), |i| {
                layer.prepare_with_math::<M>(
                    &taps[i..i + width],
                    &qkv[i],
                    &z[i],
                    &input[i].weighted_hidden_bf16,
                    input[i].inverse_rms,
                    |weights, hidden, rms| {
                        let start = detailed.then(Instant::now);
                        let gates = weights
                            .chunks_exact(hidden.len())
                            .map(|weight| dense_gate(weight, hidden, rms))
                            .collect();
                        if let Some(start) = start {
                            gate_nanos.fetch_add(
                                start.elapsed().as_nanos() as u64,
                                std::sync::atomic::Ordering::Relaxed,
                            );
                        }
                        gates
                    },
                )
            })?
        }
    };
    if let Some(start) = prepare_start {
        timing.recurrence_prepare += start.elapsed();
        timing.dense_gate_work +=
            Duration::from_nanos(gate_nanos.load(std::sync::atomic::Ordering::Relaxed));
    }
    if let Some(cache) = cache {
        for (slot, row) in cache.iter_mut().zip(&rows) {
            slot.copy_from(row);
        }
    }
    replay_recurrence(engine, layer, state, source, &rows)
}

pub(super) fn replay_recurrence(
    engine: &mut Compute,
    layer: &GatedDeltaNet,
    state: &mut GatedDeltaNetState,
    source: Option<&GatedDeltaNetState>,
    rows: &[GatedDeltaNetRow],
) -> Result<Vec<Vec<f32>>> {
    let fast = engine.cpu_stages;
    let heads: Vec<Mutex<&mut [f32]>> = state.recurrent_heads_mut().map(Mutex::new).collect();
    let sources = source.map(|s| s.recurrent_heads().collect::<Vec<_>>());
    let outputs = engine.map_range(0..heads.len(), |head| {
        let mut head_state = heads[head]
            .lock()
            .map_err(|_| Error::InvalidInput("recurrent head lock poisoned".into()))?;
        rows.iter()
            .enumerate()
            .map(|(i, row)| {
                if !fast {
                    if i == 0
                        && let Some(sources) = &sources
                    {
                        head_state.copy_from_slice(sources[head]);
                    }
                    return layer.recur_head(&mut head_state, row, head);
                }
                crate::cpu_stages::recurrence::recur_head(
                    layer,
                    &mut head_state,
                    if i == 0 {
                        sources.as_ref().map(|s| s[head])
                    } else {
                        None
                    },
                    row,
                    head,
                )
            })
            .collect::<Result<Vec<_>>>()
    })?;
    drop(heads);
    for row in rows {
        layer.advance_history(state, row)?;
    }
    Ok((0..rows.len())
        .map(|i| {
            outputs
                .iter()
                .flat_map(|head| head[i].iter().copied())
                .collect()
        })
        .collect())
}
/// Prepare, append and attend a batch of projected rows using the engine pool.
pub(super) fn attention_pass(
    engine: &mut Compute,
    attention: &mackani_ref::attention::Attention,
    state: &mut mackani_ref::attention::AttentionState,
    [q, k, v]: [&[Vec<f32>]; 3],
    timing: &mut PassTiming,
) -> Result<Vec<Vec<f32>>> {
    let position = timing.position;
    let start = Instant::now();
    let prepared = if engine.cpu_stages {
        let parallel = (|| {
            use mackani_ref::attention::RopeTable;
            enum Task {
                Side(mackani_ref::kv_codec::CacheRow),
                Query(mackani_ref::attention::QueryHead),
            }
            let ropes = (0..q.len())
                .map(|row| RopeTable::new(position + row))
                .collect::<Result<Vec<_>>>()?;
            // Per row: interleaved key/value sides in head order, then query heads in order.
            let sides_per_row = attention.kv_heads() * 2;
            let tasks_per_row = sides_per_row + attention.query_heads();
            let tasks = engine.map_range(0..q.len() * tasks_per_row, |index| {
                let row = index / tasks_per_row;
                let task = index % tasks_per_row;
                if task < sides_per_row {
                    let side = task % 2;
                    attention
                        .prepare_cache_side_with_search(
                            &ropes[row],
                            task / 2,
                            side,
                            if side == 0 { &k[row] } else { &v[row] },
                            crate::cpu_stages::kv_codec::search_chain,
                        )
                        .map(Task::Side)
                } else {
                    attention
                        .prepare_query_head(&ropes[row], task - sides_per_row, &q[row])
                        .map(Task::Query)
                }
            })?;
            let mut tasks = tasks.into_iter();
            ropes
                .iter()
                .map(|rope| {
                    let mut sides = Vec::with_capacity(sides_per_row);
                    let mut heads = Vec::with_capacity(tasks_per_row - sides_per_row);
                    for task in tasks.by_ref().take(tasks_per_row) {
                        match task {
                            Task::Side(side) => sides.push(side),
                            Task::Query(head) => heads.push(head),
                        }
                    }
                    let queries = attention.finish_query_heads(rope, heads)?;
                    attention.finish_prepared(queries, sides)
                })
                .collect::<Result<Vec<_>>>()
        })();
        match parallel {
            Ok(prepared) => prepared,
            Err(error) => {
                // On failure, replay in scalar row/stage order before any cache append.
                // Success retains independent KV-side tasks and never repeats preparation.
                for i in 0..q.len() {
                    attention.prepare_row(position + i, &q[i], &k[i], &v[i])?;
                }
                return Err(error);
            }
        }
    } else {
        engine.map_range(0..q.len(), |i| {
            attention.prepare_row(position + i, &q[i], &k[i], &v[i])
        })?
    };
    timing.kv_encode += start.elapsed();
    let start = Instant::now();
    for (i, p) in prepared.iter().enumerate() {
        attention.append_prepared(state, position + i, p)?;
    }
    let heads = attention.query_heads();
    let outputs =
        crate::cpu_stages::attention::output_rows(engine, attention, state, position, &prepared)?;
    let gated: Vec<_> = outputs
        .chunks(heads)
        .map(|heads| heads.iter().flatten().copied().collect())
        .collect();
    timing.attention += start.elapsed();
    Ok(gated)
}

#[allow(clippy::too_many_arguments)]
fn forward(
    engine: &mut Compute,
    layer: &Layer<Linear>,
    state: &mut LayerState,
    source: Option<&GatedDeltaNetState>,
    hidden: &[Vec<f32>],
    epsilon: f32,
    timing: &mut PassTiming,
    cache: Option<&mut [GatedDeltaNetRow]>,
) -> Result<Vec<Vec<f32>>> {
    let residual_add = if engine.cpu_stages {
        residual_add
    } else {
        mackani_ref::residual_norm_mlp::residual_add
    };
    let gated_product = if engine.cpu_stages {
        gated_product
    } else {
        mackani_ref::residual_norm_mlp::gated_product
    };
    let input_projection = match &layer.branch {
        Branch::Attention(l) => &l.query_gate,
        Branch::Recurrent(l) => &l.query_key_value,
    };
    let input = normalize(
        engine,
        hidden,
        &layer.input_norm,
        input_projection,
        epsilon,
        timing,
    )?;
    let projected = match (&layer.branch, state) {
        (Branch::Recurrent(l), LayerState::Recurrent(state)) => {
            let [qkv, z] =
                project_normalized(engine, [&l.query_key_value, &l.output_gate], &input, timing)?;
            let start = Instant::now();
            let gated = recurrence(
                engine,
                &l.recurrence,
                state,
                source,
                &qkv,
                &z,
                &input,
                cache,
                timing,
            )?;
            timing.recurrence += start.elapsed();
            project_rows(engine, &l.output, &gated, timing)?
        }
        (Branch::Attention(l), LayerState::Attention(state)) => {
            let [q, k, v] =
                project_normalized(engine, [&l.query_gate, &l.key, &l.value], &input, timing)?;
            let gated = attention_pass(engine, &l.attention, state, [&q, &k, &v], timing)?;
            project_rows(engine, &l.output, &gated, timing)?
        }
        _ => return Err(Error::InvalidModel("layer/state types differ".into())),
    };
    let start = timing.detailed_start();
    let hidden = engine.map_range(0..hidden.len(), |i| residual_add(&hidden[i], &projected[i]))?;
    if let Some(start) = start {
        timing.residual += start.elapsed();
    }
    let input = normalize(
        engine,
        &hidden,
        &layer.post_attention_norm,
        &layer.gate,
        epsilon,
        timing,
    )?;
    let [gate, up] = project_normalized(engine, [&layer.gate, &layer.up], &input, timing)?;
    let start = timing.detailed_start();
    let product = engine.map_range(0..hidden.len(), |i| gated_product(&gate[i], &up[i]))?;
    if let Some(start) = start {
        timing.mlp_gate += start.elapsed();
    }
    let down = project_rows(engine, &layer.down, &product, timing)?;
    let start = timing.detailed_start();
    let result = engine.map_range(0..hidden.len(), |i| residual_add(&hidden[i], &down[i]))?;
    if let Some(start) = start {
        timing.residual += start.elapsed();
    }
    Ok(result)
}

impl Model {
    pub(super) fn compute_pass(
        &mut self,
        tokens: &[u32],
        position: usize,
        requested: LogitRows,
        cache_replay: bool,
        capture_from: usize,
    ) -> Result<(Vec<Vec<f32>>, PassTiming, Vec<CapturedRow>)> {
        let start = Instant::now();
        let mut timing = PassTiming {
            position,
            rows: tokens.len(),
            detailed: self.detailed_timing,
            ..Default::default()
        };
        let mut hidden = self
            .engine
            .map_range(0..tokens.len(), |i| self.weights.embeddings.row(tokens[i]))?;
        let capture_start = if cache_replay {
            0
        } else {
            tokens.len().saturating_sub(self.capture_options.max_rows)
        }
        .max(capture_from.saturating_sub(position).min(tokens.len()));
        let mut captures = if self.capture_options.layers.is_empty() {
            Vec::new()
        } else {
            (capture_start..tokens.len())
                .map(|row| CapturedRow {
                    position: position + row,
                    values: Vec::with_capacity(
                        self.capture_options.layers.len() * self.weights.dimensions.hidden_width,
                    ),
                })
                .collect::<Vec<_>>()
        };
        for (i, layer) in self.weights.layers.iter().enumerate() {
            let (state, source) = match layer.branch {
                Branch::Attention(_) => (&mut self.states[i], None),
                Branch::Recurrent(_) => (
                    &mut self.working[i],
                    if self.engine.cpu_stages && position == self.position {
                        if let LayerState::Recurrent(source) = &self.states[i] {
                            Some(source)
                        } else {
                            None
                        }
                    } else {
                        None
                    },
                ),
            };
            hidden = forward(
                &mut self.engine,
                layer,
                state,
                source,
                &hidden,
                self.weights.dimensions.epsilon,
                &mut timing,
                if cache_replay {
                    Some(&mut self.replay[i])
                } else {
                    None
                },
            )?;
            if self.capture_options.layers.binary_search(&i).is_ok() {
                for (capture, row) in captures.iter_mut().zip(&hidden[capture_start..]) {
                    capture.values.extend_from_slice(row);
                }
            }
        }
        let selected = match requested {
            LogitRows::All => hidden.as_slice(),
            LogitRows::Last => &hidden[hidden.len() - 1..],
            LogitRows::None => &[],
        };
        let head = if selected.is_empty() {
            Vec::new()
        } else {
            let input = normalize(
                &mut self.engine,
                selected,
                &self.weights.final_norm,
                &self.weights.head,
                self.weights.dimensions.epsilon,
                &mut timing,
            )?;
            let [head] =
                project_normalized(&mut self.engine, [&self.weights.head], &input, &mut timing)?;
            head
        };
        timing.total = start.elapsed();
        Ok((head, timing, captures))
    }
}
