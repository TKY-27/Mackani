//! The MLP span from gate+up to down in one command buffer (`mlp.metal`).
use super::*;

/// Threads per threadgroup of the block kernels: two values of a 512-value
/// rotation block per thread.
const THREADS: usize = 256;

#[derive(Clone)]
pub(super) struct Pipelines {
    gated_product: Retained<Pipeline>,
    quantize: Retained<Pipeline>,
    scales: Retained<Pipeline>,
}

impl Pipelines {
    /// `None` when a block kernel's pipeline, or the test limit, allows fewer
    /// than 256 threads per threadgroup: the span then always runs unfused.
    pub(super) fn new(
        device: &Device,
        library: &ProtocolObject<dyn MTLLibrary>,
        max_threads: Option<usize>,
    ) -> Result<Option<Self>> {
        let pipeline = |name: &str| {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| metal_error("shader", format!("missing {name}")))?;
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| ns_error("pipeline", &e))
        };
        let pipelines = Self {
            gated_product: pipeline("mlp_gated_product")?,
            quantize: pipeline("mlp_quantize")?,
            scales: pipeline("mlp_scales")?,
        };
        let supported = [&pipelines.gated_product, &pipelines.quantize]
            .iter()
            .all(|p| p.maxTotalThreadsPerThreadgroup() >= THREADS)
            && max_threads.is_none_or(|n| n >= THREADS);
        Ok(supported.then_some(pipelines))
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    gate_rows: u32,
    rows: u32,
    groups: u32,
}

/// Per-tensor constants a fused span reads.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Constant {
    RowScale,
    OutputSigns,
    InputSigns,
}

/// Reusable buffers of [`MetalEngine::fused_mlp`], grown on demand.
#[derive(Default)]
pub(super) struct Scratch {
    row_inputs: Option<Retained<Buffer>>,
    product: Option<Retained<Buffer>>,
    activation: Option<Retained<Buffer>>,
    activation_scale: Option<Retained<Buffer>>,
    scales: Option<Retained<Buffer>>,
    row_quantum: Option<Retained<Buffer>>,
    inexact: Option<Retained<Buffer>>,
}

/// The down projection's sums of one fused MLP span.
pub struct FusedMlp {
    /// Row-major M×N sums, padded physical rows included.
    pub down: Vec<i64>,
    /// Each activation row's row quantum of the quantized gated product.
    pub row_quantum: Vec<f32>,
}

/// Copy `values` to the start of `buffer`.
///
/// # Safety
/// No GPU work and no other CPU reference may access the buffer's contents
/// during the call.
unsafe fn write<T: Copy>(buffer: &Buffer, values: &[T]) -> Result<()> {
    if buffer.length() < size_of_val(values) {
        return Err(metal_error("upload", "buffer too small"));
    }
    // SAFETY: the length was just checked, shared contents are page-aligned
    // and CPU-visible, and the caller guarantees exclusive access.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            buffer.contents().as_ptr().cast::<T>(),
            values.len(),
        );
    }
    Ok(())
}

/// The first `len` values of `buffer`.
///
/// # Safety
/// No GPU work may be writing the buffer during the call.
unsafe fn read<T: Copy>(buffer: &Buffer, len: usize) -> Result<Vec<T>> {
    if buffer.length() < len * size_of::<T>() {
        return Err(metal_error("readback", "buffer too small"));
    }
    // SAFETY: as for `write`; the caller guarantees no concurrent GPU write.
    Ok(unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<T>(), len) }.to_vec())
}

fn encode(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &Pipeline,
    buffers: &[&Buffer],
    params: &Params,
    threadgroups: (usize, usize),
    threads: usize,
) {
    encoder.setComputePipelineState(pipeline);
    // SAFETY: the caller retains every buffer until the command buffer
    // completes and sized them for the grid validated in `fused_mlp`. Params
    // is a plain repr(C) value copied by setBytes.
    unsafe {
        for (index, buffer) in buffers.iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
        }
        encoder.setBytes_length_atIndex(
            NonNull::from(params).cast(),
            size_of::<Params>(),
            buffers.len(),
        );
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: threadgroups.0,
            height: threadgroups.1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
}

impl MetalEngine {
    /// A tensor's constant buffer, keyed like the lookup tables: the mapping
    /// stays wrapped (and its address unused) until the engine is dropped, and
    /// each tensor has its own table offset.
    fn constant(&mut self, matrix: &Matrix, kind: Constant) -> Result<Retained<Buffer>> {
        let key = (
            matrix.mapping().as_ptr() as usize,
            matrix.descriptor().table_offset,
            kind,
        );
        if let Some(buffer) = self.constants.get(&key) {
            return Ok(buffer.clone());
        }
        let signs = |signs: &[i8]| signs.iter().map(|&s| s as u8).collect::<Vec<_>>();
        let bytes = match kind {
            Constant::RowScale => matrix
                .effective_row_scale()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect(),
            Constant::OutputSigns => signs(matrix.output_signs().unwrap_or_default()),
            Constant::InputSigns => signs(matrix.input_signs()),
        };
        let buffer = self.shared_buffer(&bytes)?;
        self.constants.insert(key, buffer.clone());
        Ok(buffer)
    }

    /// `bfloat(silu(x))` as BF16 bits for every BF16 `x`, from the reference.
    fn silu_table(&mut self) -> Result<Retained<Buffer>> {
        if let Some(table) = &self.silu {
            return Ok(table.clone());
        }
        let bytes: Vec<u8> = (0..=u16::MAX)
            .flat_map(|bits| {
                let value = mackani_ref::portable_math::silu(mackani_format::bf16_to_f32(bits));
                mackani_format::f32_to_bf16(value).to_ne_bytes()
            })
            .collect();
        let table = self.shared_buffer(&bytes)?;
        self.silu = Some(table.clone());
        Ok(table)
    }

    /// The MLP span of one normalized input in one command buffer: the gate and
    /// up projections of `activation` (M×K int8 rows with M×K/512 integer
    /// scales), their restore with each row's `row_quantum` and `inverse_rms`,
    /// inverse output rotation, the SiLU-gated BF16 product, its int8
    /// quantization and the down projection. `matrices` is [gate, up, down];
    /// gate and up must carry output rotations.
    ///
    /// The result equals the reference's: `Ok(None)` when the device cannot run
    /// the span, or when a value met on the GPU could round differently there (a
    /// subnormal, which the GPU flushes to zero) or makes the reference fail (a
    /// non-finite value). The caller then runs the stages unfused, which give
    /// the reference's result or error. Every row runs on the GPU.
    pub fn fused_mlp(
        &mut self,
        matrices: [&Matrix; 3],
        activation: &[i8],
        scales: &[i32],
        row_quantum: &[f32],
        inverse_rms: &[f32],
    ) -> Result<Option<FusedMlp>> {
        let [gate, up, down] = matrices;
        let (g, u, d) = (gate.descriptor(), up.descriptor(), down.descriptor());
        let m = row_quantum.len();
        let complete_rotation = |matrix: &Matrix| {
            matrix.output_signs().map(<[i8]>::len) == Some(matrix.descriptor().physical_rows)
        };
        if m == 0
            || inverse_rms.len() != m
            || activation.len() != m * g.cols
            || (g.rows, g.physical_rows, g.cols) != (u.rows, u.physical_rows, u.cols)
            || !g.physical_rows.is_multiple_of(512)
            || !complete_rotation(gate)
            || !complete_rotation(up)
            || d.cols != g.rows
            || down.input_signs().len() != d.cols
        {
            return Err(invalid("invalid fused MLP shapes"));
        }
        let Some(stages) = self.mlp.clone() else {
            return Ok(None);
        };
        let groups = d.cols / 512;
        let plans = [
            self.plan(gate, activation.len(), scales.len(), g.physical_rows)?,
            self.plan(up, activation.len(), scales.len(), u.physical_rows)?,
            self.plan(down, m * d.cols, m * groups, d.physical_rows)?,
        ];
        let constants = [
            self.constant(gate, Constant::RowScale)?,
            self.constant(up, Constant::RowScale)?,
            self.constant(gate, Constant::OutputSigns)?,
            self.constant(up, Constant::OutputSigns)?,
            self.constant(down, Constant::InputSigns)?,
        ];
        let silu = self.silu_table()?;
        let device = &self.device;
        let input = [
            reserve(device, &mut self.scratch_activation, activation.len() * 2)?,
            reserve(device, &mut self.scratch_scales, scales.len() * 4)?,
        ];
        if self.scratch_outputs.len() < plans.len() {
            self.scratch_outputs.resize(plans.len(), None);
        }
        let outputs = plans
            .iter()
            .zip(&mut self.scratch_outputs)
            .map(|(plan, slot)| reserve(device, slot, plan.output_len() * 8))
            .collect::<Result<Vec<_>>>()?;
        let s = &mut self.scratch_mlp;
        let row_inputs = reserve(device, &mut s.row_inputs, m * 8)?;
        let product = reserve(device, &mut s.product, m * d.cols * 4)?;
        let down_activation = reserve(device, &mut s.activation, m * d.cols * 2)?;
        let activation_scale = reserve(device, &mut s.activation_scale, m * groups * 4)?;
        let down_scales = reserve(device, &mut s.scales, m * groups * 4)?;
        let down_quantum = reserve(device, &mut s.row_quantum, m * 4)?;
        let inexact = reserve(device, &mut s.inexact, 4)?;
        let rows: Vec<[f32; 2]> = row_quantum
            .iter()
            .zip(inverse_rms)
            .map(|(&q, &r)| [q, r])
            .collect();
        // SAFETY: as in `shared`: these engine-owned buffers are borrowed through
        // `&mut self`, and no command buffer of this engine is in flight.
        unsafe {
            upload(&input[0], &input[1], activation, scales)?;
            write(&row_inputs, &rows)?;
            write(&inexact, &[0u32])?;
        }
        let params = Params {
            gate_rows: g.physical_rows as u32,
            rows: d.cols as u32,
            groups: groups as u32,
        };
        let in_flight = self.commit(|encoder| {
            for (plan, output) in plans[..2].iter().zip(&outputs) {
                plan.encode(encoder, &input[0], &input[1], output);
            }
            let [gate_scale, up_scale, gate_signs, up_signs, down_signs] = &constants;
            encode(
                encoder,
                &stages.gated_product,
                &[
                    &outputs[0],
                    &outputs[1],
                    gate_scale,
                    up_scale,
                    gate_signs,
                    up_signs,
                    &row_inputs,
                    &silu,
                    &product,
                    &inexact,
                ],
                &params,
                (g.physical_rows / 512, m),
                THREADS,
            );
            encode(
                encoder,
                &stages.quantize,
                &[
                    &product,
                    down_signs,
                    &down_activation,
                    &activation_scale,
                    &inexact,
                ],
                &params,
                (groups, m),
                THREADS,
            );
            encode(
                encoder,
                &stages.scales,
                &[&activation_scale, &down_scales, &down_quantum],
                &params,
                (m, 1),
                1,
            );
            plans[2].encode(encoder, &down_activation, &down_scales, &outputs[2]);
        })?;
        in_flight.wait()?;
        // SAFETY: `wait` returned Ok, so the command buffer has completed and
        // nothing writes these engine-owned buffers any more.
        unsafe {
            if read::<u32>(&inexact, 1)?[0] != 0 {
                return Ok(None);
            }
            Ok(Some(FusedMlp {
                down: download(&outputs[2], m, d.physical_rows, &[])?,
                row_quantum: read(&down_quantum, m)?,
            }))
        }
    }
}
