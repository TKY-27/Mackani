//! Shared buffers, command encoding and completion lifetimes.
use super::*;

/// Encode activation rows as exact binary16 integers and copy the
/// integer scales into shared buffers.
///
/// # Safety
/// No GPU work and no other CPU reference may access either buffer's contents
/// during the call.
pub(super) unsafe fn upload(
    activation_buffer: &Buffer,
    scales_buffer: &Buffer,
    activation: &[i8],
    scales: &[i32],
) -> Result<()> {
    if activation_buffer.length() < activation.len() * 2
        || scales_buffer.length() < scales.len() * 4
    {
        return Err(metal_error("upload", "input buffer too small"));
    }
    // SAFETY: the lengths were just checked against the buffers, whose shared
    // contents are page-aligned and CPU-visible; the caller guarantees exclusive
    // access, so the mutable slice is not aliased.
    unsafe {
        let wide = std::slice::from_raw_parts_mut(
            activation_buffer.contents().as_ptr().cast::<u16>(),
            activation.len(),
        );
        let codes = &*HALF;
        for (wide, &value) in wide.iter_mut().zip(activation) {
            *wide = codes[value as u8 as usize];
        }
        std::ptr::copy_nonoverlapping(
            scales.as_ptr(),
            scales_buffer.contents().as_ptr().cast::<i32>(),
            scales.len(),
        );
    }
    Ok(())
}

/// Row-major M×(g + c) sums: each activation row's `g` GPU sums from the shared
/// output buffer (M×g), followed by its `c` CPU sums from `cpu` (M×c, may be empty).
///
/// # Safety
/// No GPU work may be writing the buffer during the call.
pub(super) unsafe fn download(
    output: &Buffer,
    inputs: usize,
    g: usize,
    cpu: &[i64],
) -> Result<Vec<i64>> {
    if inputs == 0 || output.length() < inputs * g * 8 || !cpu.len().is_multiple_of(inputs) {
        return Err(metal_error("readback", "output buffer too small"));
    }
    let c = cpu.len() / inputs;
    // SAFETY: the buffer holds at least M×g i64 values (checked above; shared
    // contents are page-aligned), and the caller guarantees no concurrent GPU write.
    let gpu =
        unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<i64>(), inputs * g) };
    let mut values = Vec::with_capacity(inputs * (g + c));
    for row in 0..inputs {
        values.extend_from_slice(&gpu[row * g..(row + 1) * g]);
        values.extend_from_slice(&cpu[row * c..(row + 1) * c]);
    }
    Ok(values)
}

/// The slot's shared buffer if it holds at least `bytes`, else a new one stored there.
pub(super) fn reserve(
    device: &Device,
    slot: &mut Option<Retained<Buffer>>,
    bytes: usize,
) -> Result<Retained<Buffer>> {
    if let Some(buffer) = slot.as_ref().filter(|b| b.length() >= bytes) {
        return Ok(buffer.clone());
    }
    let buffer = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| metal_error("buffer", "allocation failed"))?;
    *slot = Some(buffer.clone());
    Ok(buffer)
}

/// A committed command buffer that is waited for when dropped, so no exit path
/// (normal return, error or unwind) leaves GPU work reading or writing buffers
/// the engine may reuse.
pub(super) struct InFlight(pub(super) Retained<CommandBuffer>);

impl InFlight {
    /// Wait for completion; the GPU execution time reported by the command buffer.
    pub(super) fn wait(self) -> Result<Duration> {
        let command = &self.0;
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            let detail = command.error().map_or_else(
                || "unknown".into(),
                |e| e.localizedDescription().to_string(),
            );
            return Err(metal_error("execution", detail));
        }
        Ok(Duration::from_secs_f64(
            (command.GPUEndTime() - command.GPUStartTime()).max(0.0),
        ))
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        // Returns at once when `wait` has already run.
        self.0.waitUntilCompleted();
    }
}

/// One matrix call with its own input and output buffers, ready to encode.
pub struct Job {
    pub(super) plan: Plan,
    pub(super) activation: Retained<Buffer>,
    pub(super) scales: Retained<Buffer>,
    pub(super) output: Retained<Buffer>,
}

impl Job {
    /// Row-major M×N sums of the last completed [`MetalEngine::run`] containing this job.
    pub fn output(&self) -> Vec<i64> {
        // SAFETY: `run` waits for completion and takes jobs mutably, so no GPU work
        // writes this job's buffer while it is borrowed here.
        unsafe {
            download(
                &self.output,
                self.plan.inputs,
                self.plan.params.rows as usize,
                &[],
            )
        }
        // The buffer was allocated for exactly this length in `MetalEngine::job`.
        .expect("job output buffer")
    }

    /// Replace this job's activation rows and integer scales (same M) before a rerun.
    pub fn write_inputs(&mut self, activation: &[i8], scales: &[i32]) -> Result<()> {
        if activation.len() != self.plan.inputs * self.plan.params.cols as usize
            || scales.len() != self.plan.inputs * self.plan.groups
        {
            return Err(invalid("inputs must keep the job's M, K and K/512"));
        }
        // SAFETY: the job's private buffers are borrowed exclusively here, and `run`
        // waits for completion, so no GPU work reads them now.
        unsafe { upload(&self.activation, &self.scales, activation, scales) }
    }

    /// Bytes of compressed code this job reads (once per pass, independent of M).
    pub fn code_bytes(&self) -> u64 {
        self.plan.code_bytes
    }

    pub(super) fn encode(&self, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        self.plan
            .encode(encoder, &self.activation, &self.scales, &self.output);
    }
}

impl Plan {
    pub(super) fn output_len(&self) -> usize {
        self.inputs * self.params.rows as usize
    }

    /// The buffers must hold at least M×K i16, M×G i32 and M×N i64 values and stay
    /// alive until the command buffer completes.
    pub(super) fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        activation: &Buffer,
        scales: &Buffer,
        output: &Buffer,
    ) {
        let n = self.params.rows as usize;
        let cols = self.params.cols as usize;
        for dispatch in &self.dispatches {
            let first = dispatch.first_input;
            encoder.setComputePipelineState(&dispatch.pipeline);
            // SAFETY: the code and table buffers are retained by this plan and the
            // others by the caller, all until the command buffer completes (every
            // submission waits). Offsets were validated in `MetalEngine::plan`: the
            // code offset plus the stream length lies inside the mapping buffer, and
            // `first` < M addresses whole rows of the activation (M×K i16), scale
            // (M×G i32) and output (M×N i64) buffers, which are at least that long.
            // Params is a plain repr(C) value copied by setBytes.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&self.codes.buffer), self.code_offset, 0);
                encoder.setBuffer_offset_atIndex(Some(&self.table), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(activation), first * cols * 2, 2);
                encoder.setBuffer_offset_atIndex(Some(scales), first * self.groups * 4, 3);
                encoder.setBuffer_offset_atIndex(Some(output), first * n * 8, 4);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&self.params).cast(),
                    size_of::<Params>(),
                    5,
                );
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: n.div_ceil(ROW_TILE),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: ROW_TILE,
                    height: self.splits,
                    depth: 1,
                },
            );
        }
    }
}
