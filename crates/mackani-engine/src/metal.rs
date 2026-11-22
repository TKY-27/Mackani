use std::{
    collections::HashMap,
    ffi::{c_int, c_void},
    ptr::NonNull,
    rc::Rc,
    time::{Duration, Instant},
};

use crate::{CpuShare, Engine, Error, Matrix, MetalOptions, Result};
use mackani_format::Layout;
use objc2::{rc::Retained, rc::autoreleasepool, runtime::ProtocolObject};
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
    MTLMathMode, MTLResourceOptions, MTLSize,
};

// MTLCreateSystemDefaultDevice lives in CoreGraphics (objc2-metal crate documentation).
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

unsafe extern "C" {
    fn getpagesize() -> c_int;
}

mod mlp;
mod submission;
pub use mlp::FusedMlp;
pub use submission::Job;
use submission::{InFlight, download, reserve, upload};

const SOURCE: &str = include_str!("linear.metal");
const MLP_SOURCE: &str = include_str!("mlp.metal");

/// Binary16 encodings of every i8 value; all are exactly representable.
static HALF: std::sync::LazyLock<[u16; 256]> = std::sync::LazyLock::new(|| {
    std::array::from_fn(|byte| mackani_format::f32_to_f16(f32::from(byte as u8 as i8)))
});

/// FORMAT.md §4.2 for one 32-step block of a `bits`-bit code. 32 steps use exactly
/// `bits` words, so all four blocks of a sequence share these constant shifts.
/// `wb[0]` is the previous block's last word, `wb[1..=bits]` this block's words:
/// the state of step j is the 12 bits ending at wb-bit 32 + bits * (j + 1).
fn decode_block(bits: usize) -> String {
    let mut text = format!("#define DECODE_BLOCK{bits}");
    for step in 0..32 {
        let start = 20 + bits * (step + 1);
        let (word, shift) = (start / 32, start % 32);
        let window = match shift {
            0 => format!("wb[{word}]"),
            1..=20 => format!("(wb[{word}] >> {shift})"),
            _ => format!(
                "((wb[{word}] >> {shift}) | (wb[{}] << {}))",
                word + 1,
                32 - shift
            ),
        };
        text += &format!(" \\\n    e[{step}] = lut[{window} & 4095];");
    }
    text + "\n"
}

fn shader_source() -> String {
    (4..=8).map(decode_block).collect::<String>() + SOURCE + MLP_SOURCE
}
/// Activation rows per dispatch; the shader's register arrays hold 16.
const MAX_ROWS_PER_DISPATCH: usize = 16;
/// Output rows per SIMD group (one row per lane).
const ROW_TILE: usize = 32;
// Four groups balance sequence parallelism and threadgroup overhead.
const SIMD_GROUPS: usize = 4;
/// The shader's threadgroup array: the 4,096-entry widened table, then the reduction.
const THREADGROUP_BYTES: usize = 16_384;
/// With |dot| <= 512 * 127 * 128 per group and |scale| <= 2^31, 516 groups cannot
/// overflow i64. Bounding the group count makes the reference's checked overflow
/// error unreachable, so the unchecked GPU sum is identical.
const MAX_GROUPS: usize = 512;

type Device = ProtocolObject<dyn MTLDevice>;
type Buffer = ProtocolObject<dyn MTLBuffer>;
type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;
type CommandBuffer = ProtocolObject<dyn MTLCommandBuffer>;

/// CPU fraction of each matrix's rows at M = 1, 2, 4, 8 and 16 (linear in between,
/// constant above 16), from `bench_metal --shares` on the real model on the
/// 4P+6E M5 with 10 CPU workers. Each value is somewhat below the measured
/// optimum: a CPU that is slower than calibrated (other load) costs far more
/// than one that is faster.
const CPU_SHARE: [(usize, f64); 5] = [(1, 0.08), (2, 0.08), (4, 0.09), (8, 0.12), (16, 0.15)];
/// The machine `CPU_SHARE` was measured on: Metal device, GPU cores, physical
/// CPU cores per performance level, and the CPU worker count. On any other
/// machine, or one these facts cannot be read for, `CpuShare::Auto` is off: a
/// CPU slower than calibrated makes the GPU wait, and no other chip was
/// measured. The name alone also matches M5 variants with fewer cores.
const CALIBRATED_DEVICE: &str = "Apple M5";
const CALIBRATED_GPU_CORES: usize = 10;
const CALIBRATED_CORE_LEVELS: [usize; 2] = [4, 6];
const CALIBRATED_THREADS: usize = 10;

/// Whether `machine`, with this Metal `device`, is the calibrated one.
fn calibrated_machine(device: &str, machine: &crate::Machine) -> bool {
    device == CALIBRATED_DEVICE
        && machine.gpu_cores == Some(CALIBRATED_GPU_CORES)
        && machine.core_levels == CALIBRATED_CORE_LEVELS
}

fn share_table_applies(calibrated_machine: bool, cpu_threads: usize) -> bool {
    calibrated_machine && cpu_threads == CALIBRATED_THREADS
}

fn calibrated_cpu_share(m: usize) -> f64 {
    let m = m.clamp(1, 16);
    let upper = CPU_SHARE
        .iter()
        .position(|&(rows, _)| rows >= m)
        .unwrap_or(CPU_SHARE.len() - 1);
    let (m1, s1) = CPU_SHARE[upper];
    if m1 == m || upper == 0 {
        return s1;
    }
    let (m0, s0) = CPU_SHARE[upper - 1];
    s0 + (s1 - s0) * (m - m0) as f64 / (m1 - m0) as f64
}

/// CPU rows of an `n`-row matrix at `share`: a multiple of the 32-row tile,
/// leaving the GPU at least one tile (so 0 when n < 64).
fn cpu_rows(share: f64, n: usize) -> usize {
    let rows = (share * n as f64) as usize / ROW_TILE * ROW_TILE;
    rows.min(n.saturating_sub(ROW_TILE) / ROW_TILE * ROW_TILE)
}

/// The CPU part of a split call: row-major M×len sums of each (matrix, row
/// range) for the call's input, as `Engine::weighted_sums_rows`.
type CpuRows<'a> = dyn FnMut(&[(&Matrix, std::ops::Range<usize>)]) -> Result<Vec<Vec<i64>>> + 'a;

fn metal_error(context: &str, detail: impl std::fmt::Display) -> Error {
    Error::Io(std::io::Error::other(format!("Metal {context}: {detail}")))
}

fn ns_error(context: &str, error: &NSError) -> Error {
    metal_error(context, error.localizedDescription())
}

fn invalid(message: &str) -> Error {
    Error::InvalidInput(message.into())
}

/// A no-copy GPU view of the page-aligned bytes `start..start + length` of a
/// read-only file mapping (the whole mapping when it fits `maxBufferLength`).
struct CodeBuffer {
    // Declared first so the Metal buffer is released before the mapping it wraps.
    buffer: Retained<Buffer>,
    start: usize,
    length: usize,
    _mapping_owner: Matrix,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    rows: u32,
    physical_rows: u32,
    sequences: u32,
    cols: u32,
    shift: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineKey {
    bits: u32,
    rows_per_dispatch: u32,
    warp: bool,
    aligned: bool,
}

struct Dispatch {
    pipeline: Retained<Pipeline>,
    first_input: usize,
}

/// One validated matrix call without its input and output buffers.
struct Plan {
    codes: Rc<CodeBuffer>,
    code_offset: usize,
    table: Retained<Buffer>,
    params: Params,
    inputs: usize,
    groups: usize,
    code_bytes: u64,
    splits: usize,
    dispatches: Vec<Dispatch>,
}

/// Exact compressed-linear kernel on the system default Metal device.
///
/// Every file mapping wrapped for the GPU stays mapped until the engine is dropped.
pub struct MetalEngine {
    device: Retained<Device>,
    name: String,
    /// Whether this is the machine the CPU share table was measured on.
    calibrated_machine: bool,
    /// The device's `maxBufferLength`, or a smaller test limit.
    max_buffer_length: usize,
    /// A test limit on threads per threadgroup below each pipeline's own.
    max_threads: Option<usize>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: HashMap<PipelineKey, Retained<Pipeline>>,
    empty: Retained<Pipeline>,
    // Keyed by mapping address. The engine keeps every mapping it has wrapped alive
    // until it is dropped, so an address (and a table key) cannot be reused meanwhile.
    // One buffer per mapping, or several when it exceeds `max_buffer_length`.
    mappings: HashMap<usize, Vec<Rc<CodeBuffer>>>,
    /// Signed-byte tables keyed by (mapping address, lookup-table offset): the
    /// quantized table depends only on the f16 table at that offset, while tensors
    /// may share a code stream but not a table.
    tables: HashMap<(usize, u64), Retained<Buffer>>,
    cpu_share: CpuShare,
    /// Reusable input and per-matrix output buffers of `weighted_sums_shared` and
    /// `weighted_sums_split`, grown on demand and never shrunk.
    scratch_activation: Option<Retained<Buffer>>,
    scratch_scales: Option<Retained<Buffer>>,
    scratch_outputs: Vec<Option<Retained<Buffer>>>,
    /// The fused MLP stages, when the device runs them.
    mlp: Option<mlp::Pipelines>,
    /// Per-tensor row scales and signs of fused MLP spans, keyed like `tables`.
    constants: HashMap<(usize, u64, mlp::Constant), Retained<Buffer>>,
    silu: Option<Retained<Buffer>>,
    scratch_mlp: mlp::Scratch,
    command_buffers: std::cell::Cell<u64>,
}

impl MetalEngine {
    /// `Ok(None)` when this process has no Metal device. Binary32 arithmetic and
    /// the calibrated CPU share; reads no environment variable. An error
    /// when the shader does not compile or the device lacks what the kernel
    /// needs (SIMD width 32, 16 KB of threadgroup memory;).
    pub fn new() -> Result<Option<Self>> {
        Self::with_limits(None, None)
    }

    /// As [`Self::new`], but with limits below the device's, so tests can force a
    /// small device: `max_buffer_length` for the no-copy code buffers (a larger
    /// mapping is wrapped in several), `max_threads_per_threadgroup` for the
    /// kernel's threadgroups (fewer SIMD groups split each tile).
    pub fn with_limits(
        max_buffer_length: Option<usize>,
        max_threads_per_threadgroup: Option<usize>,
    ) -> Result<Option<Self>> {
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            return Ok(None);
        };
        let name = device.name().to_string();
        let memory = device.maxThreadgroupMemoryLength();
        if memory < THREADGROUP_BYTES {
            return Err(metal_error(
                "device",
                format!(
                    "{name} offers {memory} bytes of threadgroup memory; the kernel needs {THREADGROUP_BYTES}"
                ),
            ));
        }
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| metal_error("queue", "newCommandQueue returned nil"))?;
        let options = MTLCompileOptions::new();
        // Every binary32 product and partial sum is an exactly representable integer.
        options.setMathMode(MTLMathMode::Safe);
        let library = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(&shader_source()),
                Some(&options),
            )
            .map_err(|e| ns_error("shader compilation", &e))?;
        let function = library
            .newFunctionWithName(&NSString::from_str("empty_kernel"))
            .ok_or_else(|| metal_error("shader", "missing empty_kernel"))?;
        let empty = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| ns_error("pipeline", &e))?;
        // The shader maps one SIMD lane to one output row of a 32-row tile.
        let width = empty.threadExecutionWidth();
        if width != ROW_TILE {
            return Err(metal_error(
                "device",
                format!("{name} has SIMD width {width}; the kernel needs {ROW_TILE}"),
            ));
        }
        let device_limit = device.maxBufferLength();
        let mlp = mlp::Pipelines::new(&device, &library, max_threads_per_threadgroup)?;
        Ok(Some(Self {
            max_buffer_length: max_buffer_length.map_or(device_limit, |n| n.min(device_limit)),
            max_threads: max_threads_per_threadgroup,
            calibrated_machine: calibrated_machine(&name, &crate::Machine::detect()),
            name,
            device,
            queue,
            library,
            pipelines: HashMap::new(),
            empty,
            mappings: HashMap::new(),
            tables: HashMap::new(),
            cpu_share: CpuShare::Auto,
            scratch_activation: None,
            scratch_scales: None,
            scratch_outputs: Vec::new(),
            mlp,
            constants: HashMap::new(),
            silu: None,
            scratch_mlp: mlp::Scratch::default(),
            command_buffers: std::cell::Cell::new(0),
        }))
    }

    /// Whether this process has a system default Metal device (no shader compilation).
    pub fn device_exists() -> bool {
        MTLCreateSystemDefaultDevice().is_some()
    }

    /// The system Metal device name used for these projections.
    pub fn device_name(&self) -> String {
        self.name.clone()
    }

    /// The largest no-copy code buffer (the device's `maxBufferLength` unless a
    /// test limit is smaller).
    pub fn max_buffer_length(&self) -> usize {
        self.max_buffer_length
    }

    /// The device's recommended GPU working-set size in bytes (reported, not enforced).
    pub fn recommended_working_set(&self) -> u64 {
        self.device.recommendedMaxWorkingSetSize()
    }

    /// No-copy code buffers created so far (one per mapping that fits
    /// `maxBufferLength`, more for larger ones).
    pub fn code_buffers(&self) -> usize {
        self.mappings.values().map(Vec::len).sum()
    }

    /// Whether `CpuShare::Auto` uses the calibrated table for a CPU pool of
    /// `cpu_threads` workers: only on the machine it was measured on.
    pub fn share_table_applies(&self, cpu_threads: usize) -> bool {
        share_table_applies(self.calibrated_machine, cpu_threads)
    }

    /// Set the CPU share for subsequent calls without changing their sums.
    pub fn set_options(&mut self, options: MetalOptions) {
        self.cpu_share = options.cpu_share;
    }

    /// Command buffers committed by this engine so far.
    pub fn command_buffers(&self) -> u64 {
        self.command_buffers.get()
    }

    /// The CPU share used for subsequent calls.
    pub fn options(&self) -> MetalOptions {
        MetalOptions {
            cpu_share: self.cpu_share,
        }
    }

    /// Requested threads per threadgroup; a call uses fewer when the device or
    /// pipeline allows fewer, or the matrix has fewer sequences than splits.
    pub fn threads_per_threadgroup(&self) -> usize {
        ROW_TILE * SIMD_GROUPS
    }

    fn pipeline(&mut self, key: PipelineKey) -> Result<Retained<Pipeline>> {
        if let Some(pipeline) = self.pipelines.get(&key) {
            return Ok(pipeline.clone());
        }
        let values = MTLFunctionConstantValues::new();
        let constants: [(NonNull<c_void>, MTLDataType); 4] = [
            (NonNull::from(&key.bits).cast(), MTLDataType::UInt),
            (
                NonNull::from(&key.rows_per_dispatch).cast(),
                MTLDataType::UInt,
            ),
            (NonNull::from(&key.warp).cast(), MTLDataType::Bool),
            (NonNull::from(&key.aligned).cast(), MTLDataType::Bool),
        ];
        for (index, (value, kind)) in constants.into_iter().enumerate() {
            // SAFETY: each pointer refers to a live u32 (UInt) or bool (Bool, one
            // byte) in `key`, matching the declared type; indices 0..3 are the
            // shader's function_constant indices.
            unsafe { values.setConstantValue_type_atIndex(value, kind, index) };
        }
        let function = self
            .library
            .newFunctionWithName_constantValues_error(&NSString::from_str("weighted_sums"), &values)
            .map_err(|e| ns_error("function specialization", &e))?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| ns_error("pipeline", &e))?;
        self.pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    /// A no-copy buffer holding `matrix`'s code stream from its 4-byte-aligned
    /// start: the whole mapping when it fits `max_buffer_length`, otherwise a
    /// window of the mapping starting at the stream's page. Nothing is copied.
    fn code_buffer(&mut self, matrix: &Matrix) -> Result<Rc<CodeBuffer>> {
        let mapping = matrix.mapping();
        let address = mapping.as_ptr() as usize;
        let range = matrix.code_range();
        let (first, end) = (range.start & !3, range.end);
        if let Some(buffer) = self.mappings.get(&address).and_then(|buffers| {
            buffers
                .iter()
                .find(|b| b.start <= first && end <= b.start + b.length)
        }) {
            return Ok(Rc::clone(buffer));
        }
        // SAFETY: getpagesize has no preconditions.
        let page = usize::try_from(unsafe { getpagesize() })
            .map_err(|_| metal_error("page size", "negative"))?;
        if page == 0 || !address.is_multiple_of(page) {
            return Err(metal_error("no-copy buffer", "mapping is not page-aligned"));
        }
        let mapped = mapping
            .len()
            .checked_next_multiple_of(page)
            .filter(|&length| length > 0 && end <= mapping.len())
            .ok_or_else(|| metal_error("no-copy buffer", "invalid mapping length"))?;
        let limit = self.max_buffer_length / page * page;
        let start = if mapped <= limit {
            0
        } else {
            first / page * page
        };
        let length = (mapped - start).min(limit);
        if end > start + length {
            return Err(metal_error(
                "no-copy buffer",
                format!(
                    "a code stream of {} bytes does not fit maxBufferLength {}",
                    range.len(),
                    self.max_buffer_length
                ),
            ));
        }
        let pointer = NonNull::new(mapping[start..].as_ptr().cast_mut().cast::<c_void>())
            .ok_or_else(|| metal_error("no-copy buffer", "null mapping"))?;
        // SAFETY: `pointer` is page-aligned (a page-aligned mapping plus a multiple
        // of the page size) inside a live read-only mapping, and `length` ends at
        // or before the end of the mapping's last page, which the VM maps as a
        // whole (bytes past EOF read as zero). The returned CodeBuffer owns a
        // Matrix clone, which keeps the mapping alive until after the Metal buffer
        // is released, so no deallocator is needed. Shaders only read through it.
        let buffer = unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    pointer,
                    length,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
        .ok_or_else(|| {
            metal_error(
                "no-copy buffer",
                "newBufferWithBytesNoCopy rejected the read-only mapping",
            )
        })?;
        let buffer = Rc::new(CodeBuffer {
            buffer,
            start,
            length,
            _mapping_owner: matrix.clone(),
        });
        self.mappings
            .entry(address)
            .or_default()
            .push(Rc::clone(&buffer));
        Ok(buffer)
    }

    fn empty_buffer(&self, length: usize) -> Result<Retained<Buffer>> {
        self.device
            .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| metal_error("buffer", "allocation failed"))
    }

    fn shared_buffer(&self, bytes: &[u8]) -> Result<Retained<Buffer>> {
        let pointer = NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>())
            .ok_or_else(|| metal_error("buffer", "null input"))?;
        // SAFETY: `pointer` addresses `bytes.len()` readable bytes; Metal copies them.
        unsafe {
            self.device.newBufferWithBytes_length_options(
                pointer,
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| metal_error("buffer", "allocation failed"))
    }

    /// Validate one call against the matrix and prepare its buffers. `n` is the
    /// number of leading physical rows computed (all of them for a full call).
    pub fn job(
        &mut self,
        matrix: &Matrix,
        activation: &[i8],
        scales: &[i32],
        n: usize,
    ) -> Result<Job> {
        let plan = self.plan(matrix, activation.len(), scales.len(), n)?;
        let mut job = Job {
            activation: self.empty_buffer(activation.len() * 2)?,
            scales: self.empty_buffer(scales.len() * 4)?,
            output: self.empty_buffer(plan.output_len() * 8)?,
            plan,
        };
        job.write_inputs(activation, scales)?;
        Ok(job)
    }

    /// Validate one call of `activation_len` i8 inputs and `scales_len` integer scales.
    fn plan(
        &mut self,
        matrix: &Matrix,
        activation_len: usize,
        scales_len: usize,
        n: usize,
    ) -> Result<Plan> {
        let d = matrix.descriptor();
        let (cols, physical_rows, bits) = (d.cols, d.physical_rows, d.bits);
        if activation_len == 0
            || cols == 0
            || !cols.is_multiple_of(512)
            || !activation_len.is_multiple_of(cols)
            || n == 0
            || n > physical_rows
        {
            return Err(invalid("invalid activation width or output row count"));
        }
        let inputs = activation_len / cols;
        let groups = cols / 512;
        let sequences = cols / 256;
        if scales_len != inputs * groups {
            return Err(invalid("integer scales must have M times K/512 elements"));
        }
        if groups > MAX_GROUPS {
            return Err(invalid("more than 512 groups could overflow i64"));
        }
        // Re-establish here every bound the shader's 32-bit indexing relies on,
        // independently of the format reader.
        let range = matrix.code_range();
        let stream_bytes = physical_rows
            .checked_mul(sequences)
            .and_then(|v| v.checked_mul(bits))
            .and_then(|v| v.checked_mul(16));
        if !(4..=8).contains(&bits)
            || stream_bytes != Some(range.len())
            || range.end > matrix.mapping().len()
            || range.len() > u32::MAX as usize - 4
            || (d.layout == Layout::Warp && !(physical_rows * sequences).is_multiple_of(32))
            || inputs
                .checked_mul(cols.max(n))
                .is_none_or(|v| v > u32::MAX as usize)
        {
            return Err(invalid("matrix shape exceeds the Metal kernel's bounds"));
        }
        // Bind at a 4-byte boundary; an unaligned stream uses byte loads from `shift`.
        let aligned = range.start.is_multiple_of(4);
        let codes = self.code_buffer(matrix)?;
        // The buffer starts at or before this boundary and contains the stream.
        let code_offset = (range.start & !3) - codes.start;
        let params = Params {
            rows: n as u32,
            physical_rows: physical_rows as u32,
            sequences: sequences as u32,
            cols: cols as u32,
            shift: (range.start & 3) as u32,
        };
        let mut dispatches = Vec::new();
        // Idle SIMD groups would only add barrier work for short rows.
        let mut splits = SIMD_GROUPS.min(sequences);
        for first_input in (0..inputs).step_by(MAX_ROWS_PER_DISPATCH) {
            let rows = (inputs - first_input).min(MAX_ROWS_PER_DISPATCH);
            let pipeline = self.pipeline(PipelineKey {
                bits: bits as u32,
                rows_per_dispatch: rows as u32,
                warp: d.layout == Layout::Warp,
                aligned,
            })?;
            // A device or pipeline allowing fewer threads than 32 x splits gets
            // fewer SIMD groups per tile: the sums are exact for any split count.
            let limit = self
                .max_threads
                .map_or(pipeline.maxTotalThreadsPerThreadgroup(), |n| {
                    n.min(pipeline.maxTotalThreadsPerThreadgroup())
                });
            if limit < ROW_TILE {
                return Err(metal_error(
                    "pipeline",
                    format!("supports {limit} threads per threadgroup, at least {ROW_TILE} needed"),
                ));
            }
            splits = splits.min(limit / ROW_TILE);
            dispatches.push(Dispatch {
                pipeline,
                first_input,
            });
        }
        let table_key = (matrix.mapping().as_ptr() as usize, d.table_offset);
        let table = match self.tables.get(&table_key) {
            Some(table) => table.clone(),
            None => {
                let bytes: Vec<u8> = matrix.weight_table8().iter().map(|&v| v as u8).collect();
                let table = self.shared_buffer(&bytes)?;
                self.tables.insert(table_key, table.clone());
                table
            }
        };
        Ok(Plan {
            codes,
            code_offset,
            table,
            params,
            inputs,
            groups,
            code_bytes: (n * sequences * bits * 16) as u64,
            splits,
            dispatches,
        })
    }

    /// Encode all jobs into one command buffer, commit, and wait once.
    /// Returns the GPU execution time reported by the command buffer.
    pub fn run(&self, jobs: &mut [Job]) -> Result<Duration> {
        self.submit(|encoder| {
            for job in jobs.iter() {
                job.encode(encoder);
            }
        })
    }

    /// Row-major M×N sums (padded physical rows included) of several matrices
    /// that read the same input, encoded into one command buffer. Input and
    /// output buffers are reused across calls.
    pub fn weighted_sums_shared(
        &mut self,
        matrices: &[&Matrix],
        activation: &[i8],
        scales: &[i32],
    ) -> Result<Vec<Vec<i64>>> {
        self.shared(None, matrices, activation, scales)
    }

    /// As [`Self::weighted_sums_shared`], but the CPU kernel on `cpu` computes
    /// the last rows of each matrix while the GPU computes the others. The row
    /// split comes only from the matrix size, M and [`Self::cpu_share`], so a run
    /// is repeatable; the sums are exact whatever the split. A matrix with fewer
    /// than 64 physical rows runs on the GPU alone. Every call covers all physical
    /// rows: there is no prefix (head slice) variant of the split, and no
    /// projection call site computes a prefix on Metal.
    pub fn weighted_sums_split(
        &mut self,
        cpu: &mut Engine,
        matrices: &[&Matrix],
        activation: &[i8],
        scales: &[i32],
    ) -> Result<Vec<Vec<i64>>> {
        let threads = cpu.threads();
        self.shared(
            Some((
                &mut |parts| cpu.weighted_sums_rows(parts, activation, scales),
                threads,
            )),
            matrices,
            activation,
            scales,
        )
    }

    /// The CPU's fraction of each matrix's rows for M activation rows and a CPU
    /// pool of `cpu_threads` workers (0 = GPU only). `CpuShare::Auto` is the
    /// calibrated table where [`Self::share_table_applies`], else 0.
    pub fn cpu_share(&self, m: usize, cpu_threads: usize) -> f64 {
        match self.cpu_share {
            CpuShare::Off => 0.0,
            CpuShare::Fixed(share) => share,
            CpuShare::Auto if self.share_table_applies(cpu_threads) => calibrated_cpu_share(m),
            CpuShare::Auto => 0.0,
        }
    }

    /// Override the CPU share for all M (`None` restores `CpuShare::Auto`).
    pub fn set_cpu_share(&mut self, share: Option<f64>) -> Result<()> {
        self.cpu_share = match share {
            None => CpuShare::Auto,
            Some(share) => CpuShare::fixed(share)?,
        };
        Ok(())
    }

    /// `cpu` is the CPU part and the CPU pool's worker count.
    fn shared(
        &mut self,
        mut cpu: Option<(&mut CpuRows, usize)>,
        matrices: &[&Matrix],
        activation: &[i8],
        scales: &[i32],
    ) -> Result<Vec<Vec<i64>>> {
        let split: Vec<(usize, usize)> = matrices
            .iter()
            .map(|matrix| {
                let d = matrix.descriptor();
                let n = d.physical_rows;
                let m = activation.len() / d.cols.max(1);
                let cpu_rows = match &cpu {
                    Some((_, threads)) => cpu_rows(self.cpu_share(m, *threads), n),
                    None => 0,
                };
                (n - cpu_rows, n)
            })
            .collect();
        let plans = matrices
            .iter()
            .zip(&split)
            .map(|(matrix, &(gpu_rows, _))| {
                self.plan(matrix, activation.len(), scales.len(), gpu_rows)
            })
            .collect::<Result<Vec<_>>>()?;
        let activation_buffer = reserve(
            &self.device,
            &mut self.scratch_activation,
            activation.len() * 2,
        )?;
        let scales_buffer = reserve(&self.device, &mut self.scratch_scales, scales.len() * 4)?;
        if self.scratch_outputs.len() < plans.len() {
            self.scratch_outputs.resize(plans.len(), None);
        }
        let outputs = plans
            .iter()
            .zip(&mut self.scratch_outputs)
            .map(|(plan, slot)| reserve(&self.device, slot, plan.output_len() * 8))
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: the scratch buffers are distinct objects owned by this engine,
        // which is borrowed mutably here. Every earlier command buffer of this
        // engine was waited for by its `InFlight` guard before its call returned
        // or unwound, and this call has not committed yet, so no GPU work or
        // other reference touches them.
        unsafe { upload(&activation_buffer, &scales_buffer, activation, scales) }?;
        let in_flight = self.commit(|encoder| {
            for (plan, output) in plans.iter().zip(&outputs) {
                plan.encode(encoder, &activation_buffer, &scales_buffer, output);
            }
        })?;
        // The CPU rows run while the GPU works; wait for the GPU even if they fail.
        let cpu_sums = match cpu.as_mut() {
            Some((cpu, _)) if split.iter().any(|&(gpu_rows, n)| gpu_rows < n) => {
                let parts: Vec<_> = matrices
                    .iter()
                    .zip(&split)
                    .map(|(matrix, &(gpu_rows, n))| (*matrix, gpu_rows..n))
                    .collect();
                cpu(&parts)
            }
            _ => Ok(vec![Vec::new(); matrices.len()]),
        };
        in_flight.wait()?;
        let cpu_sums = cpu_sums?;
        plans
            .iter()
            .zip(&outputs)
            .zip(&cpu_sums)
            .map(|((plan, output), cpu)| {
                // SAFETY: `wait` returned Ok, so the command buffer has completed and
                // nothing writes these engine-owned buffers any more.
                unsafe { download(output, plan.inputs, plan.params.rows as usize, cpu) }
            })
            .collect()
    }

    /// Encode with `encode` into one command buffer, commit, and wait for completion.
    /// Returns the GPU execution time reported by the command buffer.
    fn submit(
        &self,
        encode: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) -> Result<Duration> {
        self.commit(encode)?.wait()
    }

    /// Encode with `encode` into one command buffer and commit it; the returned
    /// guard waits for completion when it is dropped.
    fn commit(
        &self,
        encode: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) -> Result<InFlight> {
        autoreleasepool(|_| {
            let command = self
                .queue
                .commandBuffer()
                .ok_or_else(|| metal_error("command buffer", "nil"))?;
            let encoder = command
                .computeCommandEncoder()
                .ok_or_else(|| metal_error("encoder", "nil"))?;
            encode(&encoder);
            encoder.endEncoding();
            command.commit();
            self.command_buffers.set(self.command_buffers.get() + 1);
            Ok(InFlight(command))
        })
    }

    /// Wall time of committing and waiting for one trivial one-thread dispatch.
    pub fn round_trip(&self) -> Result<Duration> {
        let start = Instant::now();
        self.submit(|encoder| {
            encoder.setComputePipelineState(&self.empty);
            let one = MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(one, one);
        })?;
        Ok(start.elapsed())
    }

    /// Row-major M×N sums including padded physical rows, as `mackani_engine::Engine`.
    pub fn weighted_sums(
        &mut self,
        matrix: &Matrix,
        activation: &[i8],
        scales: &[i32],
    ) -> Result<Vec<i64>> {
        let n = matrix.descriptor().physical_rows;
        self.weighted_sums_prefix(matrix, activation, scales, n)
    }

    /// The first `n` physical rows, e.g. a vocabulary-head slice.
    pub fn weighted_sums_prefix(
        &mut self,
        matrix: &Matrix,
        activation: &[i8],
        scales: &[i32],
        n: usize,
    ) -> Result<Vec<i64>> {
        let mut job = [self.job(matrix, activation, scales, n)?];
        self.run(&mut job)?;
        Ok(job[0].output())
    }
}

#[cfg(test)]
#[path = "../tests/fixture/mod.rs"]
mod fixture;

#[cfg(test)]
mod tests;
