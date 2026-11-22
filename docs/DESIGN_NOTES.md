# Design notes

These choices constrain changes to the current Rust engine.
The [architecture](ARCHITECTURE.md) describes execution and file ownership.
The [specification](spec/README.md) describes upstream behaviour; it does not
promise that a Rust implementation reproduces every CUDA output bit.

Measurements here refer to one Apple M5 with 4 performance + 6 efficiency CPU
cores, a 10-core GPU and 24 GB memory. Timings marked approximate come from
measurements on a shared, loaded machine. They are guidance for design, not
current benchmark results. No comparison with other engines is established.

## One scalar numerical contract

The scalar `mackani-ref` implementation defines the required bits.
For identical tokens and cache mode, changing a kernel, CPU thread count,
prefill chunk size or speculative width must preserve logits and committed state.
This makes optimization independently testable and makes speculation lossless.
It also prevents an optimization from silently changing sampling behaviour.

CUDA native elementary functions, MMA instruction accumulation and some
upstream width-dependent paths do not have a complete portable bit contract.
The port chooses explicit math and orders rather than claiming CUDA identity.
Changing a canonical primitive requires an explicit contract change and new
independent evidence; matching only a quality score is insufficient.
Keep the scalar oracle safe, single-threaded and independent of fast kernels.

Recorded validation on the stated M5 scores 20,460 token positions:
profile-int8 perplexity is 2.075597096008 against upstream's 2.075899908604469;
`Kv5` is 2.075687410906 against 2.0759369466709248.
Perplexity is dimensionless; both differences fall within an absolute
perplexity tolerance of 0.001. These records do not validate arbitrary inputs
or replace exact codec/operator comparisons. Timings from loaded quality runs
do not establish current throughput.

## Evaluation order and fused arithmetic

Main-model floating products and additions round separately to binary32.
Binary64 sampler and quality operations retain their written order too.
Hadamard butterflies, XOR reductions, recurrent four-part folds, attention
split combination and sampler prefix sums must not be reassociated.
The attention substitute for `mma16` adds 16 separately rounded products in
ascending index order to its incoming accumulator. It is a port choice,
not a reconstruction of the instruction's internal schedule.

The draft preserves explicit source FMA sites as `f32::mul_add`, including
normalization and attention accumulation in `draft/math.rs`.
Other draft expressions remain separate operations.
Main-model normalization, KV encoding and floating CPU fast paths do not
introduce FMA. Do not enable compiler fast-math or contract these expressions.

`portable_math` supplies explicit exp/log, positive log1p, sin/cos,
nonlinearities and binary64 exp/log instead of platform transcendental calls.
Its square root uses an integer radicand and exact midpoint comparison,
then reciprocal square root uses a separately rounded division.
`exp2(t)` is portable `exp(t * f32::from_bits(0x3f31_7218))`, with the product
rounded before the call. This resolves the missing portable exp2 primitive.
Preserve the portable log's written zero-input behaviour and version-3
sampler's binary64 `+0.5` uniform construction, including its possible
upper endpoint; replacing it by an open-interval generator changes the contract.

## Exact Metal multiply-accumulate

Compressed weight-table values have magnitude at most 127; int8 activations
have magnitude at most 128. Conversion to binary16 is exact for these integers.
Within a sequence of 256 weights, every product has magnitude at most
2^14 and every partial sum at most 2^22. Every such integer is exactly
representable in binary32, whose consecutive-integer range reaches 2^24.
Therefore separate multiply/add and FMA return the exact same sequence sum,
regardless of accumulation order. This permission applies only to this
integer-valued projection stage.

The shader converts the sequence sum exactly to i32, scales it in i64 and
reduces in i64. Metal accepts at most 512 groups of 512 weights so even
extreme signed scales cannot overflow i64. Preserve those bounds when
changing shapes or quantization. `MTLMathMode::Safe` remains selected.
The shader's table expansion and reduction share 16 KiB of threadgroup memory;
SIMD width 32 is required. Pipelines specialize bit width, row count, layout
and alignment. Unaligned streams use byte loads.

## Work split and calibrated CPU share

Metal runs compressed projections. The CPU pool runs normalization,
quantization, recurrence, attention, residuals, nonlinearities and restoration.
These stages depend on each preceding projection's result, so the current
boundary requires four projection groups per layer and a final head.
Move a boundary only with bit-identity evidence for its floating stages.

While a projection runs, the CPU can compute its last complete 32-row output
tiles. The GPU retains at least one tile. Integer sums make this split
numerically neutral. An in-flight guard must wait even if CPU work fails or
unwinds; buffer reuse before GPU completion breaks ownership.

`CpuShare::Auto` applies only to Metal device `Apple M5`, the stated CPU/GPU
core counts and 10 CPU threads. Unknown machine facts disable it.
The fractions for 1, 2, 4, 8 and 16 activation rows are respectively
0.08, 0.08, 0.09, 0.12 and 0.15, interpolated between knots and capped above.
The conservative table avoids making the GPU wait on an overloaded CPU.
Fixed fractions in 0..=0.9 support calibration; head-prefix calls stay GPU-only.

Approximate loaded-machine kernel measurements on the stated M5 show why:
at 16 activation rows, binary32 MAC takes about 492 ms against integer MAC's
638 ms. In engine-like groups at one row, a CPU share of 0.12 takes about
134 ms against 145 ms GPU-only; heavier CPU load can reverse the gain.
These figures do not justify applying the table to other chips.
Use `bench_metal --shares` and `bench_model` with explicit execution options.

CPU projection kernels retain encoded streams and decode small tiles into
reusable scratch. They use dynamic tasks to balance unlike CPU cores.
Four independent SDOT accumulators remove a single dependency chain.
Portable and widening-NEON kernels remain correctness fallbacks.
Do not add a persistent dense or fully repacked copy of the main weights.

## Exact CPU stage fast paths

Each fast path preserves the scalar chain inside each independent lane.
Parallel completion order must not select a different result or first error.
The unchanged scalar entry points and frozen synthetic digests remain oracles.

| Stage | Fast path | Why the bits stay identical |
|---|---|---|
| KV preparation | Independent token/head/side tasks | Original scale order, rounding, packing and tie rules; scalar replay selects the first error before cache append |
| VIT2 search | Four reduced states per NEON vector, stack scratch | Same paired emission minimum before cost addition, predecessor bits, normalization and final signed cost/state tie |
| Dense recurrent gates | Deinterleave independent eight-element partitions; parallel gate outputs for a single row | Each lane retains ascending chunk/element products and additions, then the same XOR and group fold |
| Recurrent update | Four independent value channels | Separate decay, prediction, correction and readout operations in the same four-part order |
| Transaction initialization | Read committed matrices while writing the first decayed working state | Every entry is overwritten before use; only convolution history needs an initial copy; full commit swaps buffers |
| Hadamard and restoration | Independent butterflies/channels in NEON | Same butterfly distances, signs, scale order and FP16/BF16 boundaries |
| Normalization/residual/MLP | Independent elementwise SIMD with ordered reductions retained | Integer ties-to-even storage conversions and separate arithmetic; no approximate reciprocal |
| Portable nonlinearities | Same polynomial and branches across independent lanes | Same constants, grouping, divisions and rescaling; exceptional/range-limit lanes use scalar math |
| Attention | Share compact BF16 tiles across queries and rows of a KV head | Each score visits all 256 channels in order; each value channel visits visible keys in order; original softmax, self and split folds |

VIT2 SIMD checks 4/5-bit tables within [-1100,1100] and targets within
[-2048,2048]; other inputs use checked scalar search.
Overflow is an error, never assumed C++ signed wrapping.
Grouped attention uses 32 split-range tasks per KV head. Shared boundary tiles
are initialized once on a consuming thread; an owned tile is reused within a
task. Each temporary full tile occupies 33,024 bytes including scales.
Int8 operands convert exactly to BF16 and packed bases already have those bits.
Masked lanes never contribute value arithmetic or finite checks.
There is no persistent transpose or change to cache/snapshot representation.

## Cache contracts and allocation accounting

`Int8` is the plain rotated signed-byte cache with binary32 scales.
`Int8Profile` first encodes int8, dequantizes, applies profile selection and
re-encodes int8. Upstream's published “int8” quality record refers to this
second mode. `Kv5` packs eligible sides and uses `Int8Profile` elsewhere.
Changing a mode changes the numerical contract, not just its capacity.

The KVQP profile, not its filename, selects table values, candidate gains,
means, residual bases, guards and packable sides.
The codec retains the first strictly smaller scale error, nearest-even
integer/half conversions and its distinct general/VIT2 ties.
Packed attention reads BF16-rounded codebook bases, not the encoder's
unrounded reconstruction. Key means are ignored; each packed value split
adds its denominator times the mean after self and before split combination.
Prefix probability times value scale rounds to BF16; self keeps binary32.

For the released five-bit profile, 16 key sides and 13 value sides pack;
value ordinals 10, 14 and 15 fall back. Four KV heads give 116 packed and
12 int8 side rows per token over 16 attention layers.
A packed side row allocates 736 bytes: a 32-byte enum, 544-byte boxed row
including 256 BF16 bases, and 160 bytes of five-bit words.
An int8 side row allocates 288 bytes including its enum and 256 signed bytes.
Each layer adds a 48-byte token structure. Thus the product prices
89,600 bytes/token for this `Kv5` profile and 37,632 bytes/token for either
int8 mode, before allocator overhead covered by the margin.
The five-bit symbol payload alone is not the resident-memory estimate.
Recompute accounting if the Rust representation changes.

## Lossless speculation and deterministic depth

The main selector chooses every emitted token. A guess is accepted only if
it equals that choice at the same absolute position. After a mismatch,
EOS or cancellation, later provisional rows are ignored.
`commit(k)` publishes exactly the input predecessors of the emitted tokens;
the final emitted token remains pending. Canonical row math plus position-keyed
sampling gives induction from the same prefix to the same output as no draft.
Preserve both conditions; a plausible draft or high acceptance is no proof.

Partial commit replays cached recurrent inputs from committed state and
truncates attention. Full commit publishes working recurrence directly.
Snapshots copy recurrence and logits while retaining a live attention prefix;
they are neither transferable between sessions nor complete cache copies.
Joint snapshots also save the draft ring. Only committed captures update it.
Proposer failures after output begins disable drafting without losing emitted
tokens. A failure before output is an error. Prefill keeps bounded captures
and synchronizes context even when it omits vocabulary heads.

Defaults are `both`, maximum depth 7 and adaptive depth.
The latest match of the last three tokens supplies a free n-gram copy only
when its continuation spans at least six tokens; otherwise the neural draft runs.
A neural block still computes its full proposal budget; depth truncates its offer.

`DepthController` estimates P(accepted prefix >= j) at each depth j.
The prior for depths 1 through 7 is 0.84, 0.70, 0.57, 0.50, 0.45, 0.39, 0.34.
A resolved counterfactual proposal updates every depth with weight 0.25,
including guesses that were not offered. Choose depth d to maximize
`(1 + sum(e[1..=d])) / (pass_ms[d+1] + (d > 0 ? proposal_ms : 0))`;
ties choose the smaller depth. Zero skips the proposer.
A skipped pass relaxes estimates 5% toward the prior; after four skipped
passes, the next forces a positive-depth probe. Free copies bypass learning.

The fixed short-context cost table in `CostModel::apple_m5_neural` uses the
stated M5 configuration and `Kv5`/Metal. Rows 3 and 5 through 7 interpolate
measurements; the table is a policy calibration, not a fresh benchmark:

| Verification rows | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Cost (ms/pass) | 232 | 300 | 345 | 390 | 455 | 520 | 585 | 650 |

One neural proposal is priced at 47 ms/proposal. Longer passes extend the
last table step. No clock is read for a depth choice, so identical tokens and
proposals repeat the depth sequence. The table remains on other chips because
it can change speed only; CPU share needs stricter machine gating because it
can delay each projection. Context-dependent costs need separate calibration.

## Memory admission and file stability

The product inspects metadata before tokenizer/weight loading and refuses
loads whose estimate exceeds physical memory. `info` reports fit without
refusing for memory. No context, cache or system setting changes automatically.
The check uses physical capacity, not currently free memory or measured peak RSS.
Another process can still cause memory pressure after admission.

[`memory.rs`](../crates/mackani-cli/src/memory.rs) derives dimensions and sizes
from headers/profile. With byte quantities, context length C tokens, pass
width W = max(prefill width, K+1) rows, hidden H channels, FFN F channels,
vocabulary V tokens and captured feature width X channels, its formula is:

```text
cache    = cache_bytes_per_token * next_power_of_two(C)
snapshot = recurrent + 4*C + 4*V + draft_ring
working  = recurrent + W*replay_bytes_per_row + expanded_dense_vectors
         + 256*W*(H+F) + draft_ring + 8*W*X + 8*C + 4*W*V
subtotal = weights_file + draft + recurrent + cache + snapshot + working
margin   = ceil(0.15 * subtotal) + 1 GiB
needed   = subtotal + margin
```

The constants multiplying element counts include their byte sizes.
Draft counts mapped container/head files once plus owned approximate head,
projection/scales and expanded vectors. Selector predecessor/successor tables
stay mapped BF16; transient row decoding is scratch.
The cache uses next-power-of-two capacity to cover vector growth.
The 15% plus 1 GiB margin covers allocator/process/OS costs and unitemized scratch.
Preserve checked arithmetic and count each live owner when changing storage.

For the released default configuration (32,768 context tokens, W=16 rows,
`Kv5`, neural draft enabled), this formula gives 20,435,462,454 bytes
(19.03 GiB), including 2,721,905,176 draft bytes and 3,599,183,646 margin bytes.
These are calculated sizes, not measured memory peaks.
See the [configuration table](../crates/mackani-cli/README.md#options-and-environment)
for context/cache/draft alternatives; other memory capacities are estimates.
Mapped files must remain unchanged and untruncated until all owners drop.

## Startup validation and portability

`RealEngineConfig::validate` checks macOS/aarch64, macOS major version at least
26, positive settings, neural depth bounds and all required folder files.
The version parser treats 16.x as 26 for older-SDK compatibility.
Missing required facts or files fail explicitly. Metadata admission follows
before large payloads or the tokenizer load.
Default Metal probing checks device creation, shader compilation, SIMD width
32 and at least 16 KiB threadgroup memory. Threadgroup limits reduce SIMD
group count when possible; one stream exceeding `maxBufferLength` is an error.
The recommended GPU working set is reported, not an admission guarantee.

CPU instructions are detected at runtime. Physical-core queries choose a
pool size, falling back through physical count and available parallelism.
No affinity, i8mm or SME implementation is assumed.
Only the stated M5 is measured; fallback tests do not validate another Mac.
Execution switches are resolved by the CLI and passed explicitly to libraries.

## Server threading and bounded requests

The engine trait is synchronous and deliberately not `Send`.
Constructing and using it on one dedicated thread avoids moving Metal/session
state across async executors. A bounded request channel serializes inference;
Tokio handles transport while CPU pool threads run numerical tasks.
Cancellation is observed at generation boundaries, including prefill callbacks.
An eight-event response channel bounds buffering and transfers backpressure.

The [limits table](../crates/mackani-cli/README.md#security-and-transport-limits)
is authoritative for values and detailed reasons. The 8 MiB body limit leaves
room for a full text context without allowing unbounded JSON work; a 512 MiB
global budget covers body expansion and parsed values, including queued jobs.
At most 64 connections and 64 channel jobs suit one engine. A 600 s queue
wait bounds waiting separately from inference. The 30 s event backpressure
deadline releases a stalled generation; the 300 s client-I/O deadline does
not run while a job waits or computes. Keepalive comments use a 15 s interval.
Do not turn long engine silence into a client-idle cancellation.
The 64-token context reserve preserves room for control/answer bridging.

## Resolved specification boundaries

The upstream chapters retain native-operation limitations as statements about
the upstream source. The following answers define the port rather than leave
implementation questions open:

| Boundary | Answer and constraint |
|---|---|
| Dimensions | Read model fields and tensor/vector shapes; no released-width fallback. `qwen3_5` attention requires head width 256 and fixed rotary width 64 with stored frequency bits; do not regenerate from `rope_theta`. Recurrent reductions require width 128 and four convolution taps. |
| Unknown layer type | Every type other than `full_attention` selects the recurrent branch, with its shape validation, matching the source dispatch. |
| Attention masks | Use each row's own intersecting absolute tiles; masked probability is positive zero and masked value operations are skipped. Preserve signed-zero effects. |
| Table loading | Binary32 round-to-nearest-even and nearest-even integer conversion define the port. Reject all-zero/nonfinite lookup tables; zero activation rows remain valid. |
| Exceptional values | Validate finite weights/activations and checked integer costs. Do not assign portable outputs to upstream NaN conversions, unchecked directory reads or signed overflow. |
| Container layout | Read the full 64-bit header length specified by the format. The upstream reader's narrower load does not redefine the file layout. Reject undefined carry, AYA and `rar` transforms. |
| Residual embeddings | Require matching dimensions and input sign offsets, with no output rotation. Execution uses the main component's signs; differing residual sign seeds/rules are not separately rejected. |
| KVQP | Validate every consumed range, compatible dimensions and finite parameters; unknown bytes remain uninterpreted. Consume at most two declared residual bases. The released profile supplies the packing mix, not a hardcoded engine list. |
| Draft tensors | Validate warp-layout eight-bit projections, shared H512/Mulberry32 input signs, absent output rotation and BF16 dense vectors. Shape checks define supported artifacts. |
| Draft shortlist | Fill equal cutoff-bucket candidates in ascending token-ID order. This is deterministic; it does not reproduce a CUDA atomic-arrival schedule. |
| Reduced head | Rescore shortlisted candidates with SHEAD's separate int8 weights. Equality to a full compressed head is not a contract. |
| Draft sampling | Retain explicit FMA and portable draft math, independent draft temperature and fixed top-p 0.95. Main sampling version 2 requires greedy draft choice; version 3 can couple the seed. |
| Transaction ABI | `commit(k)` counts input rows directly; even one-row verification waits for commit. Main plus proposer checkpoints are explicit joint owners. |
| Prefill | Canonical row arithmetic defines chunk equivalence. CUDA memory-dependent resident batches and native width paths do not define port output bits. |
| Upstream policies | HCAP, 16-row neural blocks and next-token-table extensions are outside the port. The upstream table stores raw-logit top entries, not necessarily the sampled token. |
| Native compatibility | CUDA library/compiler versions and MMA internals remain unspecified by the source. Portable functions and ordered attention define Rust behaviour; exact native-CUDA equality is unverified. |

## Rejected approaches

Results below apply to the stated M5; timings are approximate where the
machine is loaded. Policy costs are projections through the fixed table,
not measured end-to-end speed.

| Attempt | Result | Reason to keep the current choice |
|---|---|---|
| Static equal CPU row partitions | Slower than dynamic task claiming | Unlike CPU cores leave an uneven tail |
| Generic decoding for common bit widths; one SDOT accumulator | Longer projection time | Constant shifts and independent accumulators reduce decode/dependency work |
| Persistently expanded FP32 attention tiles | Extra widening, copies and scratch traffic | Compact BF16 tiles preserve exact operands and reuse storage |
| Large CPU projection shares | Can slow a pass under external CPU load | The GPU waits for a late CPU suffix; keep conservative machine-gated fractions |
| Poll GPU completion, shared-event polling, or precommit a waiting buffer | No consistent reduction below approximately 0.11 ms/round trip | Scheduling/completion latency dominates; polling can compete with useful CPU work |
| Calibrate draft costs at startup | Identical requests can choose different depths under varying load | A fixed table makes the policy repeatable |
| Always prefer a short n-gram match | Lower proposal quality on prose/code | Require a continuation of at least six tokens before bypassing neural proposals |
| Always offer seven neural guesses | Excess expensive verification rows on hard text | Adaptive depth uses observed prefix acceptance |
| Confidence-based proposal truncation | Little benefit beyond adaptive depth; biases the counterfactual estimate | Keep full proposal scoring and truncate only the offered depth |
| Adaptive neural proposals without free long copies | Periodic probes can alias with a structured-output cycle | Long copies and relaxation avoid repeatedly observing only hard positions |

## Future work

- Reduce GPU command buffers per pass further. Fusing each MLP's gate+up and
  down brought the 64-layer logit pass from 257 to 193 submissions;
  loaded-machine measurements on the stated M5 attributed approximately 36 ms
  of a one-row pass at 257 to scheduling/completion round trips. The
  remaining groups cross recurrent and attention floating stages.
- Measure a bounded spin before CPU threads park between short jobs.
- Merge compatible per-layer pool dispatches without changing scalar order.
- Schedule useful CPU work during the GPU wait beyond the exact projection share.
- Calibrate a context-dependent draft cost table without introducing
  request-to-request timing dependence.
- Revisit thread quality-of-service classes only with controlled evidence:
  measurements on the stated M5 show no effect within 1% on eight-row passes
  and noise on one-row passes.
- Calibrate CPU shares, pool sizes and draft costs on other Apple Silicon chips.
