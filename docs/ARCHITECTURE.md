# Architecture

Mackani runs the compressed Qwen3.8-27B model on Apple Silicon.
The Rust scalar implementation defines the numerical contract.
The [upstream behaviour specification](spec/README.md) describes the source
that informs that contract; [design notes](DESIGN_NOTES.md) define port choices.

## Workspace and ownership

| Crate | Responsibility | Workspace dependencies |
|---|---|---|
| `mackani-format` | Validate `.yskn` headers, descriptors, payloads, signs and storage conversions | None |
| `mackani-ref` | Scalar operators, portable math, model state, KV codec, draft, sampler and shared generation loop | `mackani-format` |
| `mackani-engine` | Mapped weights, compressed CPU/Metal kernels, CPU pool and exact CPU stages; fast main model and draft | `mackani-format`, `mackani-ref` |
| `mackani-text` | Tokenizer, Jinja chat template, incremental decoding, reasoning and tool-call parsing | None |
| `mackani-cli` (package `mackani`) | Product configuration, memory admission, chat, HTTP, selfcheck and benchmarks | All four crates |

The engine reuses reference loading, rounding and state interfaces.
The reference does not depend on the fast engine or product.
The text crate uses `tokenizers` and `minijinja`, with no inference dependency.
The CLI's synchronous `Engine` trait connects `ChatRunner` to `RealEngine`.
An explicit stub engine supports product development without weights.

Key entry points are
[`model.rs`](../crates/mackani-engine/src/model.rs),
[`model/execution.rs`](../crates/mackani-engine/src/model/execution.rs),
[`speculation.rs`](../crates/mackani-ref/src/speculation.rs) and
[`real_engine.rs`](../crates/mackani-cli/src/real_engine.rs).

## Weights and execution resources

`MappedModel` validates a read-only file mapping through `mackani-format`.
`Matrix` handles share its lifetime and retain encoded `row` or `warp` streams.
They do not expand the complete model into dense weights.
Dense vectors, lookup tables, scales and signs are validated before execution.
The mapping API is unsafe because the caller must prevent external file
modification or truncation for the lifetime of every mapped view.
The product establishes that contract at `map_files`.

`Engine` owns persistent CPU threads and dynamically distributes independent
row, head and matrix tasks. Results return in index order.
The default thread count follows detected physical cores, bounded by available
parallelism; [`machine.rs`](../crates/mackani-engine/src/machine.rs) defines fallbacks.
CPU compressed kernels select detected dotprod, NEON, then portable Rust.
`CpuStages::Auto` separately selects exact NEON stages when available;
`Reference` selects scalar stages around the compressed projections.

`Compute` owns that pool and, for `Backend::Metal`, a `MetalEngine`.
The product probes Metal and falls back to CPU if automatic selection fails.
An explicit unavailable backend or CPU kernel is an error.
Libraries receive execution options explicitly and read no environment options.

Metal wraps code streams in shared, page-aligned buffers without copying them.
A mapping can use several views to satisfy `maxBufferLength`.
Lookup tables and specialized pipelines are cached.
A submission guard waits for every committed command on success, error or
unwind before its scratch buffers can be reused.

## A forward pass through one layer

The engine processes a batch layer by layer; each row remains causal.
Embedding lookup produces the initial hidden rows on the CPU pool.
Each layer executes the following sequence in `model/execution.rs`:

1. CPU: apply input normalization, input signs and the Hadamard transform;
   round and quantize the normalized rows into int8 activations and i32 scales.
2. Projection backend: compute exact weighted integer sums for the branch's
   shared input. A recurrent layer groups QKV and output-gate projections;
   attention groups query/gate, key and value projections.
3. CPU: restore the i64 sums to floating projections with the reference's
   scale, inverse-RMS, rotation and rounding boundaries.
4. CPU branch work: recurrent layers prepare convolution and dense gates,
   then advance each value head in token order on a pool thread.
   Attention prepares queries and cache sides, appends the encoded rows,
   then computes causal attention grouped by KV head.
5. CPU: quantize branch output. Projection backend: run the branch output
   projection. CPU: restore it and perform the BF16 residual addition.
6. CPU: post-attention normalization and activation quantization.
   Projection backend: compute gate and up together. CPU: restore them and
   form the SiLU-gated product with its specified rounding.
7. CPU: quantize the product. Projection backend: compute down.
   CPU: restore it and perform the second BF16 residual addition.
   On Metal with vectorized CPU stages, steps 6 and 7 share one command
   buffer: after gate and up, Metal kernels restore them, form the gated
   product, quantize it and compute down. A span meeting a subnormal or
   non-finite value, which the GPU could round differently or the reference
   rejects, is recomputed by the CPU stages instead.

The GPU performs compressed projections only. Each call consumes CPU-prepared
int8 activations and integer scales; Metal widens activations to exact half
integers and returns row-major i64 sums in shared buffers.
Apart from the fused MLP span, floating restoration and all branch state
remain on the CPU.
With a CPU share enabled, the CPU kernel computes the last complete output-row
tiles concurrently with the GPU prefix. Both return the same exact integers.
The fused MLP span runs every row on the GPU.
The CPU backend performs every projection on the pool instead.

Every layer has four projection groups; on Metal, gate+up and down share one
command buffer. A model with 64 layers plus its final vocabulary head uses 193
command buffers (257 with reference CPU stages) for a pass that requests
logits and fits within Metal's 16-activation-row dispatch width.
After the final layer, the CPU normalizes selected rows and the projection
backend computes the vocabulary head. `LogitRows::{All,Last,None}` controls
which heads are evaluated; it does not change layer arithmetic.
Optional capture copies selected layers' post-MLP residuals for the draft.

## Cache and attention

The cache belongs to each attention layer and grows by token.
Each token contains a key and value row for every KV head.
The model's `KvMode` fixes its representation:

| Mode | Encoding |
|---|---|
| `Int8` | Signed rotated int8 rows plus one binary32 scale per side/head |
| `Int8Profile` | Initial int8, dequantization, profile quantization, final int8 |
| `Kv5` | Profile encoding into packed symbols and a scale on eligible sides; `Int8Profile` on the others |

Library main-model loaders default to `Int8`; the product defaults to `Kv5`.
Profile modes read the manifest's KVQP profile and check its dimensions.
`PackedRow` keeps both symbols and derived BF16 bases for attention;
its name does not imply smaller resident memory than int8 in this implementation.
The profile determines side eligibility, tables, means and basis corrections.

Attention distinguishes prefix from self, including the BF16 prefix weights
and binary32 self weights. It visits each row's absolute 32-slot prefix tiles
in order and skips masked value operations. The 31/42 split schedule and its
combination order remain fixed. Grouped CPU tasks share temporary BF16 tiles
across query heads and pass rows; they do not create another persistent cache.

## Transactions, generation and prompt reuse

`Model::run_provisional` advances preallocated working recurrent state and
appends provisional attention rows without publishing a committed position.
It retains per-row recurrent inputs in bounded replay storage.
`commit(k)` counts input rows, including the frontier row:
full acceptance publishes working recurrence; partial acceptance replays only
the accepted recurrent inputs and truncates attention; zero discards the pass.
Failure leaves the committed prefix intact.

The shared `speculation::generate` loop holds one pending token.
It obtains guesses from `NgramProposer`, `DraftProposer<FastDraft>` or
`CombinedProposer`, then verifies `[pending, guesses...]` in one main-model pass.
For each row, the main selector chooses at the absolute prediction position.
The loop emits that choice and continues only if the corresponding guess
matches. It stops at mismatch, EOS or cancellation, commits the confirmed input
rows and keeps the last emitted token pending for the next pass.
Guessing cannot change the selector's logits or random key.

The neural draft consumes committed captured residuals and embeddings.
`DraftProposer` synchronizes these into a bounded context ring.
`FastDraft` uses mapped compressed projections and a separate CPU pool with
the same execution choices as the main model; the pools run in sequence.
The reduced SHEAD first approximates scores, builds a shortlist and rescores it.
The block contains eight input rows, including the frontier, and proposes up
to seven tokens. Its current-block attention is noncausal across the full block.
`both` prefers an n-gram continuation only for a sufficiently long match;
the adaptive controller chooses how many neural guesses to offer.

Prefill processes bounded chunks without speculation.
`prefill_with_logits` can omit unnecessary vocabulary heads while preserving
all recurrent, attention and capture updates.
The product prefills all prompt tokens except the final pending token.
It synchronizes draft context after each chunk and checks cancellation.

A `Snapshot` owns recurrence, token IDs, committed logits and captures.
Attention data stays in the live cache; restoration requires the same session
and a retained exact prefix. Reset invalidates snapshots.
Joint proposer snapshots also own the draft ring and its identity.
`RealEngine` maintains one reusable prompt checkpoint, reuses a live strict
prefix when possible, restores a compatible checkpoint otherwise, or resets.
It reserves 64 context tokens for product control work.

## Product and app

Tokio/Hyper handles connections, bounded body decoding and response streaming.
A bounded job channel hands requests to one dedicated engine thread.
The factory loads `ChatRunner`, tokenizer and the non-`Send` engine there;
every engine call stays there. The pool belongs below that thread.
Per-response bounded event channels provide backpressure and cancellation.
The [HTTP reference](../crates/mackani-cli/README.md#security-and-transport-limits)
gives the transport limits and their rationale.

`mackani_text::ChatFormatter` renders messages/tools through the model's Jinja template,
encodes token IDs and incrementally separates reasoning, content and tool calls.
`ChatRunner` applies request limits, stopping, thinking budgets and loop guards
around that text/engine boundary.

The [SwiftUI app](../app/README.md) is an HTTP client and process owner.
`ServerController` launches the bundled `mackani serve` on a loopback port with
a fresh key. `ServerProcess` holds a stdin pipe and uses
`--exit-when-stdin-closes` to tie server lifetime to the app.
`ChatClient` consumes SSE; the app does not link the Rust engine directly.

## Model folder readers

Names selected by `files.*` are relative to `yasakani.json` and validated for containment.

| File | Reader and purpose |
|---|---|
| `yasakani.json` | `model::model_location_with_kv`, `draft::model_paths`, product platform/memory checks: paths and context |
| Main `.yskn` (`files.weights`) | `Container`, `MappedModel`, reference model loading: header, weights and dense vectors |
| `kvq_u5.bin` (`files.kv_profile`) | `KvProfile::parse`: profile modes' tables, groups, means and basis data |
| `bdraft2_k8.yskn` (`files.draft`) | Scalar `Draft` or `FastDraft`: draft tensors and vectors, required for neural drafting |
| `bdraft2_k8.shead` (`files.draft_head`) | Reference `ReducedHead` or engine draft `Head`: reduced-head projection, selector and rescore data |
| `tokenizer.json` | `ChatFormatter::load`: tokenizer vocabulary and encoding |
| `tokenizer_config.json` | `ChatFormatter::load`: token settings and fallback chat template |
| `chat_template.jinja` | `ChatFormatter::load`: preferred template when present |
| `generation_config.json` | `RealEngineConfig::sampling_defaults`: optional sampling defaults |
| `selfcheck.json` | Product `selfcheck`: recorded prompt hashes, lengths and token digests |

## Correctness and real-model commands

Scalar operators and portable elementary functions supply independent expected
bits. Engine tests compare outputs, cache representations and state digests
across kernels, thread counts, pass widths and scalar/fast CPU stages.
Synthetic tests cover container corruption, rounding boundaries, partial commit,
snapshots and draft/no-draft equality. Perplexity is a separate quality gate;
it cannot establish a codec's bit identity or CUDA equivalence.

Run from the repository root, with a supplied model folder and token windows:

```sh
export CARGO_NET_OFFLINE=true
cargo run --release -p mackani -- selfcheck --model-dir MODEL
cargo run --release -p mackani-engine --example verify_model -- MODEL WINDOWS.json --kv kv5 --scalar
cargo run --release -p mackani-engine --example score_windows -- MODEL WINDOWS.json --kv kv5 --backend metal --rows-per-pass 16 --json score-kv5.json
cargo run --release -p mackani-ref --example score_windows_scalar -- MODEL WINDOWS.json --kv kv5
cargo run --release -p mackani --example lossless -- --model-dir MODEL --report lossless.json
```

`verify_model` compares 24 logit rows across supported execution configurations;
`--scalar` adds a direct scalar comparison. Run scoring separately for each
cache contract. Supply `WINDOWS.json` in the `quality::TokenWindows` format:
2,048 tokens/window, scoring positions 1,024 through 2,046 inclusive.
Upstream's window file and recorded scores are in `data/quality/`
(`data/quality/windows_own_thinking.json`); the model folder is an external input. See the
[lossless harness instructions](../crates/mackani-cli/README.md#lossless-development-check).
A passing CPU-only run does not establish Metal execution or another chip's calibration.
