# mackani command and HTTP reference

`mackani` provides terminal chat, an OpenAI-compatible HTTP API, model
inspection, benchmarks and token-reference checks. The default engine runs
compressed `.yskn` models on Apple silicon. `--engine stub` is a development
aid with deterministic output and no inference; it can omit the model folder.

```sh
mackani info --model-dir MODEL
mackani chat --model-dir MODEL --no-think --max-tokens 128
mackani serve --model-dir MODEL --port 8765
mackani selfcheck --model-dir MODEL
mackani bench --model-dir MODEL --draft none --draft both --json bench.json
```

Model files must remain unchanged while the process runs because weights are
mapped read only. Real commands check Apple silicon/macOS compatibility,
macOS 26 or later, required model files and physical-memory capacity before
loading large model data. `info` reports the selected configuration and memory
estimate without loading weights. A failure identifies the unsupported
platform, missing files or memory shortfall and relevant configuration choices.

## Options and environment

Use `mackani --help` for all options or `mackani COMMAND --help` for a command.
All commands accept these options; engine settings apply to the real engine.

| Option | Meaning and default |
| --- | --- |
| `--model-dir FOLDER` | Published model folder; otherwise `MACKANI_MODEL_DIR` |
| `--engine real\|stub` | `real`; `stub` is for development |
| `--ctx N` | Context capacity, 32768; minimum 66, at most the model's maximum |
| `--kv int8\|int8-profile\|kv5` | Key/value cache format, `kv5` |
| `--draft none\|ngram\|neural\|both` | Speculative decoding, `both` |
| `--k N` | Maximum guesses per pass, 7; neural/both accept 1..7 |
| `--depth adaptive\|fixed` | Neural guess-depth policy, `adaptive` |
| `--backend metal\|cpu` | Metal when available, otherwise CPU; explicit unavailable Metal fails |
| `--threads N` | Positive CPU worker count; one per physical core by default |
| `--cpu-share auto\|off\|F` | Metal CPU work fraction, 0..=0.9; `auto` uses the measured Apple M5 table only on the matching 4+6 CPU/10 GPU/10-worker configuration, otherwise off |
| `--prefill-width N` | Cancellable prefill pass width, 16 |
| `--seed0 N` | First automatically assigned request seed, 1 |
| `--help`, `-h` | Command help |

Diagnostics are separate from normal tuning: `--kernel portable` selects the
portable CPU kernel and `--cpu-stages reference` selects reference CPU stages.
Otherwise both are selected automatically. These settings and drafting change
speed, not generated tokens. Adaptive depth chooses 0..K neural guesses from
its deterministic cost policy; fixed depth offers K. `both` uses long n-gram
matches before neural proposals.

The only product environment variables are `MACKANI_MODEL_DIR`,
`MACKANI_API_KEY` and `MACKANI_THINKING_BUDGET`. Explicit folder/key options
win over their variables. Thinking budget sets the chat/server default and
uses 0 for unlimited. Engine tuning has no environment fallback.
`mackani --version` prints the version. Help, version and info output end
successfully when a downstream reader closes its pipe.

### Server

| Option | Meaning and default |
| --- | --- |
| `--host HOST`, `--port PORT` | `127.0.0.1:8765`; port 0 selects an available port |
| `--api-key KEY` | Otherwise `MACKANI_API_KEY`; required for non-loopback binds |
| `--default-max-tokens N` | Default request output limit, 32768 |
| `--max-tokens-cap N` | Optional cap on requested output limits |
| `--no-warmup` | Skip engine warmup |
| `--exit-when-stdin-closes` | Shut down successfully when the parent's input pipe closes |

The desktop app's invocation is supported:
`mackani serve --model-dir F --port P --exit-when-stdin-closes`, with
`MACKANI_API_KEY` supplied in its environment. `/health` answers during loading.
Ctrl+C or SIGTERM shuts down the server and cancels active work.

### Terminal chat

`--system TEXT`, `--max-tokens N`, `--temperature N`, `--top-p N`,
`--top-k N`, `--min-p N` and `--seed N` set the system message and sampling.
The output limit defaults to 32768. Sampling defaults come from the folder's
optional `generation_config.json`, falling back to temperature 1, top-p 0.95,
top-k 20 and min-p 0. `--no-think` disables thinking;
`--effort low|medium|xhigh` sets reasoning effort; `--hide-thinking` hides its
display; `--no-loop-guard` disables repeated-token protection.

Slash commands are `/exit` (also `/quit`, `/q`), `/reset`, `/think on|off`,
`/effort low|medium|xhigh` and `/help`. Ctrl+C stops generation or exits at
the prompt. Standard input/output support pipes. Model text and chat errors
escape terminal control characters except newline/tab, and bidirectional
controls; `--raw` disables that escaping. Top-level errors always escape them.

### Bench and selfcheck

`bench --n N` generates exactly N tokens per prose, code, CSV-to-JSON and
thinking task; N defaults to 512. It uses greedy decoding, ignores EOS and
requires each prompt plus output to fit with the 64-token context reserve.
Repeat `--draft`, `--k` and `--depth` to compare their Cartesian product.
`--json FILE` saves configuration and metrics: prefill/generation rates,
passes, verified rows, guesses offered/accepted and logical code bytes per
pass. The byte count is an estimate of compressed streams and draft head work,
not a measured DRAM counter; it excludes embeddings, vectors, scales and cache.

`selfcheck --reference FILE` defaults to `MODEL_DIR/selfcheck.json`. It compares
greedy short-answer, thinking and plain-text cases against stored prompt and
output token counts, SHA-256 hashes of little-endian u32 IDs and the first
twelve output IDs. Any mismatch fails the command. An optional full `ids`
array permits locating differences beyond the stored head; a digest alone
cannot identify their position.

## HTTP surface

| Request | Response |
| --- | --- |
| `GET /health` | 503 while loading, 200 when ready; no key required |
| `GET /v1/models`, `/v1/models/` | Model list |
| `GET /v1/models/qwen3.8-27b` | Model details |
| `POST /v1/chat/completions` | JSON completion, or SSE when `stream: true` |

Messages support text parts, assistant tool-call history and tool responses.
Function tools support `tool_choice: "auto"` or `"none"`;
`parallel_tool_calls` is accepted and calls are returned in order. Images,
forced tool choices, legacy functions/function_call, logprobs, n > 1 and
non-text response formats are rejected.

Sampling fields are `temperature`, `top_p`, `top_k` (normalized to 1..20),
`min_p` and `seed`. Output controls are `max_tokens`/`max_completion_tokens`,
up to four `stop` strings, `ignore_eos` and `loop_guard`. Thinking accepts
`thinking_budget`, `thinking`, `enable_thinking`, `reasoning_effort` and
`chat_template_kwargs`. Template kwargs take precedence for thinking/effort;
`minimal` aliases `low` and `high` aliases `xhigh`. The reserved template keys
`messages`, `tools`, `add_generation_prompt`, `bos_token`, `eos_token` and
`raise_exception` return 400.

SSE sends the assistant role, token text, complete indexed tool calls, finish
reason, optional final usage (`stream_options.include_usage`) and `[DONE]`.
Reasoning text uses `reasoning_content`. Usage includes prompt, completion,
total, cached prompt and reasoning token counts; loop-guard reports use
`usage.loop_guard`. Context errors return HTTP 400 with
`error.code = "context_length_exceeded"` before streaming headers.

## Security and transport limits

Non-loopback binds require a nonempty API key. On loopback, Host must name
localhost, 127.0.0.1 or [::1] with the actual bound port. POST Origin must be
the same HTTP origin and Sec-Fetch-Site must be same-origin or none.
Absolute-form targets must also name an allowed authority with the HTTP
scheme. Keyed remote binds bypass these browser-origin checks and still
require the key; `/health` remains unauthenticated. Key comparison hashes both
byte strings with SHA-256 and checks all digest bytes.

Defaults below bound untrusted request expansion, connection occupancy and
clients that stop sending or reading. `Limits` exposes these settings to
embedded callers; they are not extra CLI switches.

| Limit | Default | Reason |
| --- | ---: | --- |
| Actual request body | 8 MiB | Room for a large context while bounding bytes before JSON parsing |
| All pending request data | 512 MiB | Holds a largest permitted decoded request while bounding combined memory |
| JSON values per request | 524288 | Bounds parsed objects and their cleaned copies |
| Messages | 65536 | Bounds template expansion, including empty messages |
| Content parts per message | 1024 | Bounds multipart text expansion |
| Tool calls per assistant message | 1024 | Bounds tool history expansion |
| Tools | 1024 | Bounds schemas and template work |
| Renderable text | 2 MiB | Bounds strings/keys and scalar conversion before queueing |
| Rendered prompt | 2 MiB | Bounds tokenizer work after template expansion; excess is 400 `context_length_exceeded` |
| Queue wait | 600 s | Bounds waiting behind the single engine; excess is 503 with `Retry-After: 5` |
| Concurrent connections | 64 (allowed 1..1024) | One engine serializes generation; excess connections close unserved |
| Engine queue capacity | Connection limit | At most one queued/waiting request per connection |
| Buffered events per response | Bounded channel | Backpressure reaches the engine promptly |
| Header buffer / count | 32 KiB / 100 | Bounds request-head memory |
| Complete header / body deadline | 10 s / 30 s | Bounds slow senders |
| Client idle deadline | 300 s | Bounds stalled reads/writes; excludes engine work and queue wait |
| Event-channel backpressure deadline | 30 s | Cancels generation when the consumer stops draining events |
| SSE keep-alive interval | 15 s (allowed 10 ms..1 h) | Keeps silent streams usable through intermediary read timeouts |
| Closing drain / deadline | 256 KiB / 2 s | Gives rejected HTTP bytes a bounded drain before socket close |
| Context reserve | 64 tokens | Leaves room for generation bookkeeping |
| Stop strings / bytes per string | 4 / 1024 | Bounds stop filtering |
| Loop-guard maximum period | 16 tokens | Bounds repeated-pattern search |
| Thinking / answer repeat run | 256 / 1024 tokens | Cuts pathological repetition |
| Thinking digit run | 1000 tokens | Cuts pathological numeric output |

Body memory is charged as bytes arrive. Before parsing, its conservative
reservation is 5 × body bytes + 640 × JSON values; after decoding, it retains
cleaned text bytes + 640 × values until the request ends. Scalar text costs
24 bytes per number, boolean or null. Exceeding the shared budget returns 503
with `Retry-After: 5`. Transfer-Encoding requests are refused. Error excerpts
are bounded and JSON-encoded.

Header/body deadlines apply while receiving those phases, including the next
request on a persistent connection. After the body is complete, no read
idle deadline runs while the request waits or the engine works. The write
idle deadline still applies when bytes cannot move to the client. Generation
has no wall-clock deadline: its prompt/output limits and available context
bound its work. Runtime depends on model, hardware and settings; clients
must allow for long prefill and generation.

Streams send `: keep-alive` comments during silence after prompt preparation.
Non-streaming responses cannot send keep-alives; clients must set their own
read timeout above the expected total runtime. Keep-alives obey socket
backpressure: a silent engine with an unread stream can remain active until
socket buffers fill and the write deadline expires. There is no separate
response-byte budget; the event queue, transport buffers and token limits
bound response retention.

Disconnects and shutdown cancel engine work through callbacks, including
between prefill passes and token commits. A client that half-closes its write
side is treated as disconnected. Pipelined extra bytes can delay disconnect
detection until the next response write. A queued job that times out keeps
its budget share until the engine thread reaches and skips it; a job still
waiting to enter the channel drops its share immediately.

## Lossless development check

```sh
CARGO_NET_OFFLINE=true cargo run --release -p mackani --example lossless -- \
  --model-dir MODEL_FOLDER --profile quick --report lossless.json
```

This example compares committed token IDs, finish reasons and generation
stages through the product tokenizer, chat runner and real engine. It checks
repeatability, seeded sampling, actual draft activity, live/snapshot reuse,
cancellation and recovery. No server socket is needed.

Both profiles use five configurations (none, ngram K=3 fixed, neural K=3
fixed, neural K=7 fixed and both K=7 adaptive), four samplers and two sweeps.
The bundled prompt file has 24 entries. The arithmetic below is an upper
output-token budget, not a measured duration: EOS/stops can shorten it;
loading and prefill add time. Prefill dominates in practice: one `quick` run
on the Apple M5 test machine took 2 hours 50 minutes, of which 2 hours 22
minutes were prefill of 188,312 prompt tokens and 27 minutes generation of
8,103 tokens.

| Profile | Main generation | Other probes | Total | Estimate at 5 tokens/s |
| --- | ---: | ---: | ---: | ---: |
| `quick` (default) | 5 × 4 × 2 × (23 × 8 + 16) = 8000 | 460 unrelated + 205–290 reuse/recovery + 128 seed-only | 8793–8878 | 29.3–29.6 minutes |
| `full` | 5 × 4 × 2 × 1792 = 71680 | Same probe budget | 72473–72558 | About 4.03 hours |

Options are `--model-dir`, `--profile quick|full`, `--prompts FILE`,
`--max-tokens N` (256), `--ctx N`, `--backend cpu|metal`, `--threads N` and
`--report FILE`. Custom prompts or token caps change the arithmetic. The
example exits successfully only when ID comparisons and exercise gates pass;
a partial report preserves completed rows when a later operation fails.
