# Upstream behaviour specification

This specification describes YASAKANI's behaviour for this Rust port.
It is written from upstream source at commit
[758ef28](https://github.com/yasakani-dev/yasakani/tree/758ef28).
Source paths and line references in the chapters refer to that revision.
They describe upstream execution, including its CUDA-specific operations;
they do not assert portable equality with a native CUDA binary.

The [architecture](../ARCHITECTURE.md) describes current Rust execution.
The [design notes](../DESIGN_NOTES.md#resolved-specification-boundaries) state
resolved port choices, validation rules and numerical compatibility limits.
Native-function and instruction limitations stay beside the relevant algorithms.

| Chapter | Subject |
|---|---|
| [01](01_MODEL_ARCHITECTURE.md) | Model architecture and dimensions |
| [02](02_COMPRESSED_LINEAR.md) | Compressed linear operator and embeddings |
| [03](03_GATED_DELTANET.md) | Gated DeltaNet |
| [04](04_ATTENTION.md) | Attention and KV encodings |
| [05](05_RESIDUAL_NORM_MLP.md) | Residual, normalization and MLP |
| [06](06_OUTPUT_HEAD_SAMPLER.md) | Output head, sampler and quality score |
| [07](07_DRAFT_MODEL.md) | Block draft and reduced head |
| [08](08_SPECULATION.md) | Speculation, state and upstream policies |
| [09](09_LONG_PROMPT_PREFILL.md) | Long-prompt prefill |
| [10](10_AUXILIARY_FILES.md) | Model-folder and auxiliary file layouts |
| [Glossary](GLOSSARY.md) | Upstream identifiers and their meanings |

[FORMAT.md](FORMAT.md) is upstream's own format document, copied unchanged
from that commit under the Apache License 2.0, with an attribution line added.
Its original body, including CRLF line endings and relative links, is preserved
byte for byte. Those archival links use the upstream document's original
location, `docs/FORMAT.md`; follow them in the
[upstream tree](https://github.com/yasakani-dev/yasakani/tree/758ef28/docs).
The English algorithm chapters and Rust doc comments use the local copy.
