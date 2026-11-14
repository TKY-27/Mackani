# Changelog

## 0.1.0

First release of Mackani, an unofficial Apple Silicon port of YASAKANI.

- Rust reader for published `.yskn` model folders and an independent scalar reference.
- Metal compressed projections, NEON CPU stages and a thread pool, with bit-for-bit reference comparisons.
- Terminal chat, a local OpenAI-compatible streaming HTTP API, benchmark, selfcheck and model/memory information.
- A native SwiftUI chat app that manages its bundled server.
- Build instructions, behaviour specifications and third-party notices.

Tested only with the speed model on a base Apple M5 MacBook with 24 GB memory and macOS 27. No prebuilt binaries.
