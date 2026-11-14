# Building Mackani

## Prerequisites

Use an Apple Silicon Mac with macOS 26 or later. Install Rust through rustup and the Xcode command line tools. [rust-toolchain.toml](rust-toolchain.toml) selects stable Rust with rustfmt and Clippy. The native app requires Swift 6.2. Metal shader compilation also needs the Apple developer tools.

Only a base M5 MacBook with 24 GB memory and macOS 27 has been tested. See [README.md](README.md) for the model download, memory estimate and untested configurations. There are no prebuilt binaries.

## Command-line binary

Run from the repository root:

```sh
cargo build --release -p mackani
```

The result is `target/release/mackani`.

## Native app

After building the binary, run from the repository root:

```sh
./app/build-app.sh
open app/build/Mackani.app
```

The script builds the Swift package in release mode, copies the server into `app/build/Mackani.app` and signs the bundle ad hoc. The app is not notarized. It starts and stops its bundled server itself. See [app/README.md](app/README.md) for app settings and local data storage.

## Tests

From the repository root, run the Rust checks used by CI:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

CI uses an Apple Silicon macOS 26 runner and stable Rust. It sets `CARGO_PROFILE_DEV_DEBUG=0` and `CARGO_PROFILE_TEST_DEBUG=0` to omit debug symbols. The workspace's dev profile keeps assertions and overflow checks enabled with optimization.

Run the Swift tests separately:

```sh
cd app && swift test
```

Some Rust tests need the published model folder. Set `MACKANI_MODEL_DIR` to that folder to enable the model-dependent checks; without it, those tests report a skip. A workspace test pass without the model does not prove real-model output.

## Before changing arithmetic

Rebuild the binary and check the tested published model:

```sh
cargo build --release -p mackani
./target/release/mackani info --model-dir ~/models/qwen38-speed
./target/release/mackani selfcheck --model-dir ~/models/qwen38-speed
```

Run the relevant bit-for-bit comparisons with `mackani-ref` as well as the full Rust checks. The engine crate has verification examples for real-model comparisons and quality measurements; [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes them. Use those checks before accepting an arithmetic change. Synthetic tests alone do not establish parity on the published weights. Leave model files unchanged while any check is running.
