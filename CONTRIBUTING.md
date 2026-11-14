# Contributing to Mackani

## Preserve the output bits

Every fast path must equal the scalar reference in `mackani-ref` bit for bit. A change that alters output bits is a bug unless upstream's specified behaviour requires it. Keep the reference independent of the optimized implementation. Speed does not justify a different answer.

For a behaviour correction, identify the upstream evidence in [docs/spec/](docs/spec/) and preserve a regression check against that evidence. For a fast-path change, compare with the scalar reference across the affected inputs and execution paths.

## Checks and style

Run the formatting, Clippy and workspace tests in [BUILDING.md](BUILDING.md). For app changes, also run the Swift tests there. Before changing arithmetic, rebuild and run real-model `selfcheck` and the relevant verification examples described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). State which model-dependent checks ran and which were skipped.

Keep changes focused. Follow the surrounding Rust or Swift style. Comments should explain why a decision is necessary. Add a focused regression test for a behaviour change, using an independent expected result.

## Report wrong output

Open an issue at [TKY-27/Mackani](https://github.com/TKY-27/Mackani/issues) with:

- The model repository and the model files used.
- The exact command and settings, the input, the expected result and the actual result.
- The output of `./target/release/mackani info --model-dir ~/models/qwen38-speed`, using your actual model folder.
- Whether `./target/release/mackani selfcheck --model-dir ~/models/qwen38-speed` passes with that folder.

Remove API keys, private prompts and personal path details from anything you share. Report vulnerabilities privately using [SECURITY.md](SECURITY.md).

Questions about the model format, conversion or published model files belong with [YASAKANI upstream](https://github.com/yasakani-dev/yasakani). Bugs in Mackani's reader belong here.

Contributions must be work you can license under this project's Apache License 2.0. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
