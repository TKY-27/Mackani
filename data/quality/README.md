# Quality measurement data

These files come from YASAKANI (https://github.com/yasakani-dev/yasakani,
commit 758ef28, Apache-2.0), copied unchanged from `tools/quality/`.

- `windows_own_thinking.json`: the fixed token windows of upstream's
  perplexity measurement.
- `score_fastest_kv5_table_reference.json`,
  `score_fastest_int8_table_reference.json`: upstream's recorded results for
  the "fastest" model with the 5-bit and the int8 cache. The perplexity to
  reproduce is the `ppl` field (2.075937 with the 5-bit cache).

`cargo run --release -p mackani-engine --example score_windows` computes
the same measurement with Mackani.
