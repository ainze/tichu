---
id: "014c"
title: "`benchmark_p99` latency utility + slow-marked CI gate"
type: AFK
parent: "014"
blocked_by: ["014a"]
---

## What to build

- `benchmark_p99(module, *example_inputs, n: int = 1000) -> dict[str, float]` — runs `n` sequential forward passes (after a 10-iter warmup), returns `{"p50", "p95", "p99", "mean"}` in milliseconds.
- A `@pytest.mark.slow` test that traces a production-size `BCModel` (trunk_hidden=1024, trunk_depth=4) and asserts `p99 < 500.0`.
- `export_model` CLI gains `--benchmark` flag: traces, then prints the four percentiles. Non-zero exit if `p99` exceeds the configured budget (`--p99-budget-ms`, default 500).

## Acceptance criteria

- [ ] `benchmark_p99(BCModel, ...)` returns a dict with the four documented keys; all values are positive floats.
- [ ] On a tiny BCModel (depth=1, hidden=16) the test passes well under 50 ms p99.
- [ ] Slow-marked test on production-size BCModel asserts `p99 < 500.0` and is deselected by `-m 'not slow'`.
- [ ] CLI prints the four stats when `--benchmark` is passed, with a clear "p99 EXCEEDED budget" message + non-zero exit on overshoot.

## Blocked by

- [#014a TorchScript export primitive](014a-torchscript-export.md)
