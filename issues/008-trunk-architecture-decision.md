---
id: "008"
title: "Trunk architecture decision: transformer vs large MLP"
type: HITL
status: closed
decision: "Large MLP over engineered features"
adr: "docs/adr/001-trunk-architecture.md"
blocked_by: ["007"]
stories: [30]
---

## Decision (2026-05-15)

**Chosen trunk: Large MLP** over the v1 engineered features
(`tichu_training/featurizer.py`).

Concrete shape (subject to tuning in #009):
4 residual blocks of width 1024 with GELU + LayerNorm, 512-dim trunk output.
Per-task heads are independent on top.

Full rationale: [docs/adr/001-trunk-architecture.md](../docs/adr/001-trunk-architecture.md).

The benchmark in this issue was skipped in favour of an engineering decision
so #009 is not blocked. If BC accuracy plateaus on the held-out split and
failure analysis points at a representational ceiling, the natural follow-up
is the deferred transformer-vs-MLP benchmark on the same split.
---

## What to build

Run a benchmarking experiment to choose the shared trunk architecture before scaling up BC training compute. The two candidates from the PRD are a transformer over the history sequence and a large MLP over engineered features. The decision here locks in the architecture for `#009`, `#010`, `#012`, and `#013`.

**Benchmark setup.** Use a held-out parsed subset (e.g. 50K decisions, stratified by `decision_type`). Train each candidate trunk for a fixed number of steps on the `play` head only (the largest and most representative head). Measure: held-out cross-entropy loss, wall-clock time per training step, GPU memory footprint, inference latency on a single decision.

**Candidates to benchmark:**

- *Large MLP* — engineered hand features (card counts by suit/rank, public trick history as fixed-size vector, partner cards estimate, score delta). Fast to implement, predictable performance, known to work at this scale for similar card games (DouZero baseline).
- *Transformer over history* — encode each played trick as a token, attend over full round history. More expressive but heavier and slower; may overfit at moderate dataset sizes.

**Decision gate.** A human reviews the benchmark results and picks a winner. The choice is documented here (update this issue with the decision before closing) and in an ADR at `docs/adr/001-trunk-architecture.md`. No training at scale begins until this issue is closed.

## Acceptance criteria

- [ ] Both candidates are implemented and trainable on the held-out subset
- [ ] Benchmark results are recorded: held-out loss, step time, memory, inference latency for each candidate
- [ ] A human has reviewed the results and committed a decision
- [ ] `docs/adr/001-trunk-architecture.md` is written with the rationale
- [ ] This issue is updated with the chosen architecture before being closed

## Blocked by

- [#007 Parquet training records pipeline](007-parquet-training-records.md)
