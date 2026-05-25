# ADR-0003: Difficulty "easy" is a Baseline, not a weak model

- **Status:** Accepted
- **Date:** 2026-05-25
- **Related:** [ADR-0001](0001-trunk-architecture.md), [ADR-0002](0002-intent-level-action-space.md)

## Context

The inference service exposes four Difficulty levels — `easy`, `medium`,
`hard`, `master` — each mapping to a concrete Agent via the Difficulty Spec
loaded at startup (`tichu_inference/app.py`). The natural assumption is that
each tier is a progressively stronger ML Agent built from a progressively
better Checkpoint (e.g. `easy = small BC`, `master = BC + AWR refine`).

The PRD originally suggested exactly that scheme ("Easy: rule-based, Medium:
small BC, Hard: full BC, Master: BC plus offline-RL refinement"). The first
bullet — `easy = rule-based` — is preserved in the current implementation,
but it deserves an explicit ADR because it is surprising on second reading
and shapes what we owe each tier.

## Decision

**Difficulty `easy` is wired to the `RuleAgent` Baseline, not to a weak ML
Agent.** Concretely, `tichu_inference/app.py:101` instantiates
`RuleAgent()` for the `easy` slot. `medium`, `hard`, `master` are ML Agents
differing only in which Checkpoint they load.

## Rationale

1. **A deterministic Baseline is more pedagogically useful for beginners.**
   New players need predictable opponents that consistently apply legible
   heuristics. A weak ML Agent produces probabilistic, often-illegible
   behaviour that is *bad* without being *teachable*.
2. **A weak ML Agent is harder to keep weak.** Any improvement to the BC
   pipeline raises the floor of the smallest checkpoint; the `easy` tier
   would drift upward over time, breaking the difficulty contract.
3. **Lower operational cost.** The Baseline does not need torch loaded,
   does not need GPU/CPU inference budget, and has near-zero startup time.
4. **The `easy` tier doubles as a fallback target.** If an ML Agent
   produces malformed output, the service falls back to a uniformly random
   legal action (PRD #52). A Baseline is the closest in spirit to that
   fallback — both are non-learned, both are guaranteed legal.

## Consequences

- Tuning `easy` is a code change to `RuleAgent`, not a training-config
  change. There is no Checkpoint to swap.
- `easy` cannot be evaluated against the rest of the Tournament matrix
  with the same "what Checkpoint produced this row" provenance — the
  Baseline has no training lineage. The eval harness must accept Baselines
  as first-class rows.
- The Difficulty contract for Phase 1 is:
  - `easy` = RuleAgent Baseline.
  - `medium` = ML Agent on a small BC Checkpoint.
  - `hard` = ML Agent on a full BC Checkpoint.
  - `master` = ML Agent on a BC + AWR-refined Checkpoint.
- A future "extra easy" or "tutorial" tier should also be a Baseline, not
  a deliberately broken model.

## Rejected alternative

**`easy` as a small ML Agent.** Rejected for the four reasons above. The
loss of a "tier of ML difficulty" is acceptable because three ML tiers
(medium / hard / master) already span the practical strength range of
the BC + AWR pipeline.
