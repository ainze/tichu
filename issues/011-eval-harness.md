---
id: "011"
title: "Evaluation harness (tournament + held-out move prediction)"
type: AFK
blocked_by: ["003"]
stories: [40, 42, 43, 44, 45]
---

## What to build

Implement the tournament evaluation harness that is the project's north-star quality metric. Every checkpoint must have an eval row before it ships; no model comparison claim is valid without it.

**Deal pool.** Generate and persist a fixed pool of 10,000 deals (4-player card distributions, seeded and deterministic). This pool never changes for the lifetime of the project. Every tournament run uses exactly this pool.

**Tournament orchestration.** Run all-vs-all matches between named agents (registered by name in the agent registry from `#003`). For each pairing, play every deal in the pool twice with seat-swap (rotating by one position) to cancel team-assignment variance. Record per-deal score deltas for each pairing.

**Output matrix.** Produce a paired-difference matrix: for each (agent_A, agent_B) pair, report the average score delta (A minus B, per round) with a 95% bootstrap confidence interval. The matrix is stored as a Parquet file and also pretty-printed to stdout.

**Determinism.** Eval mode forces all agents into deterministic action selection (argmax or fixed RNG seed). The same config and seed must produce identical matrix entries across two runs. Reproducibility test: run the harness twice with `RandomAgent` vs `RuleAgent` on a small deal pool; assert matrix entries are identical.

**Smoke test.** `RandomAgent` vs `RuleAgent` on 100 deals. Assert `RuleAgent` wins by a positive average margin. This catches a broken orchestrator, broken legality enforcement, or broken score accounting in one test.

**Held-out move prediction.** A separate evaluation mode loads a held-out set of BSW games (not in the training data), runs the agent's policy on each decision state, and reports top-1 and top-5 accuracy vs the human action taken. This measures imitation quality independently of self-play results.

**CLI.** `eval_matrix --config <path>` runs a tournament. Config specifies agent names + checkpoint paths, deal pool path, number of deals, and output path.

## Acceptance criteria

- [ ] Fixed 10,000-deal pool is generated, seeded, and persisted; never regenerated between runs
- [ ] All-vs-all tournament runs correctly for any set of registered agents
- [ ] Seat-swap variance reduction is applied for every pairing
- [ ] Output is a paired-difference matrix with bootstrap CIs
- [ ] Determinism test passes: two identical runs produce identical matrix entries
- [ ] Smoke test passes: `RuleAgent` beats `RandomAgent` by positive margin
- [ ] Held-out move prediction mode reports top-1 and top-5 accuracy per decision type
- [ ] Every eval run is reproducible from config + seed
- [ ] `eval_matrix` CLI is functional

## Blocked by

- [#003 Baseline agents + agent-interface contract test](003-baseline-agents-contract-test.md)
