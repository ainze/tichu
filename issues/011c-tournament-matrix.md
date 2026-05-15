---
id: "011c"
title: "All-vs-all tournament with seat-swap, paired-difference matrix + bootstrap CIs"
type: AFK
parent: "011"
blocked_by: ["011b"]
---

## What to build

Orchestrate every pairing of named agents over the fixed deal pool and produce the project's headline quality matrix.

- `run_tournament(agents: dict[str, Agent], deals: list[GameState], *, bootstrap_iters: int = 1000, seed: int = 0) -> MatrixResult` — for each ordered pair `(A, B)` with `A != B`, play every deal twice:
    - **arrangement 1**: A in seats (0, 2), B in seats (1, 3) → record `score_delta = team0 - team1`
    - **arrangement 2**: seat-swap by rotating one position → A in seats (1, 3), B in seats (0, 2) → record `score_delta = team1 - team0`
  This cancels first-player and team-assignment variance.
- Aggregate per-pair: mean delta of A vs B over `2 * len(deals)` rows, plus a 95% bootstrap CI computed from those rows with `np.random.default_rng(seed)`.
- `MatrixResult` exposes both:
    - `.mean_matrix` (dict-of-dicts or 2D array indexed by agent names)
    - `.ci_lower`, `.ci_upper` (same shape)
    - `.rows`: tidy DataFrame-shaped records (`agent_a`, `agent_b`, `mean`, `ci_lower`, `ci_upper`, `n`)

## Acceptance criteria

- [ ] `run_tournament({"random": ra, "rule": rb}, deals_100)` returns a `MatrixResult` whose `rule` vs `random` mean delta is positive (rule wins).
- [ ] Each pair has `n = 2 * len(deals)` paired observations (seat-swap accounted for).
- [ ] Bootstrap CI is deterministic given the same `seed` and same per-deal deltas.
- [ ] Diagonal (`A` vs `A`) is reported as `0.0` mean with `[0, 0]` CI, or excluded — pick one and document.

## Blocked by

- [#011b Single-deal runner](011b-play-deal.md)
