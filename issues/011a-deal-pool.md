---
id: "011a"
title: "Fixed seeded deal pool (generate, persist, load)"
type: AFK
parent: "011"
blocked_by: []
---

## What to build

A deterministic pool of starting deals used as the immutable benchmark for every tournament run.

- `generate_deal_pool(seed: int, n: int) -> list[GameState]` — produces `n` post-schupfen starting states by calling `deal_initial_state(seed + i)` for `i` in `range(n)`. Schupfen is skipped at this layer (the harness measures play strength, not schupfen heuristics).
- `save_deal_pool(deals, path)` / `load_deal_pool(path) -> list[GameState]` — round-trippable serialization (Parquet, one row per deal with serialized hands + starting player).
- The pool's identity is `(seed, n)`; same inputs across two processes must produce identical hands, identical starting players, and identical serialized files.

## Acceptance criteria

- [ ] `generate_deal_pool(seed=0, n=10)` returns 10 `GameState`s; calling twice produces equal hand sets.
- [ ] `save_deal_pool` writes a Parquet file; `load_deal_pool` returns equal `GameState`s.
- [ ] Two saves of the same pool produce byte-identical files (or at least equal-after-load).
- [ ] All hands have 14 cards; full pool covers all 56 cards per deal.

## Blocked by

- None — can start immediately.
