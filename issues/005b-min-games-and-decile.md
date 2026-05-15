---
id: "005b"
title: "TrueSkill: min-game threshold and skill decile bucketing"
type: AFK
blocked_by: ["005a"]
parent: "005"
---

## Parent

[#005 TrueSkill rating computation](005-trueskill-ratings.md)

## What to build

Filter out low-volume players and bucket the surviving population by skill so the training pipeline has a categorical conditioning signal.

**Min-games threshold.** Add `--min-games N` (default `20`) to the CLI. Players with `n_games < N` are dropped from the output table. Their handles are still observable in the parsed-decision shards (#004 output) — the rating table is just narrower.

**Skill decile.** Compute `skill_decile` (int32, 0–9) on the surviving population from `mu` quantiles. Decile 0 = lowest 10% by `mu`, decile 9 = top 10%. Ties broken by `sigma` (lower σ → higher decile), then by `player_handle` lexicographic. Emit as an additional column.

## Acceptance criteria

- [ ] `--min-games` flag respected; below-threshold players absent from the output
- [ ] Output Parquet adds `skill_decile: int32` column
- [ ] On any input where the surviving population has ≥ 10 players, every decile 0..9 contains at least one player
- [ ] On the surviving population, no single decile holds > 15% of rated players (the spec's uniformity sanity check)
- [ ] Default threshold is 20 when `--min-games` is omitted

## Blocked by

- [#005a TrueSkill tracer](005a-trueskill-cli-tracer.md)
