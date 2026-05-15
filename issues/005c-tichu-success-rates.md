---
id: "005c"
title: "TrueSkill: Tichu and Grand Tichu success rates"
type: AFK
blocked_by: ["005a"]
parent: "005"
---

## Parent

[#005 TrueSkill rating computation](005-trueskill-ratings.md)

## What to build

Per-player call success rates for Tichu and Grand Tichu, computed during the same corpus sweep and emitted as auxiliary columns on the ratings table.

**Definitions.**

- A Tichu call **succeeds** for player `p` in a given round iff `p` was in `tichu_callers` for that round AND `p` was the first player out in that round.
- A Grand Tichu call **succeeds** identically but against `grand_tichu_callers`.
- "First player out" = `out_order[0]` from engine state at round end (already tracked in #004's replay). Where replay is unavailable, fall back to `round.ergebnis[team(p)] - round.ergebnis[1-team(p)] >= 100` as a proxy *for the purposes of this rate only* — Tichu success is worth +100 to the caller's team.

**Counters.** Maintain per-player `tichu_calls`, `tichu_wins`, `grand_tichu_calls`, `grand_tichu_wins` across the sweep.

**Output columns.** Append to the ratings table:

- `tichu_calls: int32`
- `tichu_success_rate: float64` — `tichu_wins / tichu_calls` (null if `tichu_calls == 0`)
- `grand_tichu_calls: int32`
- `grand_tichu_success_rate: float64` — same shape (null if 0 calls)

## Acceptance criteria

- [ ] Output Parquet adds the four columns listed above
- [ ] On the `sample/` corpus, any player who called Tichu has `tichu_calls >= 1` and a non-null `tichu_success_rate`
- [ ] Players who never called keep `tichu_calls == 0` and `tichu_success_rate IS NULL`
- [ ] Rates are bounded to `[0.0, 1.0]`

## Blocked by

- [#005a TrueSkill tracer](005a-trueskill-cli-tracer.md)
