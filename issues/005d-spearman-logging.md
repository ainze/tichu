---
id: "005d"
title: "TrueSkill: Spearman ρ sanity logging"
type: AFK
blocked_by: ["005b", "005c"]
parent: "005"
---

## Parent

[#005 TrueSkill rating computation](005-trueskill-ratings.md)

## What to build

A cheap printed sanity check that the TrueSkill ratings rank players consistently with an independent skill proxy (Tichu success rate). Print-only — no test gating on the magnitude — because the sample corpus is too small to be meaningful and the real check happens on full-corpus runs.

**Computation.** After writing the ratings Parquet, compute Spearman rank correlation between `mu` and `tichu_success_rate` over the subset of rated players who have `tichu_calls >= 5` (so the rate is non-noisy). If fewer than 5 players qualify, skip and log a warning.

**Implementation.** Stdlib only — implement Spearman ρ as Pearson correlation on ranks (with average ranks for ties). No new dependency.

**Output.** A single `INFO` log line:

```
spearman rho(mu, tichu_success_rate) = 0.412 over 318 players (tichu_calls >= 5)
```

Or when too few players qualify:

```
WARNING: only 2 players have tichu_calls >= 5; skipping spearman rho check
```

Also report decile share in the same logging block:

```
decile share (0..9): [0.099, 0.101, ...]
```

## Acceptance criteria

- [ ] CLI prints the ρ line (or the skip warning) at the end of a successful run
- [ ] CLI also prints the decile-share line
- [ ] Implementation has no new pip dependency
- [ ] Unit test verifies the Spearman implementation against a tiny hand-checked input (e.g. ρ for `[1,2,3,4]` vs `[2,1,4,3]` is exactly 0.6)
- [ ] CLI run with `--min-games` set absurdly high (so no player survives) exits cleanly with the skip warning, no traceback

## Blocked by

- [#005b min-games and decile](005b-min-games-and-decile.md)
- [#005c Tichu success rates](005c-tichu-success-rates.md)
