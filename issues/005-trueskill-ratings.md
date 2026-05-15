---
id: "005"
title: "TrueSkill rating computation"
type: AFK
blocked_by: ["004"]
stories: [23, 25, 26]
---

## What to build

Compute a TrueSkill rating for every player in the BSW corpus and emit a table that the training pipeline consumes as an input artifact.

**Rating computation.** Sweep the parsed corpus in chronological order. For each completed game, update TrueSkill ratings for all four participants using the game's team outcome (win/loss + point margin can be used as a proxy for match result). Emit a table with one row per player: `player_handle`, `mu`, `sigma`, `n_games`.

**Minimum-game threshold.** Players with fewer than a configurable minimum number of games (default: 20) are excluded from the output table or flagged as cold-start. Their decisions are still in the training data but receive no skill conditioning signal — they are assigned a neutral/population-mean skill bucket.

**Skill buckets.** Divide the resulting `mu` distribution into deciles and add a `skill_decile` (0–9) column. At inference time the top decile (9) is always selected. This bucketed value is what the model's skill-conditioning embedding consumes.

**Auxiliary signals.** Compute per-player Tichu call success rate (calls made vs calls won) and Grand Tichu call success rate as cross-validation signals. These do not replace TrueSkill; they are sanity checks that the TrueSkill ranking correlates with observable skill indicators. Emit them as additional columns in the output table.

**CLI.** `compute_trueskill --input <parsed_parquet_dir> --output <ratings_parquet>` runs the sweep and emits the table.

## Acceptance criteria

- [ ] TrueSkill ratings computed in chronological game order for all players in the corpus
- [ ] Output table contains `player_handle`, `mu`, `sigma`, `n_games`, `skill_decile`
- [ ] Players below the minimum-game threshold are excluded or flagged
- [ ] Tichu and Grand Tichu success rates present as auxiliary columns
- [ ] `skill_decile` distribution is approximately uniform (sanity check: no single decile holds > 15% of rated players)
- [ ] TrueSkill ranking correlates positively with Tichu success rate (Spearman ρ > 0.3)
- [ ] `compute_trueskill` CLI runs end-to-end and output is a valid Parquet file

## Blocked by

- [#004 BSW parser + replay validation](004-bsw-parser-replay-validation.md)
