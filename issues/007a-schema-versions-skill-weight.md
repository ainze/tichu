---
id: "007a"
title: "Parquet records: version stamps + skill_decile join + sample_weight"
type: AFK
blocked_by: ["004", "005", "006"]
parent: "007"
---

## Parent

[#007 Parquet training records pipeline](007-parquet-training-records.md)

## What to build

Bring the per-decision Parquet shards from #004 up to the full PRD training-record schema by wiring in the artifacts produced by #005 and #006.

**Stamp version fields.** Every emitted row's `featurizer_version` and `action_space_version` columns hold the live module constants (`FEATURIZER_VERSION`, `ACTION_SPACE_VERSION`).

**Skill join.** `parse_bsw` gains `--trueskill <ratings_parquet>`. When supplied, the records' new `skill_decile: int32 nullable` column is populated by looking up `player_handle` in the ratings table; cold-start players (handle absent / null decile) keep `null`. When the flag is absent, `skill_decile` is null for every row (intermediate / un-joined mode).

**Recency weighting.** New `sample_weight: float32` column.

- `--recency-cutoff-game-id N` (default `1855844` — the first 2015 game).
- `--recency-weight W` (default `0.5`) — used for games whose `game_id` is below the cutoff.
- Rows from games with `game_id >= N` get `1.0`; below-cutoff games get `W`.

**State / legal_actions_mask.** Stay null in this slice. Featurizing 2.4M games at parse time would 10× both the parquet size and the parse runtime, so featurization is deferred to the training-loop loader (#009). The columns remain reserved with the correct types.

**Schema (final v1).** Columns in this exact order:

`decision_type, game_id, round_id, timestamp, player_handle, action_taken, state, legal_actions_mask, round_outcome, round_won, featurizer_version, action_space_version, skill_decile, sample_weight`

## Acceptance criteria

- [ ] Every emitted row has `featurizer_version == FEATURIZER_VERSION` and `action_space_version == ACTION_SPACE_VERSION`
- [ ] `--trueskill` flag accepts a ratings parquet (output of `compute_trueskill`) and joins `skill_decile` on `player_handle`
- [ ] Rated players get their stored `skill_decile`; absent handles get `null`
- [ ] Without `--trueskill`, `skill_decile` is null in every row
- [ ] `sample_weight` is `1.0` when `game_id >= --recency-cutoff-game-id` (default `1855844`), else `--recency-weight` (default `0.5`)
- [ ] Schema column order matches the spec exactly
- [ ] CLI runs end-to-end on `sample/` (game_ids 2417500 & 2417501 — both above the default cutoff)

## Blocked by

- [#004 BSW parser + replay validation](004-bsw-parser-replay-validation.md)
- [#005 TrueSkill rating computation](005-trueskill-ratings.md)
- [#006 Featurizer + canonical action space](006-featurizer-action-space.md)
