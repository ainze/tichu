---
id: "007"
title: "Parquet training records pipeline"
type: AFK
blocked_by: ["004", "005", "006"]
stories: [20, 24]
---

## What to build

Finalize the training record schema and wire together the parser output, TrueSkill table, and featurizer version into complete, ML-ready Parquet shards.

**Schema finalization.** Extend the raw parser output from `#004` with the fields that require `#005` and `#006` to be available:

- `featurizer_version` — stamped from the active featurizer
- `action_space_version` — stamped from the active action space
- `skill_decile` — joined from the TrueSkill table on `player_handle`; cold-start players (below minimum-game threshold) receive `null` or a sentinel value

All fields from the PRD schema must be present: `state`, `legal_actions_mask`, `action_taken`, `decision_type`, `player_handle`, `round_outcome`, `round_won`, `game_id`, `round_id`, `timestamp`, `featurizer_version`, `action_space_version`.

**Skill conditioning.** Training records for players below the minimum-game threshold are kept in the dataset (they are not dropped). The `skill_decile` field being `null` is the signal to the training loop to assign a neutral skill embedding rather than a learned bucket embedding.

**Data filtering and weighting.** Implement the default recency weighting: games post-2015 are retained at full weight; older games are either downweighted or excluded via a config flag. The weight is stored as a `sample_weight` column. Hard skill filtering (keeping only top-N deciles) is available as a config option but is not the default.

**Sharding.** Output remains sharded by `decision_type`. Shards are named deterministically (e.g. `play_00000.parquet`) so training jobs can list them without a manifest file.

**CLI.** `parse_bsw` gains a `--trueskill <ratings_parquet>` flag that, when provided, joins skill data and stamps version fields. Without the flag it emits the raw schema for intermediate use.

## Acceptance criteria

- [ ] Output Parquet schema matches the full PRD training-record schema exactly
- [ ] `featurizer_version` and `action_space_version` are correct constants in every row
- [ ] `skill_decile` is populated for rated players and `null` for cold-start players
- [ ] `sample_weight` column is present; post-2015 games have weight 1.0 by default
- [ ] Shards are deterministically named and loadable by `decision_type` without a manifest
- [ ] Loading records with a mismatched `featurizer_version` raises or warns loudly in the training loop
- [ ] Full pipeline runs end-to-end on a small corpus sample

## Blocked by

- [#004 BSW parser + replay validation](004-bsw-parser-replay-validation.md)
- [#005 TrueSkill rating computation](005-trueskill-ratings.md)
- [#006 Featurizer + canonical action space](006-featurizer-action-space.md)
