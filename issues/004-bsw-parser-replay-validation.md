---
id: "004"
title: "BSW parser + replay validation"
type: AFK
blocked_by: ["002"]
stories: [17, 18, 19, 21, 22]
---

## What to build

Implement a parser that converts raw Brettspielwelt (BSW) log files into structured, per-decision training records, with replay validation as the primary correctness gate.

**Parser.** Read BSW log files and emit one record per player decision. Preserve: `game_id`, `round_id`, `timestamp`, `player_handle` (stable across the entire corpus), the full game sequence (all four hands, all actions, scores). Incomplete or abandoned games must be skipped gracefully with a log entry — they must not corrupt output or crash the run.

**Replay validation.** For every parsed game, replay the full action sequence through the rules engine and assert that the engine's final `Ergebnis` (score) matches BSW's recorded result. This is the highest-leverage test in the project: a 99.9% match rate across valid games catches the vast majority of parser bugs and engine bugs simultaneously. Games that fail replay are logged with their `game_id` and added to a `known_bad_games.txt` allowlist rather than blocking CI. Nightly and weekly runs increase the replay subset size (1K in CI, 10K nightly, full corpus weekly).

**Output format.** Emit Parquet files. Each record must include all fields needed for the training schema defined in the PRD: `state` (serialized `PrivateState`), `legal_actions_mask`, `action_taken`, `decision_type` (one of `play`, `pass_card`, `call_tichu`, `call_grand_tichu`, `wish_rank`, `dragon_give`), `player_handle`, `round_outcome`, `round_won`, `game_id`, `round_id`, `timestamp`. Shard output by `decision_type` so each model head can load only its relevant data. `featurizer_version` and `action_space_version` fields are reserved but may be written as `null` until those components exist.

**CLI.** `parse_bsw --input <dir> --output <dir> --subset <n>` produces validated Parquet shards.

## Acceptance criteria

- [ ] Parser processes the full BSW corpus without crash; incomplete games are skipped with log entries
- [ ] Replay validation passes on ≥ 99.9% of valid games in a fixed 1,000-game CI subset
- [ ] Output Parquet files are sharded by `decision_type`
- [ ] All required training-record fields are present in output schema
- [ ] `player_handle` is preserved and consistent across games (same string for the same BSW handle)
- [ ] `game_id` and `timestamp` are present and correct
- [ ] `known_bad_games.txt` is produced; games on it are excluded from the replay-validation pass-rate denominator
- [ ] `parse_bsw` CLI runs end-to-end on a small sample

## Blocked by

- [#002 Rules engine](002-rules-engine.md)
