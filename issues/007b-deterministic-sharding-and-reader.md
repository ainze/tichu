---
id: "007b"
title: "Deterministic shard naming + version-aware reader"
type: AFK
blocked_by: ["007a"]
parent: "007"
---

## Parent

[#007 Parquet training records pipeline](007-parquet-training-records.md)

## What to build

Two pieces, both small:

**Deterministic shard naming.** Switch `parse_bsw` output from `play.parquet` to `play_00000.parquet` (and analogous for every decision type) so a training job can list shards via `glob("play_*.parquet")` without consulting any manifest. v1 emits exactly one shard per decision type (suffix `_00000`); future multi-shard sharding can fill in higher indices without changing the loader contract.

**Reader utility with version-mismatch detection.** `tichu_training.records.load_shards(directory, decision_type, *, expected_featurizer_version, expected_action_space_version) -> pa.Table`:

- Globs `{decision_type}_*.parquet` under `directory`, in sorted name order.
- Reads each shard and concatenates the resulting `pa.Table`.
- After load, verifies that every distinct value in the two version columns matches the loader's expectation.
- On mismatch, raises `VersionMismatchError` (the same class introduced in #006c) with a message naming the column, the stored values seen, and the expected value.

## Acceptance criteria

- [ ] `parse_bsw` writes `play_00000.parquet`, `pass_card_00000.parquet`, etc.
- [ ] Existing `--input sample/` runs still succeed and produce the renamed shards
- [ ] `load_shards(dir, "play", expected_featurizer_version="v1", expected_action_space_version="v1")` returns a table
- [ ] Passing a wrong `expected_featurizer_version` raises `VersionMismatchError` with both values + the column name in the message
- [ ] Passing a wrong `expected_action_space_version` raises `VersionMismatchError` with both values + the column name
- [ ] Reader handles the multi-shard case correctly (test with two manually-written shards on disk)

## Blocked by

- [#007a Schema + version stamps + skill + weight](007a-schema-versions-skill-weight.md)
