# ADR-0013: Parquet schema is versioned by directory name, not by a per-row column

- **Status:** Accepted
- **Date:** 2026-05-28
- **Related:** [ADR-0009](0009-bsw-ingest-streaming-pipeline.md), [ADR-0011](0011-bc-training-replay-on-the-fly.md)

## Context

The parquet shards produced by `parse_bsw` carry two per-row version
columns: `featurizer_version` (stamped from
[FEATURIZER_VERSION](../../src/tichu_training/featurizer.py)) and
`action_space_version` (stamped from
[ACTION_SPACE_VERSION](../../src/tichu_training/action_space.py)). The
training loop checks these against the live module constants on every
Checkpoint load — a mismatch is a hard error. This is the only
versioning surface in the parquet pipeline today.

The AWR `game_outcome` redesign adds a new column (`game_won: bool`) to
the parquet schema without changing the featurizer, the action space,
or the meaning of any existing column. The handoff document raised the
question explicitly: "this implies bumping `featurizer_version` to v3
(since the parquet schema changed, even though the featurizer itself
didn't). Or — alternative — keep featurizer at v2 and bump a separate
`schema_version` column."

Both options are wrong:

1. **Bumping `featurizer_version` corrupts the term.** Per
   [CONTEXT.md](../../CONTEXT.md), the Featurizer is "the pure function
   `featurize(PrivateState) -> Feature Vector`." Bumping the version
   string to signal a schema change would invalidate previously-trained
   v2 Checkpoints at load time even though their features are bit-
   identical to v3 featurizer output. Semantic pollution that breaks
   the version-pin check for unrelated reasons.

2. **A per-row `parquet_schema_version` column is wasted bytes.** Arrow
   enforces per-shard schema homogeneity — a schema cannot change mid-
   shard. The column would carry the same value on every row of every
   shard. Detecting schema by column-presence
   (`"game_won" in shard.schema.names`) is one line of reader code and
   zero bytes of storage. The per-row pattern existed for
   `featurizer_version` because of a defensible-if-paranoid concern
   that the featurizer constant could be edited mid-corpus during a
   long parse. Schema can't change mid-corpus — Arrow won't allow it.

The convention already in flight on disk
(`parquet_100k` → `parquet_100k_v2` → `parquet_100k_v3`) is the
de-facto schema version mechanism. Humans use it to tell shards apart;
the file system enforces it.

## Decision

**The parquet shard directory name is the canonical schema version.
There is no per-row `parquet_schema_version` column. Readers detect
schema features by column-presence.**

Concrete shape:

- Output directories follow `parquet_<scale>_v<N>` where `N` is bumped
  on every schema change. Scales seen today: `100k`, `1m`, `full`.
- `featurizer_version` and `action_space_version` columns are
  preserved unchanged — they version what their names say and nothing
  else. Adding a parquet column does not bump them.
- Reader code that needs to distinguish schema versions does so by
  column-presence, not by string comparison:
  ```python
  if "game_won" in shard.schema.names:
      # game-outcome target is usable
  else:
      # only round-outcome target is usable
  ```
- The `_SCHEMA` constant in
  [bsw/to_parquet.py](../../src/tichu_training/bsw/to_parquet.py) is
  the source of truth for what columns the current parser emits. There
  is no separate schema-version constant.

## Rationale

1. **Featurizer and action-space versions track exactly what their
   names say.** Erosion of either term's meaning would compound across
   future schema changes — every column addition would lie about the
   featurizer.
2. **Arrow already enforces per-shard schema homogeneity.** Storing
   the version per row is redundant.
3. **The directory naming convention already works** and is the
   mechanism humans use day-to-day. Formalising it costs nothing.
4. **Column-presence detection is one line of code** and produces a
   clearer reader: "use game_won if available" rather than "if
   schema_version >= 3 then use game_won".
5. **Cheap to extend.** Future schema additions follow the same
   pattern: bump `v<N>`, add column, readers branch on presence.

## Consequences

- **Re-parses are required for schema-additive changes.** Backfilling
  a new column post-hoc requires either a one-shot migration script or
  a fresh parse. The `parse_bsw` step is fast enough relative to BC
  training that re-parse is the preferred mechanism (see Step 6 of the
  AWR game-outcome rollout).
- **Multiple schema versions can coexist on disk.** A workstation
  might have `parquet_100k_v2`, `parquet_100k_v3`, and `parquet_full_v3`
  simultaneously. Configs (`shards_dir`) name the version explicitly;
  there is no auto-discovery.
- **The reserved-but-empty columns from ADR-0011** (`state`,
  `legal_actions_mask`, `skill_decile`) remain in v3 unchanged.
  Dropping them is a separate decision and would itself bump the
  schema version when it happens.
- **No backwards-compatibility shim in readers.** A reader that
  predates a column simply ignores it (Arrow is column-projective).
  A reader that requires a newer column must check for it and fail
  cleanly if absent. There is no schema-version-to-feature-set
  lookup table.

## Rejected alternatives

- **Bump `FEATURIZER_VERSION` to `"v3"`.** Rejected: erodes the term's
  meaning, breaks the version-pin check on bit-identical-featurizer
  Checkpoints, and propagates the lie into every future schema change.
- **Add a per-row `parquet_schema_version` column.** Rejected:
  redundant with column-presence under Arrow's per-shard homogeneity
  guarantee; wastes storage; provides no information the reader can't
  derive from the shard's own schema.
- **Sidecar `manifest.json` per parquet directory.** Rejected: adds a
  second source of truth that can drift from the actual column set,
  and provides no information beyond what the directory name and
  `shard.schema.names` already give.
- **Semver in the directory name (e.g., `parquet_100k_v3.1.0`).**
  Rejected: parquet schemas don't have a meaningful "minor version"
  concept — every change is additive-or-not, and we either re-parse
  or we don't. Integer suffix is sufficient.
