# ADR-0011: BC training is a replay-on-the-fly archive-driven pipeline

- **Status:** Accepted
- **Date:** 2026-05-26
- **Related:** [ADR-0008](0008-bsw-replay-validation.md), [ADR-0009](0009-bsw-ingest-streaming-pipeline.md)

## Context

After ADR-0009 the ingest pipeline produces per-decision Parquet shards keyed
by `(decision_type, game_id, round_id, player_handle, action_taken)`. The
schema also reserves three columns intended to carry training-time payload:
`state` (serialised PrivateState), `legal_actions_mask` (packed 1809-bit
mask), and `skill_decile` (joined from the TrueSkill ratings table).

Today those three columns are empty: `state` and `legal_actions_mask` are
deliberately reserved-but-empty per [to_parquet.py](../../src/tichu_training/bsw/to_parquet.py)
("filled at training-loop load time, not at parse time"), and `skill_decile`
is empty until `parse_bsw` is re-run with `--trueskill RATINGS.parquet`.
The training-time consumer [ParquetBCDataset](../../src/tichu_training/bc/dataset.py)
is correspondingly a stub — it validates the version pins and counts rows
but does not yield `BCExample`s, and [train_bc](../../src/tichu_training/cli/train_bc.py)
hard-raises `NotImplementedError` when handed `dataset: parquet`.

The featurization-cache question is the gating decision. Three candidates
were considered:

- **α. Replay-on-the-fly.** Per epoch, look up each game in the archive,
  replay the round, featurize at the decision boundary, yield a
  `BCExample`. Nothing new on disk.
- **β. Cache serialised PrivateState in parquet.** Backfill the reserved
  `state` / `legal_actions_mask` columns at parse time; featurize from
  them at train time. **~5–20 GB at 100k, ~120–500 GB at corpus scale.**
- **γ. Cache dense 16,568-float Feature Vectors.** Pre-featurize the
  corpus once into a numpy memmap. **~530 GB at 100k, ~13 TB at corpus
  scale.** Disqualifying at corpus scale.

Within α, two loop shapes:

- **α.1 — Parquet-driven.** Iterate parquet rows; per row, look up
  `game_id` in the archive (cheap, ~20 KB seek per game per the
  `.idx` sidecar in [archive.py](../../src/tichu_training/bsw/archive.py)).
  Replays each round ~80× per epoch (once per decision in the round).
- **α.2 — Archive-driven.** Iterate the archive in offset order (the
  same pattern ADR-0009 already uses for parse). For each game, look up
  the parquet rows for this `(game_id, round_id)`; replay each round
  once, featurize every decision the parquet expects, yield all of
  them. Replays each round once per epoch.

## Decision

**BC training is α.2: archive-driven, parquet-as-manifest, replay-on-the-fly.**

Concrete shape:

1. `ParquetBCDataset` becomes a `torch.utils.data.IterableDataset`. It
   reads the parquet shards once to build a `(game_id → set[round_id])`
   manifest plus a `(game_id, round_id, player, decision_type) →
   (action_taken, sample_weight, round_outcome)` lookup. Skill decile is
   joined live against `RATINGS.parquet` by `player_handle`; the
   parquet's `skill_decile` column is **unused at training time** and
   slated for schema removal in a follow-up.

2. Per epoch, the dataset iterates the archive in offset order. For each
   game whose `game_id` appears in the manifest: parse, then for each
   round whose `(game_id, round_id)` is in the manifest, replay it via
   the existing `replay_round`. At each decision boundary, construct
   the `PrivateState` for the acting player, featurize it, derive the
   target Intent index and legal Intent mask, and yield a `BCExample`.

3. Concrete → Intent infrastructure lives in
   `tichu_training/action_space.py` alongside `encode` / `decode`. Two
   new functions: `intent_for_concrete(parsed_action, game_state) → int`
   (target label for the `play` head) and `legal_mask(decision_type,
   game_state, player) → np.ndarray[K, bool]` where K is
   `HEAD_LOGIT_DIMS[decision_type]` (1809 / 14 / 2 for play / wish /
   dragon_assignment). The forward Resolver stays in
   `tichu_inference/ml_agent.py` per the existing layering — inference
   needs it; training does not.

   **BC consumes three of six parquet decision types.** Per ADR-0007,
   `call_tichu` and `call_grand_tichu` are served by standalone Call
   Networks, not BC heads. Per [ADR-0012](0012-schupfen-is-a-standalone-network.md),
   `schupfen` is served by a standalone Schupfen Network, also not a
   BC head. The IterableDataset filters parquet rows on
   `decision_type ∈ HEAD_LOGIT_DIMS.keys()` = `{"play", "wish",
   "dragon_assignment"}`; the other shards remain in parquet for
   `train_calls` and `train_schupfen` to consume independently.

4. Worker model mirrors `parse_bsw`: `ProcessPoolExecutor`, each worker
   owns a zstd decompressor and a slice of games (round-robin by
   `game_id` hash). Yielded `BCExample`s flow through a bounded
   multiprocessing queue into a per-trainer shuffle buffer.

5. Trainer signature changes from `Sequence[BCExample]` to
   `Iterable[BCExample]`. `train_one_epoch` consumes the iterable
   incrementally with batching driven by a fill-then-yield batch builder
   keyed by `decision_type`. Resume / checkpointing is unchanged — step
   counts live in `step.csv`, not in dataset position.

## Rationale

1. **Replay cost is amortised per round, not per decision.** A single
   `replay_round` call already produces every decision of that round
   (see [to_parquet.py:300-303](../../src/tichu_training/bsw/to_parquet.py)).
   α.1 throws that work away and re-replays per decision; α.2 keeps it.
   With ~80 decisions/game in the corpus this is an order-of-magnitude
   factor.
2. **Sequential archive read beats per-row random seek.** zstd shared-
   dictionary decode amortises better on sequential offsets. Even on
   NVMe, sustained sequential decompression is the cheapest read path
   the archive offers.
3. **No new on-disk format, no new schema version.** α.2 doesn't write
   anything new to disk. β fills the reserved columns but commits the
   parquet to a serialised-PrivateState format that would then be
   coupled to engine internals (any engine refactor invalidates the
   cache). γ commits parquet to a FEATURIZER_VERSION-specific dense
   format. α.2 keeps versioning surface minimal — only the existing
   `featurizer_version` / `action_space_version` pins on the Checkpoint
   matter.
4. **Re-uses the ADR-0009 streaming architecture verbatim.** The only
   change to the streaming loop is the sink: `_Record → ParquetWriter`
   becomes `BCExample → queue`. Parser, replay, validation, worker
   pool, progress reporting all stay.
5. **Tichu rounds are short.** A Round is bounded by 14 cards × 4
   players and ≤ 8 plays per Trick, so a full round-replay is
   microseconds-per-decision in absolute terms. Per-epoch corpus-scale
   replay is plausibly faster than the featurizer it feeds.
6. **β remains additive.** If BC training needs many epochs and
   wall-clock dominates, backfilling the reserved `state` /
   `legal_actions_mask` columns in a one-shot pass is a non-breaking
   change — the trainer becomes "use cached state if present, else
   replay". α.2 is the minimum-commitment path that preserves that
   option.

## Consequences

- **Parquet is a manifest, not a dataset.** The shards carry the
  validated `(game_id, round_id)` set plus label/metadata columns
  (`action_taken`, `round_outcome`, `round_won`, `player_handle`,
  `sample_weight`). They do **not** carry training inputs. A future
  reader expecting "parquet is the BC training data" is wrong; the
  archive is.

- **`skill_decile` and `state` / `legal_actions_mask` columns are dead
  weight.** They survive in the v1 schema for backwards compatibility
  with already-emitted shards, but no training-time reader consumes
  them. A follow-up may drop them from the schema. The reserved-but-
  empty intent ([to_parquet.py:73-75](../../src/tichu_training/bsw/to_parquet.py))
  is superseded by this ADR.

- **`parse_bsw --trueskill` is no longer the canonical way to attach
  skill deciles.** The trainer reads `RATINGS.parquet` directly and
  joins live. The `--trueskill` flag remains useful for spot-checking
  but is not required for BC training.

- **Trainer signature change is non-trivial.**
  `train_one_epoch(model, examples: Sequence[BCExample], ...)` becomes
  `train_one_epoch(model, examples: Iterable[BCExample], ...)`, and
  `train_bc._build_dataset` returns an iterable, not a list. The
  feature-dim inference at `dataset_examples[0].features.shape[0]` is
  replaced by a constant from
  `tichu_training.featurizer.FEATURIZER_OUTPUT_DIM`. AWR refinement
  paths (`_run_awr_refinement`) that materialise the whole dataset for
  the value-baseline fit must explicitly drain the iterable into a list
  for that step — value-baseline fit is one-shot, not per-epoch, so the
  RAM cost is bounded.

- **Shuffle quality is bounded by buffer size.** With a 16k-example
  shuffle buffer over a sequential archive read, examples within ~16k
  positions of each other can co-occur in a batch. Across-epoch SGD
  shuffles well; within-epoch correlation is bounded but non-zero. If
  this proves a training-quality issue, the mitigation is a larger
  buffer or a multi-pass shuffle, not a pipeline redesign.

- **Action Space module grows engine coupling.**
  `tichu_training/action_space.py` gains imports from
  `tichu_engine.legality` and `tichu_engine.state`. This is consistent
  with the layering in CONTEXT.md (training-layer knows engine; engine
  does not know training-layer) and does not move the engine boundary.

- **Crash mid-epoch loses in-flight examples, not on-disk state.** No
  resume support inside an epoch — re-run the epoch from the start.
  Step-level checkpoints remain epoch-boundary granularity, same as
  today.

## Rejected alternatives

- **β. Cache serialised PrivateState in parquet.** Rejected for now:
  commits parquet to an engine-internal serialisation format, requires
  a versioning policy for that format, makes re-runs of `parse_bsw` an
  order of magnitude more expensive in time and disk. Remains additive
  later if wall-clock motivates it.
- **γ. Cache dense Feature Vectors.** Rejected: ~13 TB at corpus scale
  is impractical, and the cache is invalidated by any featurizer fix.
- **α.1. Parquet-driven loop with per-row archive seek.** Rejected:
  replays each round ~80× per epoch with no offsetting benefit; the
  random-access archive read is also slower per byte than the
  sequential read α.2 enables.
- **Engine-side `legal_intents` function.** Rejected: violates the
  CONTEXT.md layering (engine talks Combinations, not Intents). The
  same information is constructable from `legality.enumerate_legal` +
  `intent_for_concrete` in the training layer without breaking the
  boundary.
