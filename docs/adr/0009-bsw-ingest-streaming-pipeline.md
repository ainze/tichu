# ADR-0009: BSW ingest is a fused streaming pipeline

- **Status:** Accepted
- **Date:** 2026-05-25
- **Related:** [ADR-0008](0008-bsw-replay-validation.md)

## Context

The BSW Corpus is ~2.4M games. The original `parse_bsw` implementation
was list-shaped: read every `.tch` into a `games: list[ParsedGame]`,
hand the list to `validate_corpus` (which replays every Round), then
hand the *same* list to `write_parquet_shards` (which replays every
Round again to emit per-decision rows). Two problems at corpus scale:

1. **Memory.** Holding every `ParsedGame` plus every emitted `_Record`
   in Python lists is many GB before the parquet writer ever flushes.
2. **Double replay.** The expensive engine replay runs twice per Round
   — once for validation, once for record emission.

A third constraint surfaced: the corpus is now stored as
`archive.zst` + `.idx` (5 GB compressed, ~50 GB decompressed). We do
not want to materialise the decompressed `.tch` files on disk just
to feed them into a pipeline that can read them in memory.

ADR-0008 already states that replay-failed games are *excluded from
training*. The previous code did not enforce this — it wrote parquet
records for every parsed game and only logged failed IDs to
`known_bad_games.txt`. Downstream training code had no consumer of
that file, so failed games were silently included in training shards.

## Decision

**`parse_bsw` is a single fused streaming pass.**

One loop, one replay per Round:

```
for (game_id, tch_text) in iter_archive_or_dir():
    parsed_game = parse_tch(tch_text, game_id)
    for parsed_round in parsed_game.rounds:
        replay = replay_round(parsed_round)
        if replay.score_mismatch:
            mark game failed; write game_id to known_bad_games.txt; skip
        else:
            for decision in replay.decisions:
                parquet_writer[decision_type].buffer(record)
    flush_row_groups_if_batch_full()
```

The CLI grows `--archive PATH` mutually exclusive with `--input DIR`.
Both modes funnel into the same `Iterator[tuple[game_id, tch_text]]`.
The archive iterator reads `archive.zst` in offset order from the
`.idx` sidecar, decompressing each blob in memory (~20 KB per file).

Validation stats become a small running accumulator (counters +
failed-IDs list). The standalone `validate_corpus(games)` function is
absorbed into the streaming loop and no longer exists as a public
callable.

One `pyarrow.parquet.ParquetWriter` is opened per decision type at
the start of the run and closed in `try/finally`. Output naming
remains `{decision_type}_00000.parquet`; the suffix is reserved for a
future multi-shard rollover but no rollover is implemented now.

Progress is shown via `tqdm` over the known total file count, with
postfix `valid=X failed=Y rows=Z`. Both ingest modes share the
progress wrapping.

**Replay-failed rounds are excluded from the parquet shards** at
Round granularity. Records are emitted only from rounds whose
engine-computed Ergebnis matches BSW's. A game with at least one
failing round still appears in `known_bad_games.txt` for monitoring,
but its matching rounds still contribute training records. This
aligns with ADR-0008's Consequences section (which explicitly
discusses Round-level exclusion and rejects prefix-truncation). It
preserves partial training signal from games where most rounds match
— relevant on the current sample data where neither game *fully*
matches but ~60% of rounds do.

## Rationale

1. **Memory bounded by batch size, not corpus size.** At 2.4M games
   the previous list-shape would have needed multi-GB of headroom
   just for Python objects. Streaming bounds peak memory regardless
   of corpus size.
2. **One replay per Round, not two.** Validation and record emission
   are the same loop. Fixes a silent 2× cost.
3. **No 50 GB scratch directory.** Reading the archive in memory is
   strictly cheaper than decompressing it to disk.
4. **ADR-0008 alignment.** "Excluded from training" now means
   physically absent from the parquet shards, not "present but
   tagged". Removes the trap where a consumer that forgets to filter
   `known_bad_games.txt` trains on corrupted data.
5. **Single CLI, two ingest modes.** The dir-input path stays useful
   for `sample/` and tests; the archive path scales to the corpus.
   Both share the downstream pipeline.

## Consequences

- `validate_corpus` is no longer a callable public API. Anything that
  imported it must move to the streaming entry point or build its own
  small validator from `replay_round`.
- Parquet shards are now strict ADR-0008-valid subsets of the parsed
  corpus at Round granularity. Downstream readers can drop defensive
  filtering against `known_bad_games.txt`; that file remains as a
  game-level monitoring signal (a spike in its length signals a
  parser or engine regression).
- The single `_00000.parquet` per decision type is now ~5–20 GB for
  the `play` shard. `pyarrow` handles this fine, but multi-shard
  rollover is reserved for the day someone needs parallel reads or
  per-shard inspection.
- Crash mid-run loses the in-flight row-group but the closed
  `ParquetWriter` (via `finally`) keeps prior row-groups intact. No
  resume support; re-run from scratch.
- Parser-level parse failures (the `parse_tch` raise path) and
  replay-failed games both land in `known_bad_games.txt`, sharing one
  exclusion log.

## Rejected alternatives

- **Stream the archive but keep the list-shaped pipeline downstream.**
  Rejected — saves the 50 GB scratch directory but still OOMs on the
  in-memory `_Record` list at corpus scale. Fixes the wrong problem.
- **Producer/consumer with a worker pool.** Considered — parsing is
  CPU-bound and Python is single-threaded per process, so a pool
  would meaningfully speed up corpus-scale runs. Rejected for now
  because (a) the streaming single-process baseline must exist
  before we measure whether parallelism is worth the complexity, and
  (b) progress reporting + graceful interrupt are much harder across
  workers. Revisit if a corpus-scale run is too slow.
- **Multi-shard rollover from day one.** Rejected — picking the
  threshold ("every 100k games", "every 1 GB") without a measured
  motivation is making up a number. The naming convention reserves
  the suffix space so adding rollover later is non-breaking.
- **Auto-detect `--input` as either dir or archive.** Rejected — too
  much magic for a CLI that also serves as an ops surface. Explicit
  `--archive` vs `--input DIR` keeps call-sites self-documenting.
