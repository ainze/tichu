# ADR-0016: Consolidate parse_bsw + materialise_bc into one engine replay pass

- **Status:** Accepted
- **Date:** 2026-05-29
- **Related:** [ADR-0014](0014-pre-featurise-bc-corpus.md), [ADR-0011](0011-bc-training-replay-on-the-fly.md), [ADR-0009](0009-bsw-ingest-streaming-pipeline.md), [ADR-0008](0008-bsw-replay-validation.md), [ADR-0013](0013-parquet-schema-versioned-by-directory.md)

## Context

[ADR-0014](0014-pre-featurise-bc-corpus.md) shipped option β' — the
pre-featurised per-type memmap bundle — as the production training
source. Producing a trainable bundle from the raw BSW archive now takes
**two** full passes through the engine:

1. **`parse_bsw`** ([cli/parse_bsw.py](../../src/tichu_training/cli/parse_bsw.py))
   replays every round to validate it against BSW's recorded `Ergebnis`
   (ADR-0008) and emits the per-decision Parquet manifest
   ([bsw/to_parquet.py](../../src/tichu_training/bsw/to_parquet.py)).

2. **`materialise_bc`** ([cli/materialise_bc.py](../../src/tichu_training/cli/materialise_bc.py))
   drives `ParallelParquetBCDataset` — which replays the *same rounds
   again* — to build `PrivateState`s, featurise at each decision
   boundary, and write the β' bundle.

The 2026-05-29 pipeline-perf profile
([docs/notes/2026-05-29-pipeline-perf-fix.md](../notes/2026-05-29-pipeline-perf-fix.md))
that motivated ADR-0014 also measured this duplication. Two flamegraphs
(the `parse_bsw` worker and the BC materialise worker) show the **same
dominator**: `replay_round` → `engine.step` → `legal_actions` consumes
~75% of worker CPU in *both* passes. `featurize` is <3%. The expensive
thing — deriving the sequence of engine states at decision boundaries —
is computed identically in each pass and thrown away in between.

Measured cost on the 100k-game subset:

| Stage | Wall-clock |
| --- | ---: |
| `parse_bsw` (~12 games/s across 10 workers) | ~2.3 h |
| `materialise_bc` | ~1–3 h |
| **Total ingest → trainable bundle** | **~3–5 h** |

Both passes need exactly the same `PrivateState`s at the same decision
boundaries. The second replay derives nothing the first one couldn't
have emitted on the spot.

## Decision

**Fold the bundle write into `parse_bsw`'s existing replay pass, behind
an opt-in `--bundle-out-dir` flag.** When the flag is set, the round
that is already being replayed for Parquet validation *also* emits the
featurised `BCExample` stream, and that stream is materialised to a
`MemmapBCDataset`-readable bundle. One replay, both outputs.

This resolves the four open questions from the consolidation handoff:

1. **Where the consolidated pass lives → extend `parse_bsw` (option a).**
   A new `--bundle-out-dir` flag on the existing CLI, rather than a new
   `ingest_bsw` CLI (option b) or an archive-reading `materialise_bc`
   (option c). Smallest blast radius: `parse_bsw` stays the single
   canonical entry point for both manifest-only and bundle-producing
   runs, and adoption is opt-in. A run without the flag behaves exactly
   as before.

2. **Per-worker vs centralised bundle write → centralised, streaming.**
   Worker processes (`_process_game` / `_parse_and_process`) attach the
   round's `BCExample`s to the `_GameResult` they already return; the
   dispatcher exposes the results as a generator and hands that
   generator straight to `materialise()`, which pulls examples
   incrementally and flushes every `chunk_size`. Peak RAM is therefore
   bounded by the chunk size, **not** by the corpus size — the
   dispatcher never holds more than one game's results plus one chunk.
   Per-worker *sharded* bundles with an end-of-run merge remain
   **deferred** — but as a throughput/IPC optimisation, not a
   correctness or memory requirement (see Consequences).

3. **Parquet `featurizer_version` column → kept as a redundant check.**
   The bundle's `manifest.json` is the authoritative featurizer pin for
   the bundle (ADR-0014's triple pin). The per-row Parquet stamp stays
   as a cheap defensive cross-check on the manifest; it is no longer the
   sole source of truth but the cost of keeping it is negligible.

4. **Parquet manifest stays.** After consolidation the bundle is the
   training source, but the Parquet manifest remains the
   "which `(game_id, round_id)` are valid" filter. Re-materialising
   under a new `FEATURIZER_VERSION` against the same archive still needs
   it to skip the rounds that failed replay, so it is not dead weight.

Concrete shape, shipped on this branch:

- **`stream_to_parquet` / `stream_raw_to_parquet` gain
  `bundle_out_dir: Path | None`.** When set, the worker is run with
  `emit_bundle=True` and a bundle is written to that directory.
  ([bsw/to_parquet.py](../../src/tichu_training/bsw/to_parquet.py))

- **`_emit_bc_examples_for_round`** mirrors `ParquetBCDataset.__iter__`
  exactly — same decision-type filter, same featurise / target /
  legal-mask logic, same skip conditions — so the consolidated bundle
  is identical to the one the legacy path would have produced.

- **`parse_bsw --bundle-out-dir DIR`** wires the flag through to
  `stream_raw_to_parquet`. ([cli/parse_bsw.py](../../src/tichu_training/cli/parse_bsw.py))

- **The bundle write reuses ADR-0014's `materialise()` writer
  unchanged** — same per-type layout, same triple version pin, same
  `MemmapBCDataset` reader. Consolidation changes *where* the
  `BCExample` stream comes from, not the on-disk format. The dispatcher
  feeds `materialise()` a generator of results, so the writer's existing
  chunked-flush behaviour bounds RAM for free.

`materialise_bc` is **not** removed. It remains the path for
re-materialising an existing Parquet manifest under a new featurizer
version without re-running BSW validation.

## Rationale

1. **The eliminated work is the measured dominator.** The second
   replay is ~75% of `materialise_bc`'s worker CPU, and that pass exists
   only to re-derive states `parse_bsw` already had in hand. Removing it
   projects the 100k-subset ingest from ~3–5 h to **~2–3 h** — roughly
   one whole replay pass saved, the single largest lever in the
   pipeline-perf wrap-up.

2. **Byte-identical parity is the safety net that lets both paths
   co-exist.** A round of TDD pinned the invariant first: the bundle
   from `parse_bsw --bundle-out-dir` is **byte-for-byte identical**, per
   decision type, to the bundle from
   `stream_to_parquet → ParquetBCDataset → materialise()`
   ([test_to_bundle.py](../../tests/training/bsw/test_to_bundle.py)).
   This holds because `_emit_bc_examples_for_round` is a deliberate
   mirror of the canonical emit loop, fed in the same game/round/decision
   order. The parity test doubles as a drift alarm: if the two emit
   paths ever diverge, it fails.

3. **Opt-in keeps the blast radius minimal.** Manifest-only runs (the
   common case for engine-fix iteration, where only the validation
   stats and `failure_details.tsv` matter) pay nothing — `emit_bundle`
   is `False`, no featurise, no bundle. The consolidation is a pure
   addition to the existing CLI contract.

4. **Reusing `materialise()` keeps one on-disk format.** The bundle the
   consolidated path writes is the same β' bundle ADR-0014 specified,
   read by the same `MemmapBCDataset` with the same triple version pin.
   No new format, no new reader, no second versioning surface.

5. **Same Round-granularity filter, enforced in one place.** A round
   that fails replay validation (ADR-0008 / ADR-0009) contributes
   *zero* rows to the bundle, because bundle emission happens only on
   the matched branch of the same `_classify_round_failure` check that
   gates Parquet emission — pinned by
   `test_replay_failed_rounds_contribute_no_bundle_rows`.

## Consequences

- **`materialise_bc` and the legacy two-pass path are unchanged.**
  Re-materialising an existing manifest under a new featurizer version
  still uses `materialise_bc`; this ADR adds a faster path for the
  *fresh ingest* case, it doesn't remove the manifest-driven one.

- **Peak RAM is bounded by the chunk size, not the corpus.** The
  dispatcher hands `materialise()` a generator of results and lets the
  writer pull examples and flush every `chunk_size` (ADR-0014, commit
  `aeb0088`), so a full-corpus run holds at most one game's results plus
  one chunk in memory. Pinned by
  `test_bundle_is_written_by_streaming_not_full_buffering`, which asserts
  the writer starts pulling before the game source is drained. This was
  a real gap in the first cut (a `list`-buffering implementation) and is
  the reason the streaming behaviour is now an explicit regression test.

- **Per-worker sharded bundles remain a deferred throughput
  optimisation.** With centralised streaming, the bundle write itself is
  serial in the dispatcher; at high worker counts the per-game
  `BCExample` pickle + queue transfer is the same ~10% IPC tax the BC
  materialise pass already pays. Per-worker shards (each worker writes a
  self-contained bundle, a final step concatenates per-type `.dat` files
  and rebuilds `order.dat` + `manifest.json`) would remove that tax.
  This is now a *speed* lever, no longer required for memory or
  correctness.

- **An empty bundle is unreadable.** If every round fails validation
  (or a future per-worker shard sees only bad games), `materialise()`
  writes zero-length `.dat` files and `MemmapBCDataset` raises on
  `mmap` of an empty file. This is pre-existing ADR-0014 reader
  behaviour, not introduced here, but a future per-worker merge must
  handle empty shards explicitly.

- **The Parquet `featurizer_version` column is now belt-and-braces.**
  The bundle manifest is authoritative; the Parquet stamp is a
  redundant defensive check. A future ADR may drop the column once the
  manifest pin is trusted everywhere, but there is no reason to churn
  the Parquet schema for it now.

- **The emit loop is now triplicated.** `_emit_bc_examples_for_round`
  joins `ParquetBCDataset.__iter__` and
  `parallel_dataset._worker_loop` as a third near-copy of the
  per-decision emit. They differ only in skill-lookup `None` handling
  (the Parquet ratings column is nullable; the dataset loader's is not).
  Extracting a single shared `emit_bc_examples_for_round` is the obvious
  deepening; deferred to a focused refactor pass because it touches two
  other hot-path modules with their own tests. The byte-parity test
  guards against drift in the meantime.

## Rejected alternatives

- **New `ingest_bsw` CLI superseding both (handoff option b).**
  Rejected for migration churn. A cleaner mental model, but it would
  obsolete two documented CLIs and every script/runbook that calls
  them, for no functional gain over an opt-in flag. Revisit only if the
  `parse_bsw` argument surface becomes unwieldy.

- **`materialise_bc` reads the archive directly, `parse_bsw` stays
  manifest-only (handoff option c).** Rejected. It splits the pipeline
  differently without removing a replay — `materialise_bc` would still
  replay, and `parse_bsw` would still replay; the duplication survives.
  It solves a different problem (skipping Parquet) than the one measured.

- **Buffering the full `BCExample` stream into a list, then calling
  `materialise()` once.** This was the first cut, and it was wrong: it
  defeats `materialise()`'s chunked design and walls RAM at corpus
  scale. Replaced by the generator-driven streaming above, which
  reshapes `_run_stream`'s dispatch into a pull-based result generator —
  a small change, and independent of the per-worker-shard decision it
  was mistakenly thought to depend on.
