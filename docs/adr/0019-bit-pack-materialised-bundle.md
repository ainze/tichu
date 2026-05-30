# ADR-0019: Bit-pack the materialised BC bundle (schema v2)

- **Status:** Accepted
- **Date:** 2026-05-30
- **Related:** [ADR-0014](0014-pre-featurise-bc-corpus.md), [ADR-0017](0017-featurizer-v4-compact-trick-top-combo.md), [ADR-0013](0013-parquet-schema-versioned-by-directory.md)

## Context

[ADR-0014](0014-pre-featurise-bc-corpus.md) adopted the per-decision-type
memmap bundle and deliberately wrote both big arrays in their widest
form: features as dense `float32 (N, D)` and the legal mask as **raw
`uint8 (N, K)`, one byte per action, not bit-packed**. The stated reason
(Decision §1, Rationale §4) was reader cheapness — a raw mask lets the
per-batch fancy-index produce a usable boolean array via a single
`astype(bool)` with no unpacking. ADR-0014 explicitly deferred the
alternative: *"A quantised layout (int8 features + bit-packed mask)
reduces this ~4× … but quantisation is out of scope for this ADR,"* and
flagged it as material for a follow-up ADR if disk became binding.

Disk became binding. The live `materialised_*_v4` bundle (≈53.6 M `play`
decisions on the featurizer-v4 layout of [ADR-0017](0017-featurizer-v4-compact-trick-top-combo.md),
`D=224`, `K_play=1809`) measures **~146 GB**, dominated by two files:

| File | Per `play` row | Size |
| --- | ---: | ---: |
| `play_legal_mask.dat` | 1809 B | **96.9 GB** |
| `play_features.dat` | 896 B | **48.0 GB** |
| meta + order | — | ~1.1 GB |

Two structural facts make both files almost entirely wasted bits:

1. **The legal mask is 0/1.** Storing one byte per action spends 8× the
   information it carries.

2. **The feature vector is mostly 0/1.** Sampling ~200k rows of the live
   bundle: 97.2% of all values are exactly 0.0 or 1.0, and **214 of the
   224 columns are *strictly* 0/1**. This is not incidental — it is a
   construction property of
   [`featurize()`](../../src/tichu_training/featurizer.py): every
   one-hot / multi-hot section (`own_hand`, `seen_cards`, `out_order`,
   the callers, `mahjong_wish`, `current_player`, `phase`,
   `trick_top_combo`, `trick_passes`) is written as exactly `1.0`. The
   only non-indicator columns are the 10 continuous ratios
   (`hand_sizes` ÷14, `team_scores` ÷1000, `round_points` ÷100), which
   can be negative and so must stay floating-point.

ADR-0014's "raw is cheaper for the reader" rationale was correct about
the reader but never weighed against the disk cost at corpus scale,
because at the time the bundle was assumed to stay at the ~17 GB pilot
size. At 146 GB the trade inverts.

## Decision

**Bump `MATERIALISED_SCHEMA_VERSION` 1 → 2 and bit-pack every 0/1 column
of the bundle. Keep the continuous feature columns bit-exact `float32`.
Do not sparse-encode and do not quantise the continuous columns.**

Concrete shape (shipped on this branch):

1. **Legal mask → `np.packbits(mask, axis=1)`**, `ceil(K/8)` bytes per
   row. `K_play=1809` → 227 B (was 1809). The reader `np.unpackbits`-es
   and slices back to `K` columns.

2. **Features split into two files per type.** The 214 indicator columns
   are packed to `<type>_feat_bits.dat` (`uint8`, 27 B/row); the 10
   continuous columns are kept as `<type>_feat_cont.dat` (`float32`,
   40 B/row). A `play` feature row goes 896 → 67 B. The reader rebuilds
   the dense `(B, D)` float32 vector by scattering each block back into
   its original columns over precomputed contiguous runs.

3. **The featurizer owns the split.**
   [`featurizer.CONTINUOUS_SECTIONS`](../../src/tichu_training/featurizer.py)
   → `CONTINUOUS_FEATURE_COLUMNS` is the single source of truth; the
   writer records the resulting column list in `manifest.json`
   (`continuous_feature_columns`) and the reader reconstructs from the
   manifest — it never imports the featurizer for layout, the same way
   it already rebuilds `meta_dtype` from the manifest.

4. **Manifest gains `legal_mask_packed: true`, `features_packed: true`,
   and `continuous_feature_columns`.** Their absence marks a pre-v2
   bundle; the `schema_version` pin
   ([ADR-0013](0013-parquet-schema-versioned-by-directory.md) semantics)
   rejects those loudly via `VersionMismatchError`.

5. **Writer guard against silent corruption.** `materialise()` raises if
   any column it is about to pack is not 0/1, naming the offending
   column. Packbits collapses any non-zero to 1, so without the guard a
   featurizer change that made an "indicator" section continuous — or
   feeding a non-featurizer stream — would silently wreck those bits.

6. **No sparse mask, no f16/int8.** The continuous columns stay f32 so
   model inputs are bit-exact; the mask stays a dense bitset.

## Rationale

1. **The win is large and the cost is paid once.** Same 53.6 M-row
   corpus, schema v2:

   | File | v4 (raw) | v5 (packed) |
   | --- | ---: | ---: |
   | features | 48.0 GB | **3.6 GB** (bits 1.4 + cont 2.1) |
   | legal_mask | 96.9 GB | **12.2 GB** |
   | meta + order | 1.1 GB | 1.1 GB |
   | **total** | **~146 GB** | **~17 GB** (~8.6×, ~129 GB saved) |

   Features compress 13.4× (896 → 67 B/row), the mask 8× (1809 →
   227 B/row). The whole bundle now fits the pilot footprint ADR-0014
   sized for 250k rows — at 53.6 M rows.

2. **ADR-0014's "raw mask is cheaper to read" no longer holds its
   weight.** Unpacking is a vectorised `np.unpackbits` + a handful of
   slice copies per batch — microseconds against the GPU step, and
   trivial next to the `replay_round` cost the bundle exists to avoid
   (ADR-0014 measured replay at ~75% of worker CPU, `featurize` at
   ~2.5%). The reader stays a single fancy-index per block; only a small
   reconstruction copy is added. This ADR revises ADR-0014 Decision §1
   and Rationale §4 on that one point; everything else in ADR-0014
   stands.

3. **Lossless beats quantised here.** ADR-0014 floated *int8* features.
   But the indicator columns are already 1-bit information, so packing
   them is exact and strictly better than int8 (8 cols/byte vs 1). The
   only genuinely floating columns are 10 of 224; storing them f16
   would save a further ~20 B/row (~1 GB total) at the cost of a
   precision question on model inputs — not worth it when the dominant
   file is now the mask. Keeping f32 means v5 features round-trip
   bit-exact (verified against the live bundle).

4. **Coupling the split to the featurizer is the right coupling.** The
   binary/continuous partition is a fact *about featurizer output*, and
   `FEATURIZER_VERSION` already invalidates the bundle on any layout
   change. Putting the source of truth in `featurizer.py` and recording
   the derived columns in the manifest keeps the storage layer
   self-describing while the version pin keeps it honest. The writer
   guard (Decision §5) is the backstop for a featurizer change that
   alters value semantics without updating `CONTINUOUS_SECTIONS`.

5. **Chunked-write byte-identity is preserved.** Both packbits calls and
   the column split act per row (`axis=1`), so a chunked materialise is
   still byte-for-byte identical to a single-shot one — the property the
   consolidated single-pass writer of
   [ADR-0016](0016-consolidate-parse-and-materialise-into-one-replay.md)
   relies on, and which its byte-identity test still asserts.

## Consequences

- **All pre-v2 bundles are invalid and must be re-materialised.** The
  `schema_version` pin rejects them. This is the documented cost of a
  layout change (ADR-0014 Consequences); the migration here is free in
  practice because the full corpus was being re-materialised for v5
  regardless. No in-place conversion path is provided.

- **The legal mask is now the dominant file (~12 GB of ~17 GB).** That
  is the floor for a dense bitset. Sparse-encoding the mask (most states
  have few legal actions) was considered and **explicitly rejected** for
  now: it breaks the fixed-stride memmap that makes `iter_batches`
  cheap, for a one-file win on an already-17 GB bundle. A future ADR can
  revisit if the mask becomes binding.

- **Two feature files per type instead of one.** `<type>_features.dat`
  is replaced by `<type>_feat_bits.dat` + `<type>_feat_cont.dat`. Any
  out-of-tree reader of the raw files must split accordingly; the
  in-tree readers (`MemmapBCDataset`, the bench harness) and writers
  (`materialise()`, the legacy `materialise_bc_subset.py`) are updated.

- **The reader adds a small per-batch reconstruction.** An `np.empty
  (B, D)` allocation plus unpackbits and a few slice copies. Bounded and
  vectorised; the per-row `__iter__` path pays the same as a per-row
  copy it already made.

- **`SyntheticBCDataset` gained an opt-in `binary_features=True` mode.**
  Its default all-Gaussian features violate the 0/1 contract (and now
  correctly trip the writer guard), so tests that materialise synthetic
  data opt into a featurizer-consistent stream. The default is unchanged,
  so the AWR / value-baseline tests that depend on continuous synthetic
  features are unaffected.
