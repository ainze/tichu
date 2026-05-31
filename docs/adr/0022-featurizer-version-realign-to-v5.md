# ADR-0022: Realign FEATURIZER_VERSION to v5 (no feature-content change)

- **Status:** Accepted
- **Date:** 2026-05-31
- **Related:** [ADR-0017](0017-featurizer-v4-compact-trick-top-combo.md) (the v4 featurizer layout, unchanged here), [ADR-0019](0019-bit-pack-materialised-bundle.md) (the packed bundle, schema v2), [ADR-0013](0013-parquet-schema-versioned-by-directory.md) (version-axis separation), [ADR-0004](0004-payload-agnostic-checkpoint.md) (Checkpoint version pins)

## Context

Three version axes coexist in the pipeline, deliberately kept separate by
[ADR-0013](0013-parquet-schema-versioned-by-directory.md):

1. **`FEATURIZER_VERSION`** — the *content* of `featurize()`'s output. Last
   changed to `"v4"` by [ADR-0017](0017-featurizer-v4-compact-trick-top-combo.md)
   (the 224-dim compact-trick-top-combo layout). Pinned on every **Checkpoint**
   ([checkpoint.py](../../src/tichu_training/checkpoint.py) → hard
   `VersionMismatchError` on load) and on every materialised bundle.
2. **`MATERIALISED_SCHEMA_VERSION`** — the bundle's on-disk layout. Bumped
   1→2 by [ADR-0019](0019-bit-pack-materialised-bundle.md) for bit-packing.
   `featurize()` output did **not** change, so ADR-0019 explicitly left
   `FEATURIZER_VERSION` at `"v4"`.
3. **The bundle directory-generation label** — an operator naming convention
   (`materialised_<scale>_v<N>`). ADR-0019 informally called the packed
   generation "v5" while the raw one was "v4", to distinguish two on-disk
   bundles that share featurizer content but differ in layout.

That informal "v5" collided with the documented "directory suffix == featurizer
version" convention: the packed bundle is featurizer-v4, so by that rule it
would be `materialised_100k_v4` — indistinguishable from the existing *raw*
v4 bundle. The mismatch between the stamp (`"v4"`) and the generation operators
were calling "v5" was a recurring source of confusion at materialise time.

The "One parse pass, many task bundles" work ([ADR-0020](0020-one-parse-pass-many-task-bundles.md))
re-materialises the entire 100k corpus into packed per-task bundles regardless.
That is the natural moment to retire the v4 generation wholesale.

## Decision

**Set `FEATURIZER_VERSION = "v5"`. The feature *content* is unchanged — v5
output is byte-identical to v4.** The bump is a version-stamp realignment, not
a featurizer change: it makes the stamp match the directory generation operators
already use for the packed bundles, and forces a clean re-materialise + retrain
that retires the v4-stamped artifacts in one move.

Concretely:

- [`featurizer.py`](../../src/tichu_training/featurizer.py): the constant flips
  to `"v5"`; an inline comment states v5 content == v4 so no reader hunts for a
  featurizer change. `SECTION_DIMS`, `TRICK_TOP_COMBO_SUBFIELDS`,
  `CONTINUOUS_SECTIONS`, and `FEATURIZER_OUTPUT_DIM` (224) are untouched.
- All new bundles and Checkpoints stamp `"v5"`; on-disk dirs are
  `materialised_100k_v5` / `parquet_100k_feat_v5`.

## Rationale

1. **One version number per generation beats three for the operator.** The
   stamp, the bundle generation, and the directory suffix now agree (`v5`),
   eliminating the "is this v4 or v5?" question the previous split created.
2. **The cost is paid once, at a re-materialise we were doing anyway.** ADR-0020
   re-materialises the whole corpus; retiring v4 here is free in wall-clock.
3. **A clean break is simpler than a permanent split.** The alternative —
   keeping `FEATURIZER_VERSION="v4"` forever and bolting a separate
   bundle-generation constant onto the naming — preserves more history but
   leaves two numbers to reconcile indefinitely. Given v4 artifacts are being
   retired, a single forward number is the lower-maintenance contract.

## Consequences

- **Every v4-stamped Checkpoint is invalidated.** `bc_full_100k_v4_memmap`,
  `awr_full_100k_v4_memmap_{round,game}`, and any v4 Call / Schupfen
  Checkpoints raise `VersionMismatchError` against v5 code in BC training, AWR
  refine, inference (`ml_agent`), and export (`torchscript`). **They must be
  retrained** from the v5 bundles. This is accepted: the bump's explicit
  purpose is to retire the v4 generation. Because v5 features are byte-identical
  to v4, retrained checkpoints are of equivalent quality — only the stamp moves.
- **v5 is a versioning event with no featurizer diff.** This is the surprising
  part a future reader hits ("what changed in featurizer v5?" — nothing in
  content). The inline comment in `featurizer.py` and this ADR are the record.
  Future genuine featurizer changes resume at v6.
- **The three version axes still exist and remain conceptually distinct**
  (ADR-0013). This ADR realigns their *values* for one generation; it does not
  merge the axes. A future bundle-layout change still bumps
  `MATERIALISED_SCHEMA_VERSION` independently.
- **All pre-existing v4 bundles are stale** and superseded by the v5 packed
  bundles; the old `materialised_100k_v4` may be deleted once v5 is validated.

## Rejected alternatives

- **Keep `FEATURIZER_VERSION="v4"`, name the dir as a pure generation label.**
  No code change, no checkpoint breakage, ADR semantics fully intact — but
  leaves the stamp (`v4`) and the operator-facing generation (`v5`)
  permanently out of step, which is the confusion this ADR removes. Reasonable;
  rejected in favour of a clean single number now that v4 is being retired.
- **Add a dedicated `BUNDLE_GENERATION` constant** recorded in manifests and
  used for directory names, leaving `FEATURIZER_VERSION="v4"`. Cleanest in
  theory (each axis explicit, no false featurizer bump, checkpoints preserved),
  but adds a fourth version surface and still doesn't retire the v4 checkpoints
  the team wanted gone. Rejected as over-engineering for a one-time retirement.
