"""Disk-backed BC dataset: writer + reader for the materialised corpus.

The replay-on-the-fly path of ADR-0011 is CPU-bound on the BSW engine
(`replay_round` / `engine.step` / `legal_actions` — ~75% of worker CPU
in the 2026-05-29 profile). Materialising the BCExample stream once,
to per-decision-type numpy memmaps, bypasses replay entirely. See
[ADR-0014](../../../docs/adr/0014-pre-featurise-bc-corpus.md) for the
trade analysis vs ADR-0011's α.2 (replay) and rejected β (parquet-
cached state) / γ (dense feature memmap, no per-type split), and
[ADR-0019](../../../docs/adr/0019-bit-pack-materialised-bundle.md) for the
schema-v2 bit-packing of the mask + indicator feature columns below.

On-disk layout for a slice of N examples:

  <out_dir>/
    manifest.json
    play_feat_bits.dat              (N_play, 27)      uint8  (packbits of 214 flags)
    play_feat_cont.dat              (N_play, 10)      float32 (continuous columns)
    play_legal_mask.dat             (N_play, 227)     uint8  (packbits of 1809 bits)
    play_meta.dat                   (N_play,)         structured (META_DTYPE)
    wish_feat_bits.dat              (N_wish, 27)      uint8
    wish_feat_cont.dat              (N_wish, 10)      float32
    wish_legal_mask.dat             (N_wish, 2)       uint8  (packbits of 14 bits)
    wish_meta.dat                   (N_wish,)         structured
    dragon_assignment_feat_bits.dat (N_da, 27)        uint8
    dragon_assignment_feat_cont.dat (N_da, 10)        float32
    dragon_assignment_legal_mask.dat (N_da, 1)        uint8  (packbits of 2 bits)
    dragon_assignment_meta.dat      (N_da,)           structured
    order.dat                       (N_total, 2)      uint32: (type_idx, row_idx_in_type)

Schema v2 bit-packs every 0/1 column to save disk:

  * Legal masks are 0/1 boolean. Storing one byte per action wastes 8×;
    they are bit-packed with `np.packbits(mask, axis=1)` to `ceil(K/8)`
    bytes per row (K = the head's logit width): a 1809-wide play mask
    becomes 227 bytes.

  * Feature vectors are 214 indicator columns (set to exactly 1.0) plus
    10 continuous ratio columns (`hand_sizes`/`team_scores`/`round_points`,
    can be negative). The bundle splits them: indicator columns are packed
    to `<type>_feat_bits.dat` (27 bytes/row) and the continuous columns are
    kept bit-exact float32 in `<type>_feat_cont.dat` (40 bytes/row). The
    column split comes from `featurizer.CONTINUOUS_FEATURE_COLUMNS` and is
    recorded in the manifest. Net play feature row: 896 → 67 bytes (13.4×).

The reader unpacks both on read and reconstructs the dense float32 vector
/ K-wide bool mask. The manifest flags (`legal_mask_packed`,
`features_packed`) gate this; their absence marks a pre-v2 bundle, which
the schema pin rejects anyway.

The reader preserves the original emission order via `order.dat`. The
fast batched path ignores it and walks each type's contiguous memmap
range directly (much cheaper memmap access; chosen for training where
shuffle is handled at a higher layer anyway).

Version pins: `materialise()` records FEATURIZER_VERSION,
ACTION_SPACE_VERSION, and MATERIALISED_SCHEMA_VERSION into the
manifest. `MemmapBCDataset.__init__` refuses to load on mismatch with
the live code, raising `VersionMismatchError`.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.dataset import BCExample
from tichu_training.bc.decision_types import HEAD_LOGIT_DIMS
from tichu_training.checkpoint import VersionMismatchError
from tichu_training.featurizer import (
    CONTINUOUS_FEATURE_COLUMNS,
    FEATURIZER_OUTPUT_DIM,
    FEATURIZER_VERSION,
)


log = logging.getLogger("tichu_training.bc.materialised")


# Bumped whenever the on-disk layout, meta dtype, or manifest schema
# changes in a non-backwards-compatible way. Independent of
# FEATURIZER_VERSION (which invalidates the feature *content* not the
# layout) and ACTION_SPACE_VERSION (which invalidates target / mask
# semantics not layout).
#   v1 -> v2: legal masks AND feature indicator columns bit-packed via
#             np.packbits (manifest gains "legal_mask_packed" +
#             "features_packed" + "continuous_feature_columns"). v1 bundles
#             stored one uint8 per action and f32 per feature column; the
#             pin rejects them so they're re-materialised.
MATERIALISED_SCHEMA_VERSION = 2


# Stable type-index ordering encoded in order.dat. Baked into the
# manifest so the reader doesn't depend on import-time iteration of
# HEAD_LOGIT_DIMS to recover the mapping.
TYPE_ORDER: list[str] = ["play", "wish", "dragon_assignment"]


# Per-row metadata. uint16 target is enough for the 1809-wide play
# head; uint8 skill_decile (0-10); int8 game_won as a tri-valued sentinel
# (-1=None, 0=False, 1=True). round_outcome and sample_weight are
# float32. Compact: 13 bytes/row, negligible next to features.
META_DTYPE: np.dtype = np.dtype([
    ("target", "u2"),
    ("sample_weight", "f4"),
    ("skill_decile", "u1"),
    ("round_outcome", "f4"),
    ("game_won", "i1"),
])


_GAME_WON_NONE_CODE: int = -1


def _packed_mask_bytes(mask_dim: int) -> int:
    """Bytes per row after `np.packbits` of a `mask_dim`-bit boolean row."""
    return (mask_dim + 7) // 8


def _binary_feature_columns(
    feature_dim: int, continuous_columns: Iterable[int],
) -> list[int]:
    """Ascending indicator-column indices = all columns minus the continuous
    ones. The writer packs these (in this order) and the reader unpacks back
    into the same slots, so both sides must derive them identically."""
    cont = set(continuous_columns)
    return [c for c in range(feature_dim) if c not in cont]


def _contiguous_runs(cols: list[int]) -> list[tuple[int, int, int]]:
    """Group an ascending column list into `(dst_start, dst_stop, src_start)`
    runs so a scatter into the dense vector becomes a handful of slice copies
    instead of one fancy-index assignment. `src_start` is the offset of the
    run's first column within the packed/continuous source block.

    For v4's layout this collapses the 214 indicator columns to two runs
    (0:56, 66:224) and the 10 continuous columns to one (56:66) — three
    slice copies per batch, not a 224-wide gather.
    """
    runs: list[tuple[int, int, int]] = []
    i = 0
    while i < len(cols):
        j = i
        while j + 1 < len(cols) and cols[j + 1] == cols[j] + 1:
            j += 1
        runs.append((cols[i], cols[j] + 1, i))
        i = j + 1
    return runs


# ----------------------------------------------------------------------
# Writer
# ----------------------------------------------------------------------

_DEFAULT_CHUNK_SIZE: int = 25_000


def materialise(
    stream: Iterable[BCExample],
    out_dir: str | Path,
    *,
    max_examples: int | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    progress_every: int = 25_000,
) -> dict[str, int]:
    """Drive `stream`, bucket by decision_type, and write per-type memmap
    files + order.dat + manifest.json in a streaming, chunked fashion.

    `max_examples=None` (the default) drains the stream until exhaustion.
    Pass an int to cap the bundle at that many examples (counts against
    the stream after the decision-type filter, so the wall total is at
    most `max_examples`).

    `chunk_size` bounds peak RAM. A "flush all buffers" event fires
    every `chunk_size` accepted stream items — so all three per-type
    buffers get drained together, regardless of which one filled. This
    matters because the play type is ~98.8% of the stream and the
    wish/dragon_assignment types are ~1.2% each: if we waited for each
    per-type buffer to reach chunk_size independently, the wish/dragon
    buffers would never flush during a typical run and would
    accumulate millions of long-lived BCExamples, paying mounting GC
    cost (the smooth monotonic slowdown observed pre-fix). With the
    "flush all together" cadence, peak RAM stays bounded at roughly
    `chunk_size × feature_bytes` regardless of stream length. Default
    25,000 → ~1.7 GB peak. The order-pair buffer flushes on the same
    cadence.

    Returns a dict mapping decision_type -> rows written. The total is
    `sum(counts.values())` which may be less than `max_examples` (if
    the stream exhausted first) or equal to it (if the cap was hit).

    Re-running with the same `out_dir` truncates the prior bundle's
    files before writing, so the operation is idempotent.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Feature column split: indicator columns (packed to bits) vs the
    # continuous columns (kept f32). Derived once from the featurizer.
    cont_cols = list(CONTINUOUS_FEATURE_COLUMNS)
    bin_cols = _binary_feature_columns(FEATURIZER_OUTPUT_DIM, cont_cols)
    _bin_idx = np.asarray(bin_cols, dtype=np.intp)
    _cont_idx = np.asarray(cont_cols, dtype=np.intp)
    feat_bits_bytes = (len(bin_cols) + 7) // 8

    type_idx_for = {h: i for i, h in enumerate(TYPE_ORDER)}
    per_type_buf: dict[str, list[BCExample]] = {h: [] for h in TYPE_ORDER}
    # `rows_written[type]` is the number of rows ALREADY on disk for that
    # type. The row_idx of an example currently in the buffer is
    # `rows_written[type] + offset_in_buffer`, which matches its final
    # on-disk index after the next flush.
    rows_written: dict[str, int] = {h: 0 for h in TYPE_ORDER}
    order_buf: list[tuple[int, int]] = []
    bytes_written: dict[str, int] = {h: 0 for h in TYPE_ORDER}

    # Truncate any prior bundle's files so the writer is idempotent.
    # Open in "wb" (truncating) once up-front, close immediately, then
    # all subsequent writes use "ab" (append) — keeps fd churn low.
    _bits_paths = {h: out_dir / f"{h}_feat_bits.dat" for h in TYPE_ORDER}
    _cont_paths = {h: out_dir / f"{h}_feat_cont.dat" for h in TYPE_ORDER}
    _mask_paths = {h: out_dir / f"{h}_legal_mask.dat" for h in TYPE_ORDER}
    _meta_paths = {h: out_dir / f"{h}_meta.dat" for h in TYPE_ORDER}
    _order_path = out_dir / "order.dat"
    for p in (
        list(_bits_paths.values())
        + list(_cont_paths.values())
        + list(_mask_paths.values())
        + list(_meta_paths.values())
        + [_order_path]
    ):
        with p.open("wb"):
            pass  # truncate

    def _flush_type(type_name: str) -> None:
        """Stack + append the per-type buffer to disk, clear it."""
        examples = per_type_buf[type_name]
        if not examples:
            return
        mask_dim = HEAD_LOGIT_DIMS[type_name]
        feat = np.stack([e.features for e in examples]).astype(
            np.float32, copy=False,
        )
        # Split features: bit-pack the indicator columns (exactly 0/1), keep
        # the continuous columns as f32. Both packbits calls work per-row
        # (axis=1), so a chunked write is byte-identical to a single-shot
        # write — no cross-row dependency.
        bin_block = feat[:, _bin_idx]
        # Guard against silent corruption: packbits collapses anything
        # non-zero to 1, so a column we *think* is an indicator but isn't
        # 0/1 would be wrecked. This catches a featurizer change that made a
        # section continuous without updating CONTINUOUS_SECTIONS, or a
        # non-featurizer stream being materialised.
        if ((bin_block != 0.0) & (bin_block != 1.0)).any():
            bad = np.unique(np.asarray(_bin_idx)[
                np.where(((bin_block != 0.0) & (bin_block != 1.0)).any(axis=0))[0]
            ])
            raise ValueError(
                f"materialise() bit-packs indicator feature columns but found "
                f"non-0/1 values in column(s) {bad.tolist()[:10]} of type "
                f"'{type_name}'. The stream must be real featurizer output; if "
                f"the featurizer layout changed, update "
                f"featurizer.CONTINUOUS_SECTIONS."
            )
        feat_bits = np.packbits(bin_block.astype(np.uint8), axis=1)
        feat_cont = np.ascontiguousarray(feat[:, _cont_idx], dtype=np.float32)
        mask = np.packbits(
            np.stack(
                [np.asarray(e.legal_mask, dtype=np.uint8) for e in examples]
            ),
            axis=1,
        )
        meta = np.zeros(len(examples), dtype=META_DTYPE)
        for i, e in enumerate(examples):
            meta[i]["target"] = e.target
            meta[i]["sample_weight"] = e.sample_weight
            meta[i]["skill_decile"] = e.skill_decile
            meta[i]["round_outcome"] = e.round_outcome
            meta[i]["game_won"] = (
                _GAME_WON_NONE_CODE if e.game_won is None
                else (1 if e.game_won else 0)
            )
        with _bits_paths[type_name].open("ab") as fh:
            feat_bits.tofile(fh)
        with _cont_paths[type_name].open("ab") as fh:
            feat_cont.tofile(fh)
        with _mask_paths[type_name].open("ab") as fh:
            mask.tofile(fh)
        with _meta_paths[type_name].open("ab") as fh:
            meta.tofile(fh)
        n_rows = len(examples)
        rows_written[type_name] += n_rows
        bytes_written[type_name] += (
            (
                feat_bits_bytes
                + len(cont_cols) * 4
                + _packed_mask_bytes(mask_dim)
                + META_DTYPE.itemsize
            )
            * n_rows
        )
        per_type_buf[type_name] = []
        # Drop the np arrays so the next chunk's stack doesn't pile up
        # before GC sweeps.
        del feat, feat_bits, feat_cont, mask, meta, examples

    def _flush_order() -> None:
        if not order_buf:
            return
        arr = np.array(order_buf, dtype=np.uint32)
        with _order_path.open("ab") as fh:
            arr.tofile(fh)
        order_buf.clear()

    def _flush_all() -> None:
        """Flush every per-type buffer + the order buffer together.

        Keeps wish/dragon buffers from accumulating across the entire
        run. With per-type thresholds, those buffers (~1.2% of stream
        each) never reach chunk_size during a typical run and bleed
        BCExample objects into Python's long-lived heap, causing GC to
        get slower as the run progresses.
        """
        for type_name in TYPE_ORDER:
            _flush_type(type_name)
        _flush_order()

    t0 = time.perf_counter()
    n = 0
    n_since_last_flush = 0
    last_t = t0
    last_n = 0
    cap_str = "all" if max_examples is None else f"up to {max_examples}"
    log.info(
        "materialising %s BCExamples to %s (chunk_size=%d) ...",
        cap_str, out_dir, chunk_size,
    )
    for ex in stream:
        bucket = per_type_buf.get(ex.decision_type)
        if bucket is None:
            # Decision type outside HEAD_LOGIT_DIMS; defensive skip.
            continue
        order_buf.append((
            type_idx_for[ex.decision_type],
            rows_written[ex.decision_type] + len(bucket),
        ))
        bucket.append(ex)
        n += 1
        n_since_last_flush += 1
        # Flush ALL per-type buffers + order_buf together when the
        # combined intake reaches chunk_size. This bounds every
        # buffer's growth proportionally to its type's stream share —
        # play gets ~98.8% × chunk_size, wish/dragon ~1.2% each — so
        # no single buffer accumulates millions of BCExamples across a
        # long run. The "play buffer fills first" path is gone; we
        # gate on total intake instead.
        if n_since_last_flush >= chunk_size:
            _flush_all()
            n_since_last_flush = 0
        if max_examples is not None and n >= max_examples:
            break
        if n - last_n >= progress_every:
            now = time.perf_counter()
            rate = (n - last_n) / max(1e-9, now - last_t)
            total_str = (
                f"/{max_examples}" if max_examples is not None else ""
            )
            log.info("  ... %d%s  (%.0f ex/s)", n, total_str, rate)
            last_t = now
            last_n = n

    # Drain remaining partial buffers.
    _flush_all()

    elapsed = time.perf_counter() - t0
    counts = dict(rows_written)
    log.info(
        "streamed %d ex in %.1fs (%.0f ex/s); per-type counts: %s",
        n, elapsed, n / max(1e-9, elapsed), counts,
    )
    for type_name in TYPE_ORDER:
        if counts[type_name] == 0:
            continue
        log.info(
            "  %s: wrote %d rows, %.2f GB",
            type_name, counts[type_name], bytes_written[type_name] / 1e9,
        )
    log.info(
        "  order.dat: %.2f MB (%d rows)",
        _order_path.stat().st_size / 1e6, n,
    )

    manifest = {
        "schema_version": MATERIALISED_SCHEMA_VERSION,
        "featurizer_version": FEATURIZER_VERSION,
        "action_space_version": ACTION_SPACE_VERSION,
        "feature_dim": FEATURIZER_OUTPUT_DIM,
        "legal_mask_packed": True,
        "features_packed": True,
        "continuous_feature_columns": cont_cols,
        "head_logit_dims": dict(HEAD_LOGIT_DIMS),
        "type_order": TYPE_ORDER,
        "meta_dtype": [(name, str(t)) for name, t in META_DTYPE.descr],
        "counts": counts,
        "total": n,
        "files": {
            type_name: {
                "feat_bits": f"{type_name}_feat_bits.dat",
                "feat_cont": f"{type_name}_feat_cont.dat",
                "legal_mask": f"{type_name}_legal_mask.dat",
                "meta": f"{type_name}_meta.dat",
                "shape_feat_bits": [counts[type_name], feat_bits_bytes],
                "shape_feat_cont": [counts[type_name], len(cont_cols)],
                "shape_legal_mask": [
                    counts[type_name], HEAD_LOGIT_DIMS[type_name],
                ],
                "shape_legal_mask_packed": [
                    counts[type_name],
                    _packed_mask_bytes(HEAD_LOGIT_DIMS[type_name]),
                ],
            }
            for type_name in TYPE_ORDER if counts[type_name] > 0
        },
        "order_file": "order.dat",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("manifest.json written. Total %d examples, out_dir=%s", n, out_dir)
    return counts


# ----------------------------------------------------------------------
# Reader
# ----------------------------------------------------------------------

class MemmapBCDataset(Iterable[BCExample]):
    """Disk-backed BC dataset.

    Opens the per-type memmaps written by `materialise()` and yields
    examples either as `BCExample` instances (`__iter__`, drop-in for
    `ParquetBCDataset`) or as **pre-stacked batches** of tensors keyed by
    decision_type (`iter_batches`, the fast path that bypasses per-row
    Python-object construction).

    Version-pin guard: refuses to load if the manifest's
    `featurizer_version`, `action_space_version`, or `schema_version`
    don't match the live code's expectations. Modelled on the existing
    `ParquetBCDataset` pin behaviour.

    The reader holds memmap views, not copies. Per-row/per-batch access
    reconstructs the dense feature vector from its bit-packed indicator block
    + continuous f32 block, and unpacks the legal mask from its bit-packed
    form — small copies sized exactly to what the consumer needs. The batched
    path fancy-indexes the memmaps (one contiguous copy per block) then does
    the reconstruction once for the whole batch.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        expected_featurizer_version: str = FEATURIZER_VERSION,
        expected_action_space_version: str = ACTION_SPACE_VERSION,
        expected_schema_version: int = MATERIALISED_SCHEMA_VERSION,
    ) -> None:
        self.data_dir = Path(data_dir)
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no manifest.json under {self.data_dir}; "
                "did you run `python -m tichu_training.cli.materialise_bc`?"
            )
        self.manifest: dict = json.loads(manifest_path.read_text())

        # Three-pronged version check. Each pin invalidates the bundle
        # for a distinct reason and the error message says which.
        _check_pin(
            "featurizer_version",
            self.manifest.get("featurizer_version"),
            expected_featurizer_version,
        )
        _check_pin(
            "action_space_version",
            self.manifest.get("action_space_version"),
            expected_action_space_version,
        )
        _check_pin(
            "schema_version",
            self.manifest.get("schema_version"),
            expected_schema_version,
        )

        self.type_order: list[str] = self.manifest["type_order"]
        self.feature_dim: int = self.manifest["feature_dim"]
        self.counts: dict[str, int] = self.manifest["counts"]
        self.head_logit_dims: dict[str, int] = self.manifest["head_logit_dims"]

        # Feature reconstruction plan. The bundle stores indicator columns
        # bit-packed (feat_bits) and continuous columns as f32 (feat_cont);
        # rebuild the dense (D,) vector by scattering each block back into its
        # original columns. Precompute the contiguous runs once so per-batch
        # reconstruction is a few slice copies, not a fancy-index gather.
        cont_cols = list(self.manifest["continuous_feature_columns"])
        bin_cols = _binary_feature_columns(self.feature_dim, cont_cols)
        self._n_bin = len(bin_cols)
        self._feat_bits_bytes = (self._n_bin + 7) // 8
        self._bin_runs = _contiguous_runs(bin_cols)
        self._cont_runs = _contiguous_runs(cont_cols)

        # Reconstruct the meta dtype from the manifest's recorded
        # (name, dtype-str) pairs. A future bundle adding fields stays
        # readable as long as the existing fields keep their slots.
        self._meta_dtype = np.dtype([
            (n, np.dtype(t)) for n, t in self.manifest["meta_dtype"]
        ])

        self._feat_bits: dict[str, np.memmap] = {}
        self._feat_cont: dict[str, np.memmap] = {}
        self._mask: dict[str, np.memmap] = {}
        self._meta: dict[str, np.memmap] = {}
        for type_name, info in self.manifest["files"].items():
            n_rows = self.counts[type_name]
            mask_dim = self.head_logit_dims[type_name]
            self._feat_bits[type_name] = np.memmap(
                self.data_dir / info["feat_bits"],
                dtype=np.uint8, mode="r",
                shape=(n_rows, self._feat_bits_bytes),
            )
            self._feat_cont[type_name] = np.memmap(
                self.data_dir / info["feat_cont"],
                dtype=np.float32, mode="r",
                shape=(n_rows, len(cont_cols)),
            )
            self._mask[type_name] = np.memmap(
                self.data_dir / info["legal_mask"],
                dtype=np.uint8, mode="r",
                shape=(n_rows, _packed_mask_bytes(mask_dim)),
            )
            self._meta[type_name] = np.memmap(
                self.data_dir / info["meta"],
                dtype=self._meta_dtype, mode="r",
                shape=(n_rows,),
            )

        self._order = np.memmap(
            self.data_dir / self.manifest["order_file"],
            dtype=np.uint32, mode="r",
            shape=(self.manifest["total"], 2),
        )

    @property
    def n_rows(self) -> int:
        return int(self.manifest["total"])

    @property
    def total(self) -> int:
        return self.n_rows

    def _reconstruct_features(
        self, bits: np.ndarray, cont: np.ndarray,
    ) -> np.ndarray:
        """Rebuild a dense `(B, D)` float32 feature batch from its bit-packed
        indicator block + continuous f32 block. `bits` is `(B, feat_bits_bytes)`
        uint8, `cont` is `(B, n_cont)` f32. Scatter is done as a few slice
        copies over the precomputed contiguous runs (cheap, uint8→f32 casts
        on assignment)."""
        feat = np.empty((bits.shape[0], self.feature_dim), dtype=np.float32)
        binvals = np.unpackbits(bits, axis=1)
        for dst0, dst1, src0 in self._bin_runs:
            feat[:, dst0:dst1] = binvals[:, src0 : src0 + (dst1 - dst0)]
        for dst0, dst1, src0 in self._cont_runs:
            feat[:, dst0:dst1] = cont[:, src0 : src0 + (dst1 - dst0)]
        return feat

    def __iter__(self) -> Iterator[BCExample]:
        """Yield one `BCExample` per row in original emission order.

        Use this when interoperating with code that expects
        `Iterable[BCExample]` (e.g. `ParallelParquetBCDataset` compat). Each
        row's features are reconstructed from the bit-packed + continuous
        blocks and its mask is unpacked — small per-row copies. For training
        throughput, prefer `iter_batches`.
        """
        type_order = self.type_order
        feat_bits = self._feat_bits
        feat_cont = self._feat_cont
        masks = self._mask
        meta = self._meta
        order = self._order
        mask_dims = self.head_logit_dims
        for i in range(len(order)):
            type_idx = int(order[i, 0])
            row_idx = int(order[i, 1])
            type_name = type_order[type_idx]
            m = meta[type_name][row_idx]
            game_won_code = int(m["game_won"])
            yield BCExample(
                decision_type=type_name,
                features=self._reconstruct_features(
                    feat_bits[type_name][row_idx][None, :],
                    feat_cont[type_name][row_idx][None, :],
                )[0],
                target=int(m["target"]),
                legal_mask=np.unpackbits(masks[type_name][row_idx])[
                    : mask_dims[type_name]
                ].astype(bool),
                sample_weight=float(m["sample_weight"]),
                skill_decile=int(m["skill_decile"]),
                round_outcome=float(m["round_outcome"]),
                game_won=(
                    None if game_won_code == _GAME_WON_NONE_CODE
                    else bool(game_won_code)
                ),
            )

    def iter_batches(
        self,
        batch_size: int,
        *,
        drop_last: bool = False,
        preserve_order: bool = True,
    ) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
        """Yield pre-stacked per-head batches.

        Each yield is `(decision_type, batch_dict)` where `batch_dict`
        holds numpy arrays sized `(B, ...)` ready to be wrapped in
        tensors by the training loop:

          features      : float32  (B, D_feat)
          legal_mask    : bool     (B, K_head)
          target        : int64    (B,)
          sample_weight : float32  (B,)
          skill_decile  : int64    (B,)
          round_outcome : float32  (B,)
          game_won      : int8     (B,)   (-1/0/1 sentinel; consumer may
                                            choose to mask -1 rows)

        With `preserve_order=True`, examples are emitted in original
        stream order via `order.dat`; the fast batched path becomes
        almost-as-fast-as the unordered path because the per-type
        buffers fill round-robinishly and the memmap fancy-index reads
        are still vectorised. With `preserve_order=False`, the reader
        walks each type's memmap contiguously — the cheapest possible
        access pattern, suitable when the higher layer is shuffling
        anyway. `drop_last` matches PyTorch's DataLoader semantics.

        The output keys match `tichu_training.bc.training._to_tensors`
        so a caller can replace the per-example route + `_to_tensors`
        call with `torch.from_numpy(batch[k])` directly.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if preserve_order:
            buffers: dict[str, list[int]] = defaultdict(list)
            order = self._order
            type_order = self.type_order
            for i in range(len(order)):
                type_idx = int(order[i, 0])
                row_idx = int(order[i, 1])
                type_name = type_order[type_idx]
                buf = buffers[type_name]
                buf.append(row_idx)
                if len(buf) >= batch_size:
                    yield type_name, self._gather_batch(type_name, buf)
                    buf.clear()
            if not drop_last:
                for type_name, buf in buffers.items():
                    if buf:
                        yield type_name, self._gather_batch(type_name, buf)
        else:
            for type_name in self.type_order:
                n_rows = self.counts.get(type_name, 0)
                if n_rows == 0:
                    continue
                start = 0
                while start + batch_size <= n_rows:
                    idx = list(range(start, start + batch_size))
                    yield type_name, self._gather_batch(type_name, idx)
                    start += batch_size
                if start < n_rows and not drop_last:
                    idx = list(range(start, n_rows))
                    yield type_name, self._gather_batch(type_name, idx)

    def _gather_batch(
        self, type_name: str, row_indices: list[int],
    ) -> dict[str, np.ndarray]:
        """Vectorised gather of one batch worth of rows.

        Memmap fancy-indexing copies the selected rows into fresh contiguous
        arrays; features are then reconstructed from their bit-packed +
        continuous blocks and the mask is unpacked — exactly the shapes the
        training loop wants. Replaces per-row BCExample construction in the
        `__iter__` path: no Python-object allocation, dataclass __init__, or
        per-row attribute lookups.
        """
        idx = np.asarray(row_indices, dtype=np.int64)
        feat = self._reconstruct_features(
            np.asarray(self._feat_bits[type_name][idx]),
            np.asarray(self._feat_cont[type_name][idx]),
        )
        # Unpack the bit-packed mask rows back to (B, mask_dim) bool. packbits
        # padded each row up to a byte boundary; slice off the padding bits.
        packed = np.asarray(self._mask[type_name][idx])
        mask = np.unpackbits(packed, axis=1)[
            :, : self.head_logit_dims[type_name]
        ].astype(bool)
        meta = np.asarray(self._meta[type_name][idx])
        return {
            "features": feat,
            "legal_mask": mask,
            "target": meta["target"].astype(np.int64),
            "sample_weight": meta["sample_weight"].astype(np.float32),
            "skill_decile": meta["skill_decile"].astype(np.int64),
            "round_outcome": meta["round_outcome"].astype(np.float32),
            "game_won": meta["game_won"].astype(np.int8),
        }


def _check_pin(name: str, found, expected) -> None:
    """Raise VersionMismatchError if the manifest's pin disagrees with
    the live code. Keeps the bundle/code coupling explicit — the cost
    of a quietly-stale bundle would be a silently-corrupt training run.
    """
    if found != expected:
        raise VersionMismatchError(
            f"materialised bundle {name} mismatch: bundle={found!r} "
            f"live={expected!r}; re-run materialise_bc against current code"
        )
