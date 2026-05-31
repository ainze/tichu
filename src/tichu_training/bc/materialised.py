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


# The bit-packing leaf helpers now live in `bc.packing` (shared across the
# per-trainer bundles, ADR-0020). Re-exported under their original private
# names so this module's body — and the byte-identity invariant it backs —
# are unchanged.
from tichu_training.bc.packing import (  # noqa: E402
    binary_feature_columns as _binary_feature_columns,
    contiguous_runs as _contiguous_runs,
    packed_mask_bytes as _packed_mask_bytes,
)


# ----------------------------------------------------------------------
# Writer
# ----------------------------------------------------------------------

_DEFAULT_CHUNK_SIZE: int = 25_000


class BCBundleWriter:
    """Push-style writer for the BC bundle. `add_many()` accepts BCExamples
    incrementally (buckets by decision_type, flushes on the chunk cadence);
    `close()` drains the buffers and writes `manifest.json`.

    Lifted verbatim from the former `materialise()` function body so the
    consolidated parse pass ([ADR-0020](../../../docs/adr/0020-one-parse-pass-many-task-bundles.md))
    can drive it row-by-row alongside the other task writers. The
    byte-identity invariant (`test_to_bundle.py`) pins that the output is
    unchanged. `materialise()` remains as a thin pull-style wrapper.

    `chunk_size` bounds peak RAM. A "flush all buffers" event fires every
    `chunk_size` accepted items — all three per-type buffers drain together,
    regardless of which one filled. This matters because play is ~98.8% of the
    stream and wish/dragon ~1.2% each: per-type thresholds would let the small
    buffers accumulate millions of long-lived BCExamples (mounting GC cost).
    The cadence persists across `add_many()` calls, so feeding one game at a
    time still bounds RAM. Default 25,000 → ~1.7 GB peak.
    """

    def __init__(
        self, out_dir: str | Path, *, chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size

        # Feature column split: indicator columns (packed to bits) vs the
        # continuous columns (kept f32). Derived once from the featurizer.
        self.cont_cols = list(CONTINUOUS_FEATURE_COLUMNS)
        bin_cols = _binary_feature_columns(FEATURIZER_OUTPUT_DIM, self.cont_cols)
        self._bin_idx = np.asarray(bin_cols, dtype=np.intp)
        self._cont_idx = np.asarray(self.cont_cols, dtype=np.intp)
        self.feat_bits_bytes = (len(bin_cols) + 7) // 8

        self._type_idx_for = {h: i for i, h in enumerate(TYPE_ORDER)}
        self._per_type_buf: dict[str, list[BCExample]] = {h: [] for h in TYPE_ORDER}
        # `rows_written[type]` is the number of rows ALREADY on disk for that
        # type. The row_idx of an example currently in the buffer is
        # `rows_written[type] + offset_in_buffer`, matching its final on-disk
        # index after the next flush.
        self._rows_written: dict[str, int] = {h: 0 for h in TYPE_ORDER}
        self._order_buf: list[tuple[int, int]] = []
        self._bytes_written: dict[str, int] = {h: 0 for h in TYPE_ORDER}

        # Truncate any prior bundle's files so the writer is idempotent.
        self._bits_paths = {h: self.out_dir / f"{h}_feat_bits.dat" for h in TYPE_ORDER}
        self._cont_paths = {h: self.out_dir / f"{h}_feat_cont.dat" for h in TYPE_ORDER}
        self._mask_paths = {h: self.out_dir / f"{h}_legal_mask.dat" for h in TYPE_ORDER}
        self._meta_paths = {h: self.out_dir / f"{h}_meta.dat" for h in TYPE_ORDER}
        self._order_path = self.out_dir / "order.dat"
        for p in (
            list(self._bits_paths.values())
            + list(self._cont_paths.values())
            + list(self._mask_paths.values())
            + list(self._meta_paths.values())
            + [self._order_path]
        ):
            with p.open("wb"):
                pass  # truncate

        self._n = 0
        self._n_since_last_flush = 0
        self._t0 = time.perf_counter()

    def _flush_type(self, type_name: str) -> None:
        """Stack + append the per-type buffer to disk, clear it."""
        examples = self._per_type_buf[type_name]
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
        bin_block = feat[:, self._bin_idx]
        # Guard against silent corruption: packbits collapses anything
        # non-zero to 1, so a column we *think* is an indicator but isn't
        # 0/1 would be wrecked.
        if ((bin_block != 0.0) & (bin_block != 1.0)).any():
            bad = np.unique(np.asarray(self._bin_idx)[
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
        feat_cont = np.ascontiguousarray(feat[:, self._cont_idx], dtype=np.float32)
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
        with self._bits_paths[type_name].open("ab") as fh:
            feat_bits.tofile(fh)
        with self._cont_paths[type_name].open("ab") as fh:
            feat_cont.tofile(fh)
        with self._mask_paths[type_name].open("ab") as fh:
            mask.tofile(fh)
        with self._meta_paths[type_name].open("ab") as fh:
            meta.tofile(fh)
        n_rows = len(examples)
        self._rows_written[type_name] += n_rows
        self._bytes_written[type_name] += (
            (
                self.feat_bits_bytes
                + len(self.cont_cols) * 4
                + _packed_mask_bytes(mask_dim)
                + META_DTYPE.itemsize
            )
            * n_rows
        )
        self._per_type_buf[type_name] = []
        del feat, feat_bits, feat_cont, mask, meta, examples

    def _flush_order(self) -> None:
        if not self._order_buf:
            return
        arr = np.array(self._order_buf, dtype=np.uint32)
        with self._order_path.open("ab") as fh:
            arr.tofile(fh)
        self._order_buf.clear()

    def _flush_all(self) -> None:
        """Flush every per-type buffer + the order buffer together."""
        for type_name in TYPE_ORDER:
            self._flush_type(type_name)
        self._flush_order()

    def add_many(
        self,
        stream: Iterable[BCExample],
        *,
        max_examples: int | None = None,
        progress_every: int = 25_000,
    ) -> None:
        """Consume `stream`, bucketing by decision_type and flushing on the
        chunk cadence. May be called repeatedly; cadence + counts persist."""
        last_t = time.perf_counter()
        last_n = self._n
        for ex in stream:
            bucket = self._per_type_buf.get(ex.decision_type)
            if bucket is None:
                continue  # decision type outside HEAD_LOGIT_DIMS; defensive skip
            self._order_buf.append((
                self._type_idx_for[ex.decision_type],
                self._rows_written[ex.decision_type] + len(bucket),
            ))
            bucket.append(ex)
            self._n += 1
            self._n_since_last_flush += 1
            if self._n_since_last_flush >= self.chunk_size:
                self._flush_all()
                self._n_since_last_flush = 0
            if max_examples is not None and self._n >= max_examples:
                break
            if self._n - last_n >= progress_every:
                now = time.perf_counter()
                rate = (self._n - last_n) / max(1e-9, now - last_t)
                log.info("  ... %d  (%.0f ex/s)", self._n, rate)
                last_t = now
                last_n = self._n

    def close(self) -> dict[str, int]:
        """Drain remaining buffers, write the manifest, return per-type counts."""
        self._flush_all()
        counts = dict(self._rows_written)
        elapsed = time.perf_counter() - self._t0
        log.info(
            "streamed %d ex in %.1fs (%.0f ex/s); per-type counts: %s",
            self._n, elapsed, self._n / max(1e-9, elapsed), counts,
        )
        manifest = {
            "schema_version": MATERIALISED_SCHEMA_VERSION,
            "featurizer_version": FEATURIZER_VERSION,
            "action_space_version": ACTION_SPACE_VERSION,
            "feature_dim": FEATURIZER_OUTPUT_DIM,
            "legal_mask_packed": True,
            "features_packed": True,
            "continuous_feature_columns": self.cont_cols,
            "head_logit_dims": dict(HEAD_LOGIT_DIMS),
            "type_order": TYPE_ORDER,
            "meta_dtype": [(name, str(t)) for name, t in META_DTYPE.descr],
            "counts": counts,
            "total": self._n,
            "files": {
                type_name: {
                    "feat_bits": f"{type_name}_feat_bits.dat",
                    "feat_cont": f"{type_name}_feat_cont.dat",
                    "legal_mask": f"{type_name}_legal_mask.dat",
                    "meta": f"{type_name}_meta.dat",
                    "shape_feat_bits": [counts[type_name], self.feat_bits_bytes],
                    "shape_feat_cont": [counts[type_name], len(self.cont_cols)],
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
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        log.info(
            "manifest.json written. Total %d examples, out_dir=%s",
            self._n, self.out_dir,
        )
        return counts


def materialise(
    stream: Iterable[BCExample],
    out_dir: str | Path,
    *,
    max_examples: int | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    progress_every: int = 25_000,
) -> dict[str, int]:
    """Pull-style wrapper over `BCBundleWriter`: drive `stream` to a bundle at
    `out_dir` and return per-type counts. Output is byte-identical to feeding
    the same examples through `BCBundleWriter.add_many` (pinned by
    `test_to_bundle.py`).

    `max_examples=None` drains the stream; an int caps the bundle at that many
    examples. Re-running with the same `out_dir` truncates first (idempotent).
    """
    writer = BCBundleWriter(out_dir, chunk_size=chunk_size)
    writer.add_many(stream, max_examples=max_examples, progress_every=progress_every)
    return writer.close()


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


# `_check_pin` now lives in `bc.packing` (shared across bundles). Re-exported
# under its original private name for any external importer.
from tichu_training.bc.packing import check_pin as _check_pin  # noqa: E402
