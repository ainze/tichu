"""Disk-backed BC dataset: writer + reader for the materialised corpus.

The replay-on-the-fly path of ADR-0011 is CPU-bound on the BSW engine
(`replay_round` / `engine.step` / `legal_actions` — ~75% of worker CPU
in the 2026-05-29 profile). Materialising the BCExample stream once,
to per-decision-type numpy memmaps, bypasses replay entirely. See
[ADR-0014](../../../docs/adr/0014-pre-featurise-bc-corpus.md) for the
trade analysis vs ADR-0011's α.2 (replay) and rejected β (parquet-
cached state) / γ (dense feature memmap, no per-type split).

On-disk layout for a slice of N examples:

  <out_dir>/
    manifest.json
    play_features.dat               (N_play, D_feat)  float32
    play_legal_mask.dat             (N_play, 1809)    uint8
    play_meta.dat                   (N_play,)         structured (META_DTYPE)
    wish_features.dat               (N_wish, D_feat)  float32
    wish_legal_mask.dat             (N_wish, 14)      uint8
    wish_meta.dat                   (N_wish,)         structured
    dragon_assignment_features.dat  (N_da, D_feat)    float32
    dragon_assignment_legal_mask.dat (N_da, 2)        uint8
    dragon_assignment_meta.dat      (N_da,)           structured
    order.dat                       (N_total, 2)      uint32: (type_idx, row_idx_in_type)

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
from tichu_training.bc.heads import HEAD_LOGIT_DIMS
from tichu_training.checkpoint import VersionMismatchError
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM, FEATURIZER_VERSION


log = logging.getLogger("tichu_training.bc.materialised")


# Bumped whenever the on-disk layout, meta dtype, or manifest schema
# changes in a non-backwards-compatible way. Independent of
# FEATURIZER_VERSION (which invalidates the feature *content* not the
# layout) and ACTION_SPACE_VERSION (which invalidates target / mask
# semantics not layout).
MATERIALISED_SCHEMA_VERSION = 1


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
    _feat_paths = {h: out_dir / f"{h}_features.dat" for h in TYPE_ORDER}
    _mask_paths = {h: out_dir / f"{h}_legal_mask.dat" for h in TYPE_ORDER}
    _meta_paths = {h: out_dir / f"{h}_meta.dat" for h in TYPE_ORDER}
    _order_path = out_dir / "order.dat"
    for p in (
        list(_feat_paths.values())
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
        mask = np.stack(
            [np.asarray(e.legal_mask, dtype=np.uint8) for e in examples]
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
        with _feat_paths[type_name].open("ab") as fh:
            feat.tofile(fh)
        with _mask_paths[type_name].open("ab") as fh:
            mask.tofile(fh)
        with _meta_paths[type_name].open("ab") as fh:
            meta.tofile(fh)
        n_rows = len(examples)
        rows_written[type_name] += n_rows
        bytes_written[type_name] += (
            (FEATURIZER_OUTPUT_DIM * 4 + mask_dim + META_DTYPE.itemsize)
            * n_rows
        )
        per_type_buf[type_name] = []
        # Drop the np arrays so the next chunk's stack doesn't pile up
        # before GC sweeps.
        del feat, mask, meta, examples

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
        "head_logit_dims": dict(HEAD_LOGIT_DIMS),
        "type_order": TYPE_ORDER,
        "meta_dtype": [(name, str(t)) for name, t in META_DTYPE.descr],
        "counts": counts,
        "total": n,
        "files": {
            type_name: {
                "features": f"{type_name}_features.dat",
                "legal_mask": f"{type_name}_legal_mask.dat",
                "meta": f"{type_name}_meta.dat",
                "shape_features": [counts[type_name], FEATURIZER_OUTPUT_DIM],
                "shape_legal_mask": [
                    counts[type_name], HEAD_LOGIT_DIMS[type_name],
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

    The reader holds memmap views, not copies. Per-row access via
    `__iter__` is zero-copy until the consumer's `_to_tensors` calls
    `np.stack` to build a batch. The batched path indexes the memmap
    with a fancy-index list, which produces a single contiguous copy
    sized exactly to the batch tensor — what the training loop needs.
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

        # Reconstruct the meta dtype from the manifest's recorded
        # (name, dtype-str) pairs. A future bundle adding fields stays
        # readable as long as the existing fields keep their slots.
        self._meta_dtype = np.dtype([
            (n, np.dtype(t)) for n, t in self.manifest["meta_dtype"]
        ])

        self._features: dict[str, np.memmap] = {}
        self._mask: dict[str, np.memmap] = {}
        self._meta: dict[str, np.memmap] = {}
        for type_name, info in self.manifest["files"].items():
            n_rows = self.counts[type_name]
            mask_dim = self.head_logit_dims[type_name]
            self._features[type_name] = np.memmap(
                self.data_dir / info["features"],
                dtype=np.float32, mode="r",
                shape=(n_rows, self.feature_dim),
            )
            self._mask[type_name] = np.memmap(
                self.data_dir / info["legal_mask"],
                dtype=np.uint8, mode="r",
                shape=(n_rows, mask_dim),
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

    def __iter__(self) -> Iterator[BCExample]:
        """Yield one `BCExample` per row in original emission order.

        Per-row reads are memmap slice views — no copy until the
        consumer batches. Use this when interoperating with code that
        expects `Iterable[BCExample]` (e.g. `ParallelParquetBCDataset`
        compat). For training throughput, prefer `iter_batches`.
        """
        type_order = self.type_order
        features = self._features
        masks = self._mask
        meta = self._meta
        order = self._order
        for i in range(len(order)):
            type_idx = int(order[i, 0])
            row_idx = int(order[i, 1])
            type_name = type_order[type_idx]
            m = meta[type_name][row_idx]
            game_won_code = int(m["game_won"])
            yield BCExample(
                decision_type=type_name,
                features=features[type_name][row_idx],
                target=int(m["target"]),
                legal_mask=masks[type_name][row_idx].astype(bool),
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

        Memmap fancy-indexing copies the selected rows into a fresh
        contiguous array — exactly the shape the training loop wants.
        This is the single memcpy that replaces per-row BCExample
        construction in the `__iter__` path. Avoids Python-object
        allocation, dataclass __init__, and per-row attribute lookups.
        """
        idx = np.asarray(row_indices, dtype=np.int64)
        feat = np.asarray(self._features[type_name][idx])
        mask = np.asarray(self._mask[type_name][idx]).astype(bool)
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
