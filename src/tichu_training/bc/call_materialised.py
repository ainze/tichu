"""Disk-backed Calls dataset: writer + reader for the materialised Calls
bundle.

The standalone Call Networks ([ADR-0007](../../../docs/adr/0007-calls-are-standalone-networks.md))
make a binary call/skip decision. Grand-Tichu is parse-bound (synthetic
deal-time state); Tichu is replay-bound (first-non-pass-play state,
[ADR-0018](../../../docs/adr/0018-tichu-call-featurises-at-first-non-pass-play.md)).
Both call types live in one bundle directory — `train_calls` trains one
network at a time and reads its type's slice — mirroring how the BC bundle
holds three decision types. This is the calls slice of "one parse pass,
many task bundles" ([ADR-0020](../../../docs/adr/0020-one-parse-pass-many-task-bundles.md)),
reusing the shared bit-packing of `bc.packing`.

On-disk layout for a slice of N examples of type T:

  <out_dir>/
    manifest.json
    T_feat_bits.dat   (N, 27)   uint8   (packbits of 214 indicator cols)
    T_feat_cont.dat   (N, 10)   float32 (continuous columns)
    T_meta.dat        (N,)      structured (META_DTYPE)

There is **no legal-mask file** — the action space is binary. The target
is the binary call/skip label, stored in meta. No `order.dat`: each call
type is read contiguously (no cross-type emission-order requirement).

Version pins (featurizer / action-space / schema) are recorded in the
manifest; `MemmapCallDataset.__init__` raises `VersionMismatchError` on any
mismatch.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np

from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.call_training import CallExample
from tichu_training.bc.packing import FeatureCodec, check_pin
from tichu_training.featurizer import (
    CONTINUOUS_FEATURE_COLUMNS,
    FEATURIZER_OUTPUT_DIM,
    FEATURIZER_VERSION,
)


log = logging.getLogger("tichu_training.bc.call_materialised")


# Bumped on any non-backwards-compatible layout / meta / manifest change.
# Independent of FEATURIZER_VERSION / ACTION_SPACE_VERSION and of the other
# per-trainer bundles' schema versions (ADR-0020).
CALL_SCHEMA_VERSION = 1


# Stable ordering of the call types stored in one bundle directory.
CALL_TYPE_ORDER: list[str] = ["call_tichu", "call_grand_tichu"]


# Per-row metadata. Binary target (u1); skill_decile (u1, 0..10);
# sample_weight (f4); game_id (u4) + round_id (u1) provenance (ADR-0020).
# 11 bytes/row.
META_DTYPE: np.dtype = np.dtype([
    ("target", "u1"),
    ("sample_weight", "f4"),
    ("skill_decile", "u1"),
    ("game_id", "u4"),
    ("round_id", "u1"),
])


_DEFAULT_CHUNK_SIZE: int = 25_000


def _gid_to_u4(game_id) -> int:
    """CallExample.game_id is the string game key; the bundle stores it as a
    u4 provenance column. BSW ids are numeric, so int-parse; 0 if not."""
    try:
        return int(game_id)
    except (TypeError, ValueError):
        return 0


class CallBundleWriter:
    """Push-style writer for the calls bundle (both call types in one dir).
    `add_many(call_type, stream)` buffers per type and flushes on the chunk
    cadence; `close()` drains every type and writes `manifest.json`. Driven
    row-by-row by the consolidated parse pass (ADR-0020). A call type's files
    are created lazily on first feed, so an unfed type leaves no files."""

    def __init__(
        self, out_dir: str | Path, *, chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.cont_cols = list(CONTINUOUS_FEATURE_COLUMNS)
        self._codec = FeatureCodec(FEATURIZER_OUTPUT_DIM, self.cont_cols)
        self._buf: dict[str, list[CallExample]] = {t: [] for t in CALL_TYPE_ORDER}
        self._rows_written: dict[str, int] = {t: 0 for t in CALL_TYPE_ORDER}
        self._seen: dict[str, int] = {t: 0 for t in CALL_TYPE_ORDER}
        self._opened: set[str] = set()
        self._t0 = time.perf_counter()

    def _paths(self, call_type: str) -> tuple[Path, Path, Path]:
        return (
            self.out_dir / f"{call_type}_feat_bits.dat",
            self.out_dir / f"{call_type}_feat_cont.dat",
            self.out_dir / f"{call_type}_meta.dat",
        )

    def _flush_type(self, call_type: str) -> None:
        buf = self._buf[call_type]
        if not buf:
            return
        bits_path, cont_path, meta_path = self._paths(call_type)
        feat = np.stack([e.features for e in buf]).astype(np.float32, copy=False)
        feat_bits, feat_cont = self._codec.pack(feat)
        meta = np.zeros(len(buf), dtype=META_DTYPE)
        for i, e in enumerate(buf):
            meta[i]["target"] = int(e.target)
            meta[i]["sample_weight"] = e.sample_weight
            meta[i]["skill_decile"] = e.skill_decile
            meta[i]["game_id"] = _gid_to_u4(e.game_id)
            meta[i]["round_id"] = e.round_id
        with bits_path.open("ab") as fh:
            feat_bits.tofile(fh)
        with cont_path.open("ab") as fh:
            feat_cont.tofile(fh)
        with meta_path.open("ab") as fh:
            meta.tofile(fh)
        self._rows_written[call_type] += len(buf)
        buf.clear()
        del feat, feat_bits, feat_cont, meta

    def add_many(
        self,
        call_type: str,
        stream: Iterable[CallExample],
        *,
        max_examples: int | None = None,
    ) -> None:
        if call_type not in self._buf:
            raise KeyError(
                f"unknown call_type {call_type!r}; expected one of {CALL_TYPE_ORDER}"
            )
        if call_type not in self._opened:
            for p in self._paths(call_type):
                with p.open("wb"):
                    pass  # truncate on first feed
            self._opened.add(call_type)
        buf = self._buf[call_type]
        for ex in stream:
            buf.append(ex)
            self._seen[call_type] += 1
            if len(buf) >= self.chunk_size:
                self._flush_type(call_type)
            if max_examples is not None and self._seen[call_type] >= max_examples:
                break

    def close(self) -> dict[str, int]:
        for call_type in CALL_TYPE_ORDER:
            self._flush_type(call_type)
        counts = {t: self._rows_written[t] for t in CALL_TYPE_ORDER if t in self._opened}
        files = {
            t: {
                "feat_bits": f"{t}_feat_bits.dat",
                "feat_cont": f"{t}_feat_cont.dat",
                "meta": f"{t}_meta.dat",
                "shape_feat_bits": [counts[t], self._codec.feat_bits_bytes],
                "shape_feat_cont": [counts[t], len(self.cont_cols)],
            }
            for t in counts
        }
        manifest = {
            "schema_version": CALL_SCHEMA_VERSION,
            "featurizer_version": FEATURIZER_VERSION,
            "action_space_version": ACTION_SPACE_VERSION,
            "feature_dim": FEATURIZER_OUTPUT_DIM,
            "features_packed": True,
            "continuous_feature_columns": self.cont_cols,
            "type_order": [t for t in CALL_TYPE_ORDER if t in counts],
            "meta_dtype": [(name, str(t)) for name, t in META_DTYPE.descr],
            "counts": counts,
            "files": files,
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        log.info(
            "materialised calls bundle to %s in %.1fs: %s",
            self.out_dir, time.perf_counter() - self._t0, counts,
        )
        return counts


def materialise_calls(
    streams: Mapping[str, Iterable[CallExample]],
    out_dir: str | Path,
    *,
    max_examples: int | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> dict[str, int]:
    """Pull-style wrapper over `CallBundleWriter`. `streams` maps a call type
    to its `CallExample` stream; absent types are not written. `max_examples`
    caps each type independently. Idempotent. Returns `{call_type: rows}`."""
    writer = CallBundleWriter(out_dir, chunk_size=chunk_size)
    for call_type in CALL_TYPE_ORDER:
        stream = streams.get(call_type)
        if stream is None:
            continue
        writer.add_many(call_type, stream, max_examples=max_examples)
    return writer.close()


class MemmapCallDataset(Iterable[CallExample]):
    """Disk-backed Calls dataset for a single call type. Opens the packed
    memmaps written by `materialise_calls()` and yields `CallExample`s
    (`__iter__`, drop-in for `ParquetCallDataset`) or pre-stacked batches
    (`iter_batches`)."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        call_type: str,
        expected_featurizer_version: str = FEATURIZER_VERSION,
        expected_action_space_version: str = ACTION_SPACE_VERSION,
        expected_schema_version: int = CALL_SCHEMA_VERSION,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.call_type = call_type
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no manifest.json under {self.data_dir}; "
                "did you run materialise_calls()?"
            )
        self.manifest: dict = json.loads(manifest_path.read_text())

        check_pin(
            "featurizer_version",
            self.manifest.get("featurizer_version"),
            expected_featurizer_version,
        )
        check_pin(
            "action_space_version",
            self.manifest.get("action_space_version"),
            expected_action_space_version,
        )
        check_pin(
            "schema_version",
            self.manifest.get("schema_version"),
            expected_schema_version,
        )

        if call_type not in self.manifest["counts"]:
            raise KeyError(
                f"call_type {call_type!r} not in bundle {self.data_dir} "
                f"(has {sorted(self.manifest['counts'])})"
            )

        self.feature_dim: int = self.manifest["feature_dim"]
        self.count: int = self.manifest["counts"][call_type]

        cont_cols = list(self.manifest["continuous_feature_columns"])
        self._codec = FeatureCodec(self.feature_dim, cont_cols)

        self._meta_dtype = np.dtype([
            (n, np.dtype(t)) for n, t in self.manifest["meta_dtype"]
        ])

        info = self.manifest["files"][call_type]
        self._feat_bits = np.memmap(
            self.data_dir / info["feat_bits"], dtype=np.uint8, mode="r",
            shape=(self.count, self._codec.feat_bits_bytes),
        )
        self._feat_cont = np.memmap(
            self.data_dir / info["feat_cont"], dtype=np.float32, mode="r",
            shape=(self.count, len(cont_cols)),
        )
        self._meta = np.memmap(
            self.data_dir / info["meta"], dtype=self._meta_dtype, mode="r",
            shape=(self.count,),
        )

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[CallExample]:
        for i in range(self.count):
            m = self._meta[i]
            features = self._codec.unpack(
                self._feat_bits[i][None, :], self._feat_cont[i][None, :],
            )[0]
            yield CallExample(
                features=features,
                target=int(m["target"]),
                skill_decile=int(m["skill_decile"]),
                sample_weight=float(m["sample_weight"]),
                game_id=(str(int(m["game_id"])) if int(m["game_id"]) else ""),
                round_id=int(m["round_id"]),
            )

    def iter_batches(
        self, batch_size: int, *, drop_last: bool = False,
        shuffle: bool = False, seed: int = 0,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield pre-stacked batches: `features` (B,D) f32, `target` (B,) i64,
        `skill_decile` (B,) i64, `sample_weight` (B,) f32, `game_id` (B,) u4.

        `shuffle=True` walks a full random permutation of the rows (seeded by
        `seed`). The calls slice is single-digit GB, so a global permutation is
        cheap (the index array is ~8 B/row) and random memmap access stays in
        the OS page cache — no block-shuffle machinery needed (unlike the
        280 GB BC bundle). `game_id` is included so the trainer can carve a
        game-level val holdout from the stream without materialising."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        order = (
            np.random.default_rng(seed).permutation(self.count)
            if shuffle else None
        )
        start = 0
        while start < self.count:
            stop = min(start + batch_size, self.count)
            if stop - start < batch_size and drop_last:
                break
            idx = order[start:stop] if order is not None else np.arange(start, stop)
            feat = self._codec.unpack(
                np.asarray(self._feat_bits[idx]), np.asarray(self._feat_cont[idx]),
            )
            meta = np.asarray(self._meta[idx])
            yield {
                "features": feat,
                "target": meta["target"].astype(np.int64),
                "skill_decile": meta["skill_decile"].astype(np.int64),
                "sample_weight": meta["sample_weight"].astype(np.float32),
                "game_id": meta["game_id"].astype(np.uint32),
            }
            start = stop
