"""Disk-backed Schupfen dataset: writer + reader for the materialised
Schupfen bundle.

The standalone Schupfen Network ([ADR-0012](../../../docs/adr/0012-schupfen-is-a-standalone-network.md))
is parse-bound, not replay-bound — it featurises a synthetic pre-schupfen
`GameState` from `start_hands`. Materialising its example stream once, to
packed numpy memmaps, lets `train_schupfen` skip re-parsing the archive.
This is the schupfen slice of the "one parse pass, many task bundles"
pipeline ([ADR-0020](../../../docs/adr/0020-one-parse-pass-many-task-bundles.md)),
reusing the bit-packing machinery of the BC bundle
([ADR-0019](../../../docs/adr/0019-bit-pack-materialised-bundle.md)).

On-disk layout for a slice of N examples:

  <out_dir>/
    manifest.json
    schupfen_feat_bits.dat   (N, 27)   uint8   (packbits of 214 indicator cols)
    schupfen_feat_cont.dat   (N, 10)   float32 (continuous columns)
    schupfen_hand_mask.dat   (N, 7)    uint8   (packbits of the 56-slot hand mask)
    schupfen_meta.dat        (N,)      structured (META_DTYPE)

The 3-vector target (to_next / to_partner / to_previous slot ids, each
0..55) is stored as three uint8 fields in meta — no separate target file.

Version pins: `materialise_schupfen()` records FEATURIZER_VERSION,
ACTION_SPACE_VERSION, and SCHUPFEN_SCHEMA_VERSION into the manifest.
`MemmapSchupfenDataset.__init__` refuses to load on mismatch with the live
code, raising `VersionMismatchError` (same class as the BC bundle / the
`Checkpoint` container, so existing handlers catch all three).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.packing import (
    FeatureCodec,
    check_pin,
    pack_bool_rows,
    packed_mask_bytes,
    unpack_bool_rows,
)
from tichu_training.bc.schupfen_training import SchupfenExample
from tichu_training.card_slots import CARD_SLOTS
from tichu_training.featurizer import (
    CONTINUOUS_FEATURE_COLUMNS,
    FEATURIZER_OUTPUT_DIM,
    FEATURIZER_VERSION,
)


log = logging.getLogger("tichu_training.bc.schupfen_materialised")


# Bumped on any non-backwards-compatible change to the on-disk layout, meta
# dtype, or manifest schema. Independent of FEATURIZER_VERSION /
# ACTION_SPACE_VERSION. Tracked separately from the BC bundle's schema
# version — the per-trainer bundles have independent lifecycles (ADR-0020).
SCHUPFEN_SCHEMA_VERSION = 1


# Per-row metadata. The 3-vector slot target is three uint8s (each 0..55);
# skill_decile is uint8 (0..10); sample_weight is float32; game_id (u4) +
# round_id (u1) are provenance (ADR-0020). 13 bytes/row.
META_DTYPE: np.dtype = np.dtype([
    ("target_to_next", "u1"),
    ("target_to_partner", "u1"),
    ("target_to_previous", "u1"),
    ("sample_weight", "f4"),
    ("skill_decile", "u1"),
    ("game_id", "u4"),
    ("round_id", "u1"),
])


_DEFAULT_CHUNK_SIZE: int = 25_000


class SchupfenBundleWriter:
    """Push-style writer for the schupfen bundle. `add_many()` buffers
    SchupfenExamples and flushes on the chunk cadence; `close()` drains and
    writes `manifest.json`. Driven row-by-row by the consolidated parse pass
    (ADR-0020) alongside the other task writers."""

    def __init__(
        self, out_dir: str | Path, *, chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.cont_cols = list(CONTINUOUS_FEATURE_COLUMNS)
        self._codec = FeatureCodec(FEATURIZER_OUTPUT_DIM, self.cont_cols)
        self.feat_bits_bytes = self._codec.feat_bits_bytes
        self.hand_mask_bytes = packed_mask_bytes(CARD_SLOTS)

        self._bits_path = self.out_dir / "schupfen_feat_bits.dat"
        self._cont_path = self.out_dir / "schupfen_feat_cont.dat"
        self._mask_path = self.out_dir / "schupfen_hand_mask.dat"
        self._meta_path = self.out_dir / "schupfen_meta.dat"
        for p in (self._bits_path, self._cont_path, self._mask_path, self._meta_path):
            with p.open("wb"):
                pass  # truncate

        self._buf: list[SchupfenExample] = []
        self._rows_written = 0
        self._seen = 0
        self._t0 = time.perf_counter()

    def _flush(self) -> None:
        if not self._buf:
            return
        feat = np.stack([e.features for e in self._buf]).astype(np.float32, copy=False)
        feat_bits, feat_cont = self._codec.pack(feat)
        hand_mask = pack_bool_rows(np.stack([e.hand_mask for e in self._buf]))
        meta = np.zeros(len(self._buf), dtype=META_DTYPE)
        for i, e in enumerate(self._buf):
            meta[i]["target_to_next"] = int(e.target[0])
            meta[i]["target_to_partner"] = int(e.target[1])
            meta[i]["target_to_previous"] = int(e.target[2])
            meta[i]["sample_weight"] = e.sample_weight
            meta[i]["skill_decile"] = e.skill_decile
            meta[i]["game_id"] = e.game_id
            meta[i]["round_id"] = e.round_id
        with self._bits_path.open("ab") as fh:
            feat_bits.tofile(fh)
        with self._cont_path.open("ab") as fh:
            feat_cont.tofile(fh)
        with self._mask_path.open("ab") as fh:
            hand_mask.tofile(fh)
        with self._meta_path.open("ab") as fh:
            meta.tofile(fh)
        self._rows_written += len(self._buf)
        self._buf.clear()
        del feat, feat_bits, feat_cont, hand_mask, meta

    def add_many(
        self, stream: Iterable[SchupfenExample], *, max_examples: int | None = None,
    ) -> None:
        for ex in stream:
            self._buf.append(ex)
            self._seen += 1
            if len(self._buf) >= self.chunk_size:
                self._flush()
            if max_examples is not None and self._seen >= max_examples:
                break

    def close(self) -> int:
        self._flush()
        manifest = {
            "schema_version": SCHUPFEN_SCHEMA_VERSION,
            "featurizer_version": FEATURIZER_VERSION,
            "action_space_version": ACTION_SPACE_VERSION,
            "feature_dim": FEATURIZER_OUTPUT_DIM,
            "features_packed": True,
            "continuous_feature_columns": self.cont_cols,
            "hand_mask_dim": CARD_SLOTS,
            "meta_dtype": [(name, str(t)) for name, t in META_DTYPE.descr],
            "count": self._rows_written,
            "files": {
                "feat_bits": "schupfen_feat_bits.dat",
                "feat_cont": "schupfen_feat_cont.dat",
                "hand_mask": "schupfen_hand_mask.dat",
                "meta": "schupfen_meta.dat",
                "shape_feat_bits": [self._rows_written, self.feat_bits_bytes],
                "shape_feat_cont": [self._rows_written, len(self.cont_cols)],
                "shape_hand_mask": [self._rows_written, self.hand_mask_bytes],
            },
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        log.info(
            "materialised %d schupfen examples to %s in %.1fs",
            self._rows_written, self.out_dir, time.perf_counter() - self._t0,
        )
        return self._rows_written


def materialise_schupfen(
    stream: Iterable[SchupfenExample],
    out_dir: str | Path,
    *,
    max_examples: int | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> int:
    """Pull-style wrapper over `SchupfenBundleWriter`. Returns rows written.
    Idempotent (truncates the prior bundle at `out_dir`)."""
    writer = SchupfenBundleWriter(out_dir, chunk_size=chunk_size)
    writer.add_many(stream, max_examples=max_examples)
    return writer.close()


class MemmapSchupfenDataset(Iterable[SchupfenExample]):
    """Disk-backed Schupfen dataset. Opens the packed memmaps written by
    `materialise_schupfen()` and yields `SchupfenExample`s (`__iter__`,
    drop-in for `ParquetSchupfenDataset`) or pre-stacked batches
    (`iter_batches`, the fast path)."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        expected_featurizer_version: str = FEATURIZER_VERSION,
        expected_action_space_version: str = ACTION_SPACE_VERSION,
        expected_schema_version: int = SCHUPFEN_SCHEMA_VERSION,
    ) -> None:
        self.data_dir = Path(data_dir)
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no manifest.json under {self.data_dir}; "
                "did you run materialise_schupfen()?"
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

        self.feature_dim: int = self.manifest["feature_dim"]
        self.count: int = self.manifest["count"]
        self.hand_mask_dim: int = self.manifest["hand_mask_dim"]

        cont_cols = list(self.manifest["continuous_feature_columns"])
        self._codec = FeatureCodec(self.feature_dim, cont_cols)
        self._feat_bits_bytes = self._codec.feat_bits_bytes
        self._hand_mask_bytes = packed_mask_bytes(self.hand_mask_dim)

        self._meta_dtype = np.dtype([
            (n, np.dtype(t)) for n, t in self.manifest["meta_dtype"]
        ])

        info = self.manifest["files"]
        self._feat_bits = np.memmap(
            self.data_dir / info["feat_bits"], dtype=np.uint8, mode="r",
            shape=(self.count, self._feat_bits_bytes),
        )
        self._feat_cont = np.memmap(
            self.data_dir / info["feat_cont"], dtype=np.float32, mode="r",
            shape=(self.count, len(cont_cols)),
        )
        self._hand_mask = np.memmap(
            self.data_dir / info["hand_mask"], dtype=np.uint8, mode="r",
            shape=(self.count, self._hand_mask_bytes),
        )
        self._meta = np.memmap(
            self.data_dir / info["meta"], dtype=self._meta_dtype, mode="r",
            shape=(self.count,),
        )

    def __len__(self) -> int:
        return self.count

    def _unpack_hand_mask(self, packed: np.ndarray) -> np.ndarray:
        """Unpack (B, hand_mask_bytes) uint8 → (B, hand_mask_dim) float32,
        matching `SchupfenExample.hand_mask`'s 1.0/0.0 contract."""
        return unpack_bool_rows(packed, self.hand_mask_dim).astype(np.float32)

    def __iter__(self) -> Iterator[SchupfenExample]:
        for i in range(self.count):
            m = self._meta[i]
            features = self._codec.unpack(
                self._feat_bits[i][None, :], self._feat_cont[i][None, :],
            )[0]
            hand_mask = self._unpack_hand_mask(self._hand_mask[i][None, :])[0]
            target = np.array(
                [m["target_to_next"], m["target_to_partner"],
                 m["target_to_previous"]],
                dtype=np.int64,
            )
            yield SchupfenExample(
                features=features,
                hand_mask=hand_mask,
                target=target,
                skill_decile=int(m["skill_decile"]),
                sample_weight=float(m["sample_weight"]),
                game_id=int(m["game_id"]),
                round_id=int(m["round_id"]),
            )

    def iter_batches(
        self, batch_size: int, *, drop_last: bool = False,
        shuffle: bool = False, seed: int = 0,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield pre-stacked batches keyed for the schupfen training loop:
        `features` (B,D) f32, `hand_mask` (B,56) f32, `target` (B,3) i64,
        `skill_decile` (B,) i64, `sample_weight` (B,) f32.

        `shuffle=True` walks a full random permutation of the rows (seeded by
        `seed`). The schupfen slice is single-digit GB, so a global permutation
        is cheap and random memmap access stays in page cache — no block-shuffle
        machinery needed (unlike the 280 GB BC bundle)."""
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
            yield self._gather_batch(idx)
            start = stop

    def _gather_batch(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        feat = self._codec.unpack(
            np.asarray(self._feat_bits[idx]), np.asarray(self._feat_cont[idx]),
        )
        hand_mask = self._unpack_hand_mask(np.asarray(self._hand_mask[idx]))
        meta = np.asarray(self._meta[idx])
        target = np.stack(
            [meta["target_to_next"], meta["target_to_partner"],
             meta["target_to_previous"]],
            axis=1,
        ).astype(np.int64)
        return {
            "features": feat,
            "hand_mask": hand_mask,
            "target": target,
            "skill_decile": meta["skill_decile"].astype(np.int64),
            "sample_weight": meta["sample_weight"].astype(np.float32),
            "game_id": meta["game_id"].astype(np.uint32),
        }
