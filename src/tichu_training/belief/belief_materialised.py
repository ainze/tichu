"""Disk-backed Belief dataset: writer + reader for the materialised Belief
bundle (ADR-0021).

Belief reuses the same `featurize()` vector as the other tasks (see
`input_spec.py`), so it shares the `bc.packing` machinery. Its labels are a
`(3 opponents, 56 cards)` multi-hot grid (opponents in relative-seat order
next/partner/previous); its mask is a `(56,)` card-level "unknown" vector
(every unplayed non-own card is in *some* opponent's hand) stored once and
broadcast to `(3,56)` on read.

On-disk layout for a slice of N examples (byte widths shown at v6: 591 feature
columns, 37 of them continuous):

  <out_dir>/
    manifest.json
    belief_feat_bits.dat   (N, 70)   uint8   (packbits of the 554 indicator cols)
    belief_feat_cont.dat   (N, 37)   float32 (the policy's continuous cols)
    belief_labels.dat      (N, 21)   uint8   (packbits of the 3*56=168 label bits)
    belief_mask.dat        (N, 7)    uint8   (packbits of the 56-card mask)
    belief_meta.dat        (N,)      structured (cards_played u1, game_id u4, round_id u1)

The exact split is self-describing: the writer records `feature_dim` and
`continuous_feature_columns` into the manifest and the reader rebuilds the
dense vector from them.

`cards_played` (= 56 - sum(hand_sizes)) lets the belief trainer filter out the
low-information opening of each round at read time, no re-materialise needed.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.packing import FeatureCodec, check_pin, pack_bool_rows, unpack_bool_rows
from tichu_training.belief.dataset import BeliefExample
from tichu_training.belief.input_spec import (
    BELIEF_FEATURE_DIM,
    BELIEF_INPUT_VERSION,
    belief_continuous_columns,
)
from tichu_training.featurizer import FEATURIZER_VERSION


log = logging.getLogger("tichu_training.belief.belief_materialised")


# v2: the belief bundle's on-disk *layout* — packed feature bits + continuous
# floats, packed labels, a stored card-level mask, and the meta record. What
# goes into the feature columns is versioned separately by
# `belief_input_version` (see input_spec.py), so a belief-input change does not
# need a layout bump.
BELIEF_SCHEMA_VERSION = 2

_NUM_OPPONENTS = 3
_NUM_CARDS = 56


# cards_played (u1, 0..56); game_id (u4) + round_id (u1) provenance. 6 B/row.
META_DTYPE: np.dtype = np.dtype([
    ("cards_played", "u1"),
    ("game_id", "u4"),
    ("round_id", "u1"),
])


_DEFAULT_CHUNK_SIZE: int = 25_000


class BeliefBundleWriter:
    """Push-style writer for the belief bundle. `add_many()` buffers
    BeliefExamples and flushes on the chunk cadence; `close()` writes the
    manifest. The `(3,56)` label grid packs to 21 B/row; the mask is stored
    once as the `(56,)` card-level vector (row 0 of the broadcast `(3,56)`)."""

    def __init__(
        self, out_dir: str | Path, *, chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.cont_cols = list(belief_continuous_columns())
        self._codec = FeatureCodec(BELIEF_FEATURE_DIM, self.cont_cols)
        self._label_dim = _NUM_OPPONENTS * _NUM_CARDS

        self._bits_path = self.out_dir / "belief_feat_bits.dat"
        self._cont_path = self.out_dir / "belief_feat_cont.dat"
        self._labels_path = self.out_dir / "belief_labels.dat"
        self._mask_path = self.out_dir / "belief_mask.dat"
        self._meta_path = self.out_dir / "belief_meta.dat"
        for p in (self._bits_path, self._cont_path, self._labels_path,
                  self._mask_path, self._meta_path):
            with p.open("wb"):
                pass  # truncate

        self._buf: list[BeliefExample] = []
        self._rows_written = 0
        self._seen = 0
        self._t0 = time.perf_counter()

    def _flush(self) -> None:
        if not self._buf:
            return
        feat = np.stack([e.features for e in self._buf]).astype(np.float32, copy=False)
        feat_bits, feat_cont = self._codec.pack(feat)
        # Labels (3,56) -> flat (168,) -> packbits. Mask: the card-level (56,)
        # vector (row 0; all opponent rows are identical by construction).
        labels = pack_bool_rows(
            np.stack([e.labels.reshape(-1) for e in self._buf])
        )
        mask = pack_bool_rows(np.stack([np.asarray(e.mask)[0] for e in self._buf]))
        meta = np.zeros(len(self._buf), dtype=META_DTYPE)
        for i, e in enumerate(self._buf):
            meta[i]["cards_played"] = e.cards_played
            meta[i]["game_id"] = e.game_id
            meta[i]["round_id"] = e.round_id
        with self._bits_path.open("ab") as fh:
            feat_bits.tofile(fh)
        with self._cont_path.open("ab") as fh:
            feat_cont.tofile(fh)
        with self._labels_path.open("ab") as fh:
            labels.tofile(fh)
        with self._mask_path.open("ab") as fh:
            mask.tofile(fh)
        with self._meta_path.open("ab") as fh:
            meta.tofile(fh)
        self._rows_written += len(self._buf)
        self._buf.clear()
        del feat, feat_bits, feat_cont, labels, mask, meta

    def add_many(
        self, stream: Iterable[BeliefExample], *, max_examples: int | None = None,
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
            "schema_version": BELIEF_SCHEMA_VERSION,
            "featurizer_version": FEATURIZER_VERSION,
            "belief_input_version": BELIEF_INPUT_VERSION,
            "action_space_version": ACTION_SPACE_VERSION,
            "feature_dim": BELIEF_FEATURE_DIM,
            "features_packed": True,
            "continuous_feature_columns": self.cont_cols,
            "num_opponents": _NUM_OPPONENTS,
            "num_cards": _NUM_CARDS,
            "meta_dtype": [(name, str(t)) for name, t in META_DTYPE.descr],
            "count": self._rows_written,
            "files": {
                "feat_bits": "belief_feat_bits.dat",
                "feat_cont": "belief_feat_cont.dat",
                "labels": "belief_labels.dat",
                "mask": "belief_mask.dat",
                "meta": "belief_meta.dat",
            },
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        log.info(
            "materialised %d belief examples to %s in %.1fs",
            self._rows_written, self.out_dir, time.perf_counter() - self._t0,
        )
        return self._rows_written


def materialise_belief(
    stream: Iterable[BeliefExample],
    out_dir: str | Path,
    *,
    max_examples: int | None = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> int:
    """Pull-style wrapper over `BeliefBundleWriter`. Returns rows written."""
    writer = BeliefBundleWriter(out_dir, chunk_size=chunk_size)
    writer.add_many(stream, max_examples=max_examples)
    return writer.close()


class MemmapBeliefDataset(Iterable[BeliefExample]):
    """Disk-backed Belief dataset. Yields `BeliefExample`s with the dense
    `(3,56)` labels and the `(3,56)` mask (broadcast from the stored `(56,)`
    card vector), drop-in for `SyntheticBeliefDataset`."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        expected_featurizer_version: str = FEATURIZER_VERSION,
        expected_action_space_version: str = ACTION_SPACE_VERSION,
        expected_schema_version: int = BELIEF_SCHEMA_VERSION,
        expected_belief_input_version: str = BELIEF_INPUT_VERSION,
    ) -> None:
        self.data_dir = Path(data_dir)
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no manifest.json under {self.data_dir}; "
                "did you run materialise_belief()?"
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
        check_pin(
            "belief_input_version",
            self.manifest.get("belief_input_version"),
            expected_belief_input_version,
        )

        self.feature_dim: int = self.manifest["feature_dim"]
        self.count: int = self.manifest["count"]
        self.num_opponents: int = self.manifest["num_opponents"]
        self.num_cards: int = self.manifest["num_cards"]
        self._label_dim = self.num_opponents * self.num_cards

        cont_cols = list(self.manifest["continuous_feature_columns"])
        self._codec = FeatureCodec(self.feature_dim, cont_cols)
        self._meta_dtype = np.dtype([
            (n, np.dtype(t)) for n, t in self.manifest["meta_dtype"]
        ])

        info = self.manifest["files"]
        self._feat_bits = np.memmap(
            self.data_dir / info["feat_bits"], dtype=np.uint8, mode="r",
            shape=(self.count, self._codec.feat_bits_bytes),
        )
        self._feat_cont = np.memmap(
            self.data_dir / info["feat_cont"], dtype=np.float32, mode="r",
            shape=(self.count, len(cont_cols)),
        )
        self._labels = np.memmap(
            self.data_dir / info["labels"], dtype=np.uint8, mode="r",
            shape=(self.count, (self._label_dim + 7) // 8),
        )
        self._mask = np.memmap(
            self.data_dir / info["mask"], dtype=np.uint8, mode="r",
            shape=(self.count, (self.num_cards + 7) // 8),
        )
        self._meta = np.memmap(
            self.data_dir / info["meta"], dtype=self._meta_dtype, mode="r",
            shape=(self.count,),
        )

    def __len__(self) -> int:
        return self.count

    def iter_arrays(self, batch_size: int):
        """Stream batched numpy arrays straight off the memmaps — the low-RAM
        path for scale training (no full-list materialisation). Yields
        ``(features (B, feature_dim) f32, labels (B, 3, 56) f32,
        card_mask (B, 56) bool)`` in stored order. Vectorised unpack (one bulk
        bit-unpack per batch), far faster than per-example `__iter__`."""
        for start in range(0, self.count, batch_size):
            end = min(start + batch_size, self.count)
            feats = self._codec.unpack(
                np.asarray(self._feat_bits[start:end]),
                np.asarray(self._feat_cont[start:end]),
            )
            labels = unpack_bool_rows(
                np.asarray(self._labels[start:end]), self._label_dim
            ).reshape(end - start, self.num_opponents, self.num_cards).astype(np.float32)
            card_mask = unpack_bool_rows(
                np.asarray(self._mask[start:end]), self.num_cards
            ).astype(bool)
            yield feats, labels, card_mask

    def __iter__(self) -> Iterator[BeliefExample]:
        for i in range(self.count):
            m = self._meta[i]
            features = self._codec.unpack(
                self._feat_bits[i][None, :], self._feat_cont[i][None, :],
            )[0]
            labels = unpack_bool_rows(self._labels[i][None, :], self._label_dim)[0]
            labels = labels.reshape(self.num_opponents, self.num_cards).astype(np.float32)
            card_mask = unpack_bool_rows(self._mask[i][None, :], self.num_cards)[0].astype(bool)
            mask = np.broadcast_to(card_mask, (self.num_opponents, self.num_cards)).copy()
            yield BeliefExample(
                features=features,
                labels=labels,
                mask=mask,
                cards_played=int(m["cards_played"]),
                game_id=int(m["game_id"]),
                round_id=int(m["round_id"]),
            )
