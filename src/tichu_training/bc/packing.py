"""Shared bit-packing primitives for the materialised task bundles.

Every per-trainer bundle ([ADR-0020](../../../docs/adr/0020-one-parse-pass-many-task-bundles.md))
stores the same 224-dim `featurize()` output and packs its 0/1 columns
the same way ([ADR-0019](../../../docs/adr/0019-bit-pack-materialised-bundle.md)).
This module is the single home for that machinery, so the BC, schupfen,
calls, and belief writers/readers share one implementation instead of
copying it:

* `FeatureCodec` — splits a dense feature batch into the bit-packed
  indicator block + the float32 continuous block, and reconstructs it.
  A deep module: the column split, the 0/1 guard, and the contiguous-run
  scatter all live behind `pack` / `unpack`.
* `pack_bool_rows` / `unpack_bool_rows` — pack any per-row boolean array
  (legal masks, schupfen hand masks, belief label / mask grids) to
  `ceil(dim/8)` bytes and back.
* `check_pin` — the version-pin guard shared by every reader.
* Leaf helpers (`packed_mask_bytes`, `binary_feature_columns`,
  `contiguous_runs`) kept public for callers that need the pieces.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from tichu_training.checkpoint import VersionMismatchError


def packed_mask_bytes(mask_dim: int) -> int:
    """Bytes per row after `np.packbits` of a `mask_dim`-bit boolean row."""
    return (mask_dim + 7) // 8


def binary_feature_columns(
    feature_dim: int, continuous_columns: Iterable[int],
) -> list[int]:
    """Ascending indicator-column indices = all columns minus the continuous
    ones. The writer packs these (in this order) and the reader unpacks back
    into the same slots, so both sides must derive them identically."""
    cont = set(continuous_columns)
    return [c for c in range(feature_dim) if c not in cont]


def contiguous_runs(cols: list[int]) -> list[tuple[int, int, int]]:
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


def check_pin(name: str, found, expected) -> None:
    """Raise VersionMismatchError if a bundle manifest's pin disagrees with
    the live code. Keeps the bundle/code coupling explicit — the cost of a
    quietly-stale bundle would be a silently-corrupt training run."""
    if found != expected:
        raise VersionMismatchError(
            f"materialised bundle {name} mismatch: bundle={found!r} "
            f"live={expected!r}; re-run the materialise pass against current code"
        )


def pack_bool_rows(rows: np.ndarray) -> np.ndarray:
    """Pack a `(N, dim)` boolean/0-1 array to `(N, ceil(dim/8))` uint8 via
    `np.packbits` (per-row, axis=1 — so a chunked write is byte-identical to
    a single-shot one)."""
    return np.packbits(np.asarray(rows, dtype=np.uint8), axis=1)


def unpack_bool_rows(packed: np.ndarray, dim: int) -> np.ndarray:
    """Inverse of `pack_bool_rows`: `(N, ceil(dim/8))` uint8 → `(N, dim)`
    uint8 0/1, slicing off the byte-boundary padding. Callers cast to bool
    or float as their contract requires."""
    return np.unpackbits(np.asarray(packed), axis=1)[:, :dim]


class FeatureCodec:
    """Splits a dense `(N, D)` float32 feature batch into a bit-packed
    indicator block + a float32 continuous block, and reconstructs it.

    The 0/1-vs-continuous partition is a fact about `featurize()` output
    (ADR-0019): the ~214 indicator columns pack to 1 bit each; the ~10
    continuous ratio columns must stay float32. Both directions act per row
    (axis=1), so a chunked materialise is byte-identical to a single-shot one.
    """

    def __init__(self, feature_dim: int, continuous_columns: Iterable[int]) -> None:
        self.feature_dim = feature_dim
        self.cont_cols = list(continuous_columns)
        self.bin_cols = binary_feature_columns(feature_dim, self.cont_cols)
        self._bin_idx = np.asarray(self.bin_cols, dtype=np.intp)
        self._cont_idx = np.asarray(self.cont_cols, dtype=np.intp)
        self.feat_bits_bytes = (len(self.bin_cols) + 7) // 8
        self._bin_runs = contiguous_runs(self.bin_cols)
        self._cont_runs = contiguous_runs(self.cont_cols)

    def pack(self, feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`(N, D)` f32 → (`feat_bits` `(N, feat_bits_bytes)` uint8,
        `feat_cont` `(N, n_cont)` f32). Guards against non-0/1 indicator
        columns — packbits collapses any non-zero to 1, so a featurizer change
        that made an "indicator" section continuous (or a non-featurizer
        stream) would silently wreck those bits."""
        feat = np.asarray(feat, dtype=np.float32)
        bin_block = feat[:, self._bin_idx]
        if ((bin_block != 0.0) & (bin_block != 1.0)).any():
            bad = np.unique(np.asarray(self._bin_idx)[
                np.where(((bin_block != 0.0) & (bin_block != 1.0)).any(axis=0))[0]
            ])
            raise ValueError(
                f"FeatureCodec.pack found non-0/1 values in indicator "
                f"column(s) {bad.tolist()[:10]}. The stream must be real "
                f"featurizer output; if the featurizer layout changed, update "
                f"featurizer.CONTINUOUS_SECTIONS."
            )
        feat_bits = np.packbits(bin_block.astype(np.uint8), axis=1)
        feat_cont = np.ascontiguousarray(feat[:, self._cont_idx], dtype=np.float32)
        return feat_bits, feat_cont

    def unpack(self, bits: np.ndarray, cont: np.ndarray) -> np.ndarray:
        """Inverse of `pack`: rebuild the dense `(N, D)` float32 batch by
        scattering each block back into its original columns over the
        precomputed contiguous runs (a few slice copies, not a fancy-index)."""
        feat = np.empty((bits.shape[0], self.feature_dim), dtype=np.float32)
        binvals = np.unpackbits(bits, axis=1)
        for dst0, dst1, src0 in self._bin_runs:
            feat[:, dst0:dst1] = binvals[:, src0 : src0 + (dst1 - dst0)]
        for dst0, dst1, src0 in self._cont_runs:
            feat[:, dst0:dst1] = cont[:, src0 : src0 + (dst1 - dst0)]
        return feat
