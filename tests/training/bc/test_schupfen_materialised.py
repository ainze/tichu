"""Roundtrip + version-pin tests for the materialised Schupfen bundle.

Mirrors `test_materialised.py` (the BC bundle) but for the standalone
Schupfen Network's example shape: features + a 56-slot hand_mask + a
3-vector slot target (to_next / to_partner / to_previous). See
[ADR-0020](../../../docs/adr/0020-one-parse-pass-many-task-bundles.md) for
the per-trainer bundle topology and [ADR-0019](../../../docs/adr/0019-bit-pack-materialised-bundle.md)
for the bit-packing this reuses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tichu_training.bc.schupfen_materialised import (
    MemmapSchupfenDataset,
    SCHUPFEN_SCHEMA_VERSION,
    materialise_schupfen,
)
from tichu_training.bc.schupfen_training import (
    SchupfenExample,
    SyntheticSchupfenDataset,
)
from tichu_training.checkpoint import VersionMismatchError
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM


@pytest.fixture
def materialised_smoke(tmp_path: Path) -> tuple[list[SchupfenExample], Path]:
    """Materialise a binary-feature SyntheticSchupfenDataset slice to disk.

    `binary_features=True` is required because the writer bit-packs the
    indicator feature columns and guards against non-0/1 values — the
    default all-Gaussian features would (correctly) trip that guard.
    """
    src = SyntheticSchupfenDataset(seed=42, n_examples=30, binary_features=True)
    examples = list(src)
    out_dir = tmp_path / "schupfen_bundle"
    materialise_schupfen(iter(examples), out_dir, max_examples=len(examples))
    return examples, out_dir


def test_roundtrip_reads_back_examples_exactly(materialised_smoke):
    """The tracer bullet: every example round-trips through the packed
    bundle byte-for-byte — features, hand_mask, the 3-vector target, skill,
    and sample_weight all come back equal to what was written."""
    examples, out_dir = materialised_smoke
    read = list(MemmapSchupfenDataset(out_dir))
    assert len(read) == len(examples)
    for got, want in zip(read, examples):
        np.testing.assert_array_equal(got.features, want.features)
        np.testing.assert_array_equal(got.hand_mask, want.hand_mask)
        np.testing.assert_array_equal(got.target, want.target)
        assert got.skill_decile == want.skill_decile
        assert got.sample_weight == pytest.approx(want.sample_weight)


def test_features_and_hand_mask_are_bit_packed_on_disk(materialised_smoke):
    """Locks the packed format against a silent revert to dense storage.
    Indicator cols pack to ceil(214/8)=27 B; the 10 continuous cols stay
    f32 (40 B); the 56-slot hand mask packs to ceil(56/8)=7 B per row."""
    examples, out_dir = materialised_smoke
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["features_packed"] is True
    n = len(examples)
    n_cont = len(manifest["continuous_feature_columns"])
    feat_bits_bytes = (FEATURIZER_OUTPUT_DIM - n_cont + 7) // 8
    assert (out_dir / "schupfen_feat_bits.dat").stat().st_size == n * feat_bits_bytes
    assert (out_dir / "schupfen_feat_cont.dat").stat().st_size == n * n_cont * 4
    assert (out_dir / "schupfen_hand_mask.dat").stat().st_size == n * 7


@pytest.mark.parametrize(
    "pin,bad",
    [
        ("expected_featurizer_version", "vBOGUS"),
        ("expected_action_space_version", "vBOGUS"),
        ("expected_schema_version", 999),
    ],
)
def test_version_pin_mismatch_raises(materialised_smoke, pin, bad):
    """A stale bundle must fail loudly at load, not silently corrupt a run."""
    _, out_dir = materialised_smoke
    with pytest.raises(VersionMismatchError):
        MemmapSchupfenDataset(out_dir, **{pin: bad})


def test_iter_batches_matches_iter(materialised_smoke):
    """The fast batched path yields the same rows as the per-example path."""
    _, out_dir = materialised_smoke
    ds = MemmapSchupfenDataset(out_dir)
    rows = list(ds)
    batched = list(ds.iter_batches(batch_size=8))
    feat = np.concatenate([b["features"] for b in batched])
    mask = np.concatenate([b["hand_mask"] for b in batched])
    target = np.concatenate([b["target"] for b in batched])
    assert feat.shape[0] == len(rows)
    for i, ex in enumerate(rows):
        np.testing.assert_array_equal(feat[i], ex.features)
        np.testing.assert_array_equal(mask[i], ex.hand_mask)
        np.testing.assert_array_equal(target[i], ex.target)


def test_writer_guard_rejects_non_binary_features(tmp_path):
    """Default (Gaussian) synthetic features violate the 0/1 indicator
    contract and must trip the writer guard rather than silently corrupt
    the packed bits (packbits collapses any non-zero to 1)."""
    src = list(SyntheticSchupfenDataset(seed=1, n_examples=5))
    with pytest.raises(ValueError, match="non-0/1"):
        materialise_schupfen(iter(src), tmp_path / "bad")
