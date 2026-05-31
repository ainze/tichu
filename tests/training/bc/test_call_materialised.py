"""Roundtrip + version-pin tests for the materialised Calls bundle.

Mirrors `test_materialised.py` / `test_schupfen_materialised.py` but for the
Call Networks ([ADR-0007](../../../docs/adr/0007-calls-are-standalone-networks.md)):
two call types (`call_tichu` / `call_grand_tichu`) in one bundle directory,
a binary target, and **no legal mask** (the decision is call / skip). See
[ADR-0020](../../../docs/adr/0020-one-parse-pass-many-task-bundles.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tichu_training.bc.call_materialised import (
    MemmapCallDataset,
    materialise_calls,
)
from tichu_training.bc.call_training import CallExample, SyntheticCallDataset
from tichu_training.checkpoint import VersionMismatchError
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM


@pytest.fixture
def materialised_smoke(tmp_path: Path) -> tuple[dict[str, list[CallExample]], Path]:
    """Materialise two binary-feature synthetic call streams (one per call
    type) into a single bundle directory."""
    streams = {
        "call_tichu": list(
            SyntheticCallDataset(seed=1, n_examples=20, binary_features=True)
        ),
        "call_grand_tichu": list(
            SyntheticCallDataset(seed=2, n_examples=15, binary_features=True)
        ),
    }
    out_dir = tmp_path / "calls_bundle"
    materialise_calls({k: iter(v) for k, v in streams.items()}, out_dir)
    return streams, out_dir


def test_roundtrip_reads_back_each_call_type_exactly(materialised_smoke):
    """Tracer: each call type's examples round-trip through the packed
    bundle — features, binary target, skill, and sample_weight."""
    streams, out_dir = materialised_smoke
    for call_type, examples in streams.items():
        read = list(MemmapCallDataset(out_dir, call_type=call_type))
        assert len(read) == len(examples)
        for got, want in zip(read, examples):
            np.testing.assert_array_equal(got.features, want.features)
            assert got.target == want.target
            assert got.skill_decile == want.skill_decile
            assert got.sample_weight == pytest.approx(want.sample_weight)


def test_features_bit_packed_and_no_legal_mask_file(materialised_smoke):
    """Indicator cols pack to ceil(214/8)=27 B, continuous stay 40 B — and a
    calls bundle has NO legal-mask file (the decision is binary)."""
    streams, out_dir = materialised_smoke
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["features_packed"] is True
    n_cont = len(manifest["continuous_feature_columns"])
    feat_bits_bytes = (FEATURIZER_OUTPUT_DIM - n_cont + 7) // 8
    for call_type, examples in streams.items():
        n = len(examples)
        assert (out_dir / f"{call_type}_feat_bits.dat").stat().st_size == n * feat_bits_bytes
        assert (out_dir / f"{call_type}_feat_cont.dat").stat().st_size == n * n_cont * 4
        assert not (out_dir / f"{call_type}_legal_mask.dat").exists()


@pytest.mark.parametrize(
    "pin,bad",
    [
        ("expected_featurizer_version", "vBOGUS"),
        ("expected_action_space_version", "vBOGUS"),
        ("expected_schema_version", 999),
    ],
)
def test_version_pin_mismatch_raises(materialised_smoke, pin, bad):
    _, out_dir = materialised_smoke
    with pytest.raises(VersionMismatchError):
        MemmapCallDataset(out_dir, call_type="call_tichu", **{pin: bad})


def test_unknown_call_type_raises(materialised_smoke):
    _, out_dir = materialised_smoke
    with pytest.raises(KeyError):
        MemmapCallDataset(out_dir, call_type="call_nonsense")


def test_iter_batches_matches_iter(materialised_smoke):
    _, out_dir = materialised_smoke
    ds = MemmapCallDataset(out_dir, call_type="call_grand_tichu")
    rows = list(ds)
    batched = list(ds.iter_batches(batch_size=4))
    feat = np.concatenate([b["features"] for b in batched])
    target = np.concatenate([b["target"] for b in batched])
    assert feat.shape[0] == len(rows)
    for i, ex in enumerate(rows):
        np.testing.assert_array_equal(feat[i], ex.features)
        assert int(target[i]) == ex.target


def test_writer_guard_rejects_non_binary_features(tmp_path):
    """Default (Gaussian) synthetic features must trip the codec's 0/1
    guard rather than silently corrupt the packed bits."""
    bad = list(SyntheticCallDataset(seed=3, n_examples=5))  # binary_features=False
    with pytest.raises(ValueError, match="non-0/1"):
        materialise_calls({"call_tichu": iter(bad)}, tmp_path / "bad")
