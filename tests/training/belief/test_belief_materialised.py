"""Roundtrip + version-pin tests for the materialised Belief bundle (ADR-0021).

Belief is the one play-scale new bundle: per Play Decision, the acting
player's 224-dim feature vector + a (3 opponents, 56 cards) multi-hot label
grid + a (56,) card-level "unknown" mask (broadcast to (3,56) on read) +
`cards_played` provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tichu_training.belief.belief_materialised import (
    MemmapBeliefDataset,
    materialise_belief,
)
from tichu_training.belief.dataset import BeliefExample, SyntheticBeliefDataset
from tichu_training.checkpoint import VersionMismatchError
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM


@pytest.fixture
def materialised_smoke(tmp_path: Path) -> tuple[list[BeliefExample], Path]:
    src = list(SyntheticBeliefDataset(
        seed=0, n_examples=20, feature_dim=FEATURIZER_OUTPUT_DIM,
        binary_features=True,
    ))
    out_dir = tmp_path / "belief_bundle"
    materialise_belief(iter(src), out_dir)
    return src, out_dir


def test_roundtrip_reads_back_examples_exactly(materialised_smoke):
    """Tracer: features, the (3,56) labels, and the (3,56) mask round-trip."""
    src, out_dir = materialised_smoke
    read = list(MemmapBeliefDataset(out_dir))
    assert len(read) == len(src)
    for got, want in zip(read, src):
        np.testing.assert_array_equal(got.features, want.features)
        np.testing.assert_array_equal(got.labels, want.labels)
        np.testing.assert_array_equal(got.mask, want.mask)


def test_labels_and_mask_bit_packed_on_disk(materialised_smoke):
    """Labels (3*56=168 bits) pack to 21 B/row; the (56,) card mask to 7 B/row."""
    src, out_dir = materialised_smoke
    n = len(src)
    assert (out_dir / "belief_labels.dat").stat().st_size == n * 21
    assert (out_dir / "belief_mask.dat").stat().st_size == n * 7


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
        MemmapBeliefDataset(out_dir, **{pin: bad})
