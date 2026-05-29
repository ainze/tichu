"""Roundtrip + version-pin tests for the materialised BC bundle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tichu_training.bc.dataset import BCExample, SyntheticBCDataset
from tichu_training.bc.heads import HEAD_LOGIT_DIMS
from tichu_training.bc.materialised import (
    MATERIALISED_SCHEMA_VERSION,
    MemmapBCDataset,
    TYPE_ORDER,
    materialise,
)
from tichu_training.checkpoint import VersionMismatchError
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM, FEATURIZER_VERSION


@pytest.fixture
def materialised_smoke(tmp_path: Path) -> tuple[list[BCExample], Path]:
    """Materialise a SyntheticBCDataset slice to disk and return (source, out_dir)."""
    src = SyntheticBCDataset(seed=42, n_per_head=20, skill_buckets=10)
    examples = list(src)
    out_dir = tmp_path / "bundle"
    materialise(iter(examples), out_dir, max_examples=len(examples))
    return examples, out_dir


def test_materialise_emits_manifest_and_per_type_files(materialised_smoke):
    _, out_dir = materialised_smoke
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == MATERIALISED_SCHEMA_VERSION
    assert manifest["featurizer_version"] == FEATURIZER_VERSION
    assert manifest["feature_dim"] == FEATURIZER_OUTPUT_DIM
    assert manifest["type_order"] == TYPE_ORDER
    # All three heads should have written files in the synthetic case.
    for head in HEAD_LOGIT_DIMS:
        assert (out_dir / f"{head}_features.dat").exists()
        assert (out_dir / f"{head}_legal_mask.dat").exists()
        assert (out_dir / f"{head}_meta.dat").exists()
    assert (out_dir / "order.dat").exists()


def test_roundtrip_iter_yields_same_examples_in_order(materialised_smoke):
    """Source order via SyntheticBCDataset must match the reader's __iter__
    after a materialise pass. Preserves the contract that the bundle is a
    drop-in for Iterable[BCExample]."""
    src_examples, out_dir = materialised_smoke
    ds = MemmapBCDataset(out_dir)
    read_examples = list(ds)
    assert len(read_examples) == len(src_examples)
    for src, got in zip(src_examples, read_examples):
        assert got.decision_type == src.decision_type
        assert got.target == src.target
        np.testing.assert_array_equal(got.features, src.features)
        np.testing.assert_array_equal(got.legal_mask, src.legal_mask)
        assert got.sample_weight == pytest.approx(src.sample_weight)
        assert got.skill_decile == src.skill_decile
        assert got.round_outcome == pytest.approx(src.round_outcome, abs=1e-4)
        assert got.game_won == src.game_won


def test_iter_batches_preserves_rows_against_iter(materialised_smoke):
    """The fast batched path must contain exactly the same per-row data as
    the per-example __iter__ path. This is the contract that lets training
    swap from `route-per-example + _to_tensors(batch)` to `iter_batches`
    without changing the loss curve."""
    _, out_dir = materialised_smoke
    ds = MemmapBCDataset(out_dir)

    # Collect per-head expected sequences via __iter__.
    expected: dict[str, list[BCExample]] = {h: [] for h in HEAD_LOGIT_DIMS}
    for ex in ds:
        expected[ex.decision_type].append(ex)

    # Walk batched output and re-assemble per-head sequences.
    got: dict[str, dict[str, list]] = {
        h: {"features": [], "target": [], "legal_mask": [],
            "sample_weight": [], "skill_decile": []}
        for h in HEAD_LOGIT_DIMS
    }
    for head, batch in ds.iter_batches(batch_size=8, preserve_order=True):
        got[head]["features"].append(batch["features"])
        got[head]["target"].append(batch["target"])
        got[head]["legal_mask"].append(batch["legal_mask"])
        got[head]["sample_weight"].append(batch["sample_weight"])
        got[head]["skill_decile"].append(batch["skill_decile"])

    for head, exs in expected.items():
        if not exs:
            continue
        feat_got = np.concatenate(got[head]["features"])
        feat_exp = np.stack([e.features for e in exs])
        np.testing.assert_array_equal(feat_got, feat_exp)

        tgt_got = np.concatenate(got[head]["target"])
        tgt_exp = np.array([e.target for e in exs], dtype=np.int64)
        np.testing.assert_array_equal(tgt_got, tgt_exp)

        mask_got = np.concatenate(got[head]["legal_mask"])
        mask_exp = np.stack([e.legal_mask for e in exs])
        np.testing.assert_array_equal(mask_got, mask_exp)


def test_iter_batches_drop_last(materialised_smoke):
    """`drop_last=True` must drop partial trailing batches; default keeps them."""
    _, out_dir = materialised_smoke
    ds = MemmapBCDataset(out_dir)
    bs = 7  # 20 examples per head / 7 → 2 full + 1 partial
    kept_counts: dict[str, int] = {h: 0 for h in HEAD_LOGIT_DIMS}
    for head, batch in ds.iter_batches(bs, drop_last=False, preserve_order=False):
        kept_counts[head] += len(batch["target"])
    dropped_counts: dict[str, int] = {h: 0 for h in HEAD_LOGIT_DIMS}
    for head, batch in ds.iter_batches(bs, drop_last=True, preserve_order=False):
        dropped_counts[head] += len(batch["target"])
    # Each head has 20 rows; drop_last=True yields 14 (2 full batches of 7).
    for head in HEAD_LOGIT_DIMS:
        assert kept_counts[head] == 20
        assert dropped_counts[head] == 14


def test_version_mismatch_refuses_to_load(materialised_smoke):
    """Reader must refuse a bundle whose featurizer_version doesn't match
    the caller's expectation. Same contract as ParquetBCDataset's pin."""
    _, out_dir = materialised_smoke
    with pytest.raises(VersionMismatchError, match="featurizer_version"):
        MemmapBCDataset(
            out_dir,
            expected_featurizer_version="NOT-A-REAL-VERSION",
        )


def test_schema_version_mismatch_refuses_to_load(materialised_smoke):
    """A future schema bump must invalidate older bundles cleanly."""
    _, out_dir = materialised_smoke
    with pytest.raises(VersionMismatchError, match="schema_version"):
        MemmapBCDataset(
            out_dir,
            expected_schema_version=MATERIALISED_SCHEMA_VERSION + 1,
        )


def test_action_space_version_mismatch_refuses_to_load(materialised_smoke):
    """Action-space changes invalidate target/mask semantics — must be caught."""
    _, out_dir = materialised_smoke
    with pytest.raises(VersionMismatchError, match="action_space_version"):
        MemmapBCDataset(
            out_dir,
            expected_action_space_version="NOT-A-REAL-VERSION",
        )


def test_missing_manifest_raises_filenotfound(tmp_path: Path):
    """An empty directory looks nothing like a bundle; hint at the CLI."""
    with pytest.raises(FileNotFoundError, match="materialise_bc"):
        MemmapBCDataset(tmp_path)


def test_iter_batches_invalid_batch_size_raises(materialised_smoke):
    _, out_dir = materialised_smoke
    ds = MemmapBCDataset(out_dir)
    with pytest.raises(ValueError, match="batch_size"):
        next(iter(ds.iter_batches(0)))


def test_n_rows_matches_manifest(materialised_smoke):
    src_examples, out_dir = materialised_smoke
    ds = MemmapBCDataset(out_dir)
    assert ds.n_rows == len(src_examples)
    assert ds.total == ds.n_rows


def test_game_won_none_roundtrips(materialised_smoke):
    """SyntheticBCDataset emits ~10% game_won=None rows; the i1 sentinel
    encoding (-1) must roundtrip through materialise + read."""
    src_examples, out_dir = materialised_smoke
    ds = MemmapBCDataset(out_dir)
    src_nones = sum(1 for e in src_examples if e.game_won is None)
    got_nones = sum(1 for e in ds if e.game_won is None)
    assert src_nones == got_nones
    # And the seed-42 stream must contain at least one None to make this
    # test meaningful.
    assert src_nones > 0
