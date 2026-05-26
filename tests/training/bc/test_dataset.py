"""Synthetic dataset + parquet dataset version-mismatch handling."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tichu_training.action_space import ACTION_SPACE_SIZE
from tichu_training.bc.dataset import (
    ParquetBCDataset,
    SyntheticBCDataset,
)
from tichu_training.bc.heads import HEAD_LOGIT_DIMS
from tichu_training.checkpoint import VersionMismatchError
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM


def test_synthetic_dataset_is_deterministic():
    a = list(SyntheticBCDataset(seed=42, n_per_head=10, feature_dim=8))
    b = list(SyntheticBCDataset(seed=42, n_per_head=10, feature_dim=8))
    assert len(a) == len(b)
    for ea, eb in zip(a, b):
        assert ea.decision_type == eb.decision_type
        assert (ea.features == eb.features).all()
        assert ea.target == eb.target


def test_synthetic_dataset_yields_at_least_n_per_head():
    examples = list(SyntheticBCDataset(seed=0, n_per_head=10, feature_dim=8))
    counts = {h: 0 for h in HEAD_LOGIT_DIMS}
    for e in examples:
        counts[e.decision_type] += 1
    for h in HEAD_LOGIT_DIMS:
        assert counts[h] == 10


def test_synthetic_targets_are_within_legal_mask():
    for e in SyntheticBCDataset(seed=1, n_per_head=20, feature_dim=8):
        assert e.legal_mask[e.target], (
            f"target {e.target} not in legal mask for {e.decision_type}"
        )


def test_synthetic_features_match_requested_dim():
    e = next(iter(SyntheticBCDataset(seed=0, n_per_head=1, feature_dim=64)))
    assert e.features.shape == (64,)


def test_parquet_dataset_raises_on_featurizer_version_mismatch(tmp_path):
    _write_minimal_shard(tmp_path / "play_00000.parquet", featurizer_version="v1")
    # archive_path is required by the new ADR-0011 constructor but is not
    # touched during version-pin validation, so a dummy path is fine here.
    with pytest.raises(VersionMismatchError):
        ParquetBCDataset(
            shards_dir=tmp_path,
            archive_path=tmp_path / "irrelevant.zst",
            expected_featurizer_version="v2",
            expected_action_space_version="v1",
        )


def test_parquet_dataset_accepts_matching_versions(tmp_path):
    _write_minimal_shard(tmp_path / "play_00000.parquet", featurizer_version="v1")
    ds = ParquetBCDataset(
        shards_dir=tmp_path,
        archive_path=tmp_path / "irrelevant.zst",
        expected_featurizer_version="v1",
        expected_action_space_version="v1",
    )
    assert ds.n_rows == 1


def _write_minimal_shard(path, featurizer_version: str = "v1") -> None:
    table = pa.table({
        "decision_type": ["play"],
        "game_id": ["0"],
        "round_id": pa.array([0], type=pa.int32()),
        "timestamp": pa.array([None], type=pa.int64()),
        "player_handle": ["alice"],
        "action_taken": ["pass"],
        "state": pa.array([None], type=pa.binary()),
        "legal_actions_mask": pa.array([None], type=pa.binary()),
        "round_outcome": pa.array([0], type=pa.int32()),
        "round_won": [False],
        "featurizer_version": [featurizer_version],
        "action_space_version": ["v1"],
        "skill_decile": pa.array([None], type=pa.int32()),
        "sample_weight": pa.array([1.0], type=pa.float32()),
    })
    pq.write_table(table, path)
