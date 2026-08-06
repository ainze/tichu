"""Reader utility that concatenates shards and validates version columns."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tichu_training.checkpoint import VersionMismatchError
from tichu_training.records import load_shards


def _stub_table(featurizer_version: str = "v1", action_space_version: str = "v1", n_rows: int = 4) -> pa.Table:
    return pa.table({
        "decision_type": ["play"] * n_rows,
        "game_id": [str(i) for i in range(n_rows)],
        "round_id": pa.array([0] * n_rows, type=pa.int32()),
        "timestamp": pa.array([None] * n_rows, type=pa.int64()),
        "player_handle": ["alice"] * n_rows,
        "action_taken": ["pass"] * n_rows,
        "state": pa.array([None] * n_rows, type=pa.binary()),
        "legal_actions_mask": pa.array([None] * n_rows, type=pa.binary()),
        "round_outcome": pa.array([0] * n_rows, type=pa.int32()),
        "round_won": [False] * n_rows,
        "featurizer_version": [featurizer_version] * n_rows,
        "action_space_version": [action_space_version] * n_rows,
        "skill_decile": pa.array([None] * n_rows, type=pa.int32()),
        "sample_weight": pa.array([1.0] * n_rows, type=pa.float32()),
    })


def test_loads_single_shard(tmp_path):
    pq.write_table(_stub_table(), tmp_path / "play_00000.parquet")
    table = load_shards(
        tmp_path, "play",
        expected_featurizer_version="v1",
        expected_action_space_version="v1",
    )
    assert table.num_rows == 4


def test_loads_multiple_shards_in_sorted_order(tmp_path):
    t0 = _stub_table(n_rows=2)
    t1 = _stub_table(n_rows=3)
    pq.write_table(t0, tmp_path / "play_00000.parquet")
    pq.write_table(t1, tmp_path / "play_00001.parquet")
    table = load_shards(tmp_path, "play",
                        expected_featurizer_version="v1",
                        expected_action_space_version="v1")
    assert table.num_rows == 5
    # First two game_ids are from t0 ("0","1"), next three from t1 ("0","1","2").
    assert table.column("game_id").to_pylist() == ["0", "1", "0", "1", "2"]


def test_featurizer_version_mismatch_raises(tmp_path):
    pq.write_table(_stub_table(featurizer_version="v1"), tmp_path / "play_00000.parquet")
    with pytest.raises(VersionMismatchError) as ei:
        load_shards(tmp_path, "play",
                    expected_featurizer_version="v2",
                    expected_action_space_version="v1")
    msg = str(ei.value)
    assert "featurizer_version" in msg
    assert "v1" in msg and "v2" in msg


def test_action_space_version_mismatch_raises(tmp_path):
    pq.write_table(_stub_table(action_space_version="v1"), tmp_path / "play_00000.parquet")
    with pytest.raises(VersionMismatchError) as ei:
        load_shards(tmp_path, "play",
                    expected_featurizer_version="v1",
                    expected_action_space_version="v99")
    msg = str(ei.value)
    assert "action_space_version" in msg
    assert "v1" in msg and "v99" in msg


def test_no_shards_found_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_shards(tmp_path, "play",
                    expected_featurizer_version="v1",
                    expected_action_space_version="v1")


def test_mixed_versions_across_shards_raises(tmp_path):
    pq.write_table(_stub_table(featurizer_version="v1"), tmp_path / "play_00000.parquet")
    pq.write_table(_stub_table(featurizer_version="v2"), tmp_path / "play_00001.parquet")
    with pytest.raises(VersionMismatchError) as ei:
        load_shards(tmp_path, "play",
                    expected_featurizer_version="v1",
                    expected_action_space_version="v1")
    msg = str(ei.value)
    assert "featurizer_version" in msg
    assert "v2" in msg


def test_column_projection_avoids_reading_the_whole_row_group(tmp_path):
    """`_build_manifest` needs only (game_id, round_id), but load_shards used to
    read all 15 columns of every shard and then `to_pylist()` the version column.

    On the full corpus that is 1.45 BILLION Python strings and it dies with
    MemoryError before a single row is materialised. Projecting the columns is
    the fix; this pins the projection so the manifest path cannot silently start
    dragging the whole table back in.
    """
    pq.write_table(_stub_table(featurizer_version="v6"),
                   tmp_path / "play_00000.parquet")
    table = load_shards(
        tmp_path, "play",
        expected_featurizer_version="v6",
        expected_action_space_version="v1",
        columns=["game_id", "round_id"],
    )
    assert table.column_names == ["game_id", "round_id"]


def test_featurizer_version_pin_can_be_waived_for_the_replay_path(tmp_path):
    """The parquet shards carry NO featurized content — `state` and
    `legal_actions_mask` are 100% null, and features are computed at
    materialise time by replaying the archive (ADR-0011). Their
    `featurizer_version` stamp therefore describes a column that does not exist.

    Pinning it would force a multi-hour BSW re-parse on every featurizer bump for
    a vestigial field. `None` waives it explicitly. The real guard is unaffected:
    `materialise()` stamps the live FEATURIZER_VERSION into the BUNDLE manifest
    and `MemmapBCDataset` refuses to load on mismatch.
    """
    pq.write_table(_stub_table(featurizer_version="v6"),
                   tmp_path / "play_00000.parquet")
    table = load_shards(
        tmp_path, "play",
        expected_featurizer_version=None,     # waived, deliberately
        expected_action_space_version="v1",   # still pinned: labels are v1
    )
    assert table.num_rows > 0

    with pytest.raises(VersionMismatchError):
        load_shards(
            tmp_path, "play",
            expected_featurizer_version="v6",
            expected_action_space_version="v99",  # a pin that DOES matter
        )
