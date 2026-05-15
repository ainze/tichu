"""Verify Parquet shards are produced with the expected schema."""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tichu_training.bsw.parser import parse_tch
from tichu_training.bsw.to_parquet import write_parquet_shards


_SAMPLES = Path(__file__).resolve().parents[3] / "sample"
_EXPECTED_DECISION_TYPES = {
    "play", "pass_card", "call_tichu", "call_grand_tichu", "wish_rank", "dragon_give"
}
_EXPECTED_COLUMNS = {
    "decision_type", "game_id", "round_id", "timestamp", "player_handle",
    "action_taken", "state", "legal_actions_mask", "round_outcome",
    "round_won", "featurizer_version", "action_space_version",
}


@pytest.fixture
def games():
    return [
        parse_tch((_SAMPLES / "2417500.tch").read_text(encoding="utf-8"), game_id="2417500"),
        parse_tch((_SAMPLES / "2417501.tch").read_text(encoding="utf-8"), game_id="2417501"),
    ]


def test_creates_one_parquet_per_decision_type(tmp_path, games):
    write_parquet_shards(games, tmp_path)
    written = {p.stem for p in tmp_path.glob("*.parquet")}
    assert written == _EXPECTED_DECISION_TYPES


def test_each_shard_has_the_required_schema(tmp_path, games):
    write_parquet_shards(games, tmp_path)
    for path in tmp_path.glob("*.parquet"):
        table = pq.read_table(path)
        assert set(table.column_names) == _EXPECTED_COLUMNS, (
            f"{path.name} schema missing: {_EXPECTED_COLUMNS - set(table.column_names)}"
        )


def test_play_shard_contains_play_and_pass_rows(tmp_path, games):
    write_parquet_shards(games, tmp_path)
    table = pq.read_table(tmp_path / "play.parquet")
    assert table.num_rows > 0
    decision_types = set(table.column("decision_type").to_pylist())
    assert decision_types == {"play"}
    actions = set(table.column("action_taken").to_pylist())
    assert any(a == "pass" for a in actions), "pass actions should be in the play shard"
    assert any(a.startswith("play:") for a in actions)


def test_pass_card_shard_contains_schupfen_rows(tmp_path, games):
    write_parquet_shards(games, tmp_path)
    table = pq.read_table(tmp_path / "pass_card.parquet")
    # Each round has 4 schupfen submissions; both samples have 10 rounds.
    assert table.num_rows == 4 * 10 * 2
    actions = table.column("action_taken").to_pylist()
    assert all(a.startswith("schupfen:") for a in actions)


def test_player_handles_match_seat_assignments(tmp_path, games):
    write_parquet_shards(games, tmp_path)
    table = pq.read_table(tmp_path / "play.parquet")
    handles = set(table.column("player_handle").to_pylist())
    # Game 00 seats: evi_sea, 1David, evdokia!!, Lisaaaaaaa.
    # Game 01 seats: evi_sea, 1David, evdokia!!, Steffi0722.
    expected = {"evi_sea", "1David", "evdokia!!", "Lisaaaaaaa", "Steffi0722"}
    assert handles == expected


def test_reserved_fields_are_null_until_featurizer_lands(tmp_path, games):
    write_parquet_shards(games, tmp_path)
    table = pq.read_table(tmp_path / "play.parquet")
    assert all(v is None for v in table.column("state").to_pylist())
    assert all(v is None for v in table.column("legal_actions_mask").to_pylist())
    assert all(v is None for v in table.column("featurizer_version").to_pylist())
    assert all(v is None for v in table.column("action_space_version").to_pylist())


def test_row_counts_returned_match_disk(tmp_path, games):
    counts = write_parquet_shards(games, tmp_path)
    for decision_type, n in counts.items():
        if n == 0:
            continue
        table = pq.read_table(tmp_path / f"{decision_type}.parquet")
        assert table.num_rows == n
