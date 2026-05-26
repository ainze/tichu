"""End-to-end test of the `parse_bsw` CLI on the checked-in samples."""

from pathlib import Path

import pyarrow.parquet as pq

from tichu_training.cli.parse_bsw import main


_SAMPLES = Path(__file__).resolve().parents[3] / "sample"


def test_cli_runs_end_to_end_on_samples(tmp_path):
    output = tmp_path / "out"
    rc = main([
        "--input", str(_SAMPLES),
        "--output", str(output),
    ])
    assert rc == 0

    # All six decision-type shards should be written.
    written = {p.stem for p in output.glob("*.parquet")}
    assert written == {
        "play_00000", "schupfen_00000", "call_tichu_00000",
        "call_grand_tichu_00000", "wish_00000", "dragon_assignment_00000",
    }
    assert (output / "known_bad_games.txt").exists()

    # Play shard has many rows; verify schema sanity through one column.
    table = pq.read_table(output / "play_00000.parquet")
    assert table.num_rows > 100
    assert "decision_type" in table.column_names


def test_cli_subset_flag_limits_inputs(tmp_path):
    from tichu_training.bsw.parser import parse_tch
    from tichu_training.bsw.validate import validate_game

    output = tmp_path / "out"
    rc = main([
        "--input", str(_SAMPLES),
        "--output", str(output),
        "--subset", "1",
    ])
    assert rc == 0
    pass_card = pq.read_table(output / "schupfen_00000.parquet")
    # Per-round filtering (ADR-0009): one game × matching_rounds × 4 schupfen.
    first_sample = sorted(_SAMPLES.glob("*.tch"))[0]
    game = parse_tch(first_sample.read_bytes().decode("utf-8"), game_id=first_sample.stem)
    matching_rounds = sum(1 for r in validate_game(game).rounds if r.matches)
    assert pass_card.num_rows == matching_rounds * 4


def test_cli_rejects_missing_input_dir(tmp_path):
    rc = main([
        "--input", str(tmp_path / "does-not-exist"),
        "--output", str(tmp_path / "out"),
    ])
    assert rc == 2


def test_cli_joins_trueskill_when_flag_provided(tmp_path):
    import pyarrow as pa
    ratings_path = tmp_path / "ratings.parquet"
    pq.write_table(
        pa.table({
            "player_handle": ["evi_sea", "1David"],
            "skill_decile": pa.array([8, 2], type=pa.int32()),
        }),
        ratings_path,
    )
    output = tmp_path / "out"
    rc = main([
        "--input", str(_SAMPLES),
        "--output", str(output),
        "--trueskill", str(ratings_path),
    ])
    assert rc == 0
    table = pq.read_table(output / "play_00000.parquet")
    by_handle = dict(zip(
        table.column("player_handle").to_pylist(),
        table.column("skill_decile").to_pylist(),
    ))
    assert by_handle["evi_sea"] == 8
    assert by_handle["1David"] == 2
    # Unjoined handle stays null.
    assert by_handle.get("evdokia!!") is None


def test_cli_recency_flags_downweight_below_cutoff(tmp_path):
    output = tmp_path / "out"
    # Sample game_ids 2417500/2417501 — cutoff above forces downweighting.
    rc = main([
        "--input", str(_SAMPLES),
        "--output", str(output),
        "--recency-cutoff-game-id", "99999999",
        "--recency-weight", "0.1",
    ])
    assert rc == 0
    table = pq.read_table(output / "play_00000.parquet")
    weights = set(table.column("sample_weight").to_pylist())
    assert weights == {0.10000000149011612} or weights == {0.1}  # float32 round-trip
