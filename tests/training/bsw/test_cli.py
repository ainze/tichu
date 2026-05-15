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
        "play", "pass_card", "call_tichu", "call_grand_tichu", "wish_rank", "dragon_give"
    }
    assert (output / "known_bad_games.txt").exists()

    # Play shard has many rows; verify schema sanity through one column.
    table = pq.read_table(output / "play.parquet")
    assert table.num_rows > 100
    assert "decision_type" in table.column_names


def test_cli_subset_flag_limits_inputs(tmp_path):
    output = tmp_path / "out"
    rc = main([
        "--input", str(_SAMPLES),
        "--output", str(output),
        "--subset", "1",
    ])
    assert rc == 0
    pass_card = pq.read_table(output / "pass_card.parquet")
    # Only one game processed → 10 rounds × 4 schupfen = 40 rows.
    assert pass_card.num_rows == 40


def test_cli_rejects_missing_input_dir(tmp_path):
    rc = main([
        "--input", str(tmp_path / "does-not-exist"),
        "--output", str(tmp_path / "out"),
    ])
    assert rc == 2
