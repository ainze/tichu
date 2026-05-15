"""End-to-end test of the `compute_trueskill` CLI on the checked-in samples."""

import io
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import trueskill
import zstandard as zstd

from tichu_training.cli.compute_trueskill import main


_SAMPLES = Path(__file__).resolve().parents[3] / "sample"
_DEFAULT_MU = trueskill.TrueSkill().mu


def test_cli_runs_end_to_end_on_samples(tmp_path):
    output = tmp_path / "ratings.parquet"
    rc = main([
        "--input", str(_SAMPLES),
        "--output", str(output),
        "--min-games", "1",
    ])
    assert rc == 0
    assert output.exists()

    table = pq.read_table(output)
    cols = set(table.column_names)
    assert {"player_handle", "mu", "sigma", "n_games"} <= cols

    handles = set(table.column("player_handle").to_pylist())
    assert {"evi_sea", "1David", "evdokia!!"} <= handles

    # Every player in the sample should have at least one update applied.
    n_games = dict(zip(
        table.column("player_handle").to_pylist(),
        table.column("n_games").to_pylist(),
    ))
    mus = dict(zip(
        table.column("player_handle").to_pylist(),
        table.column("mu").to_pylist(),
    ))
    for handle in handles:
        assert n_games[handle] >= 1
        assert mus[handle] != _DEFAULT_MU, f"{handle} mu was never updated"


def test_cli_accepts_tar_zst_archive(tmp_path):
    archive = tmp_path / "games.tar.zst"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for tch in sorted(_SAMPLES.glob("*.tch")):
            data = tch.read_bytes()
            info = tarfile.TarInfo(name=tch.name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    archive.write_bytes(zstd.ZstdCompressor().compress(buf.getvalue()))

    out_dir = tmp_path / "from_dir.parquet"
    out_zst = tmp_path / "from_zst.parquet"
    assert main(["--input", str(_SAMPLES), "--output", str(out_dir), "--min-games", "1"]) == 0
    assert main(["--input", str(archive), "--output", str(out_zst), "--min-games", "1"]) == 0

    t_dir = pq.read_table(out_dir).to_pydict()
    t_zst = pq.read_table(out_zst).to_pydict()
    assert t_dir == t_zst


def test_cli_output_independent_of_archive_member_order(tmp_path):
    """Members written to the tar in reverse should still produce sorted-order ratings."""
    archive = tmp_path / "shuffled.tar.zst"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        # Reverse the natural sort order on purpose.
        for tch in sorted(_SAMPLES.glob("*.tch"), reverse=True):
            data = tch.read_bytes()
            info = tarfile.TarInfo(name=tch.name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    archive.write_bytes(zstd.ZstdCompressor().compress(buf.getvalue()))

    sorted_out = tmp_path / "sorted.parquet"
    shuffled_out = tmp_path / "shuffled.parquet"
    assert main(["--input", str(_SAMPLES), "--output", str(sorted_out), "--min-games", "1"]) == 0
    assert main(["--input", str(archive), "--output", str(shuffled_out), "--min-games", "1"]) == 0

    assert pq.read_table(sorted_out).to_pydict() == pq.read_table(shuffled_out).to_pydict()


def test_cli_rejects_missing_input(tmp_path):
    rc = main([
        "--input", str(tmp_path / "does-not-exist"),
        "--output", str(tmp_path / "ratings.parquet"),
    ])
    assert rc == 2


def test_cli_logs_decile_share_and_spearman_rho(tmp_path, caplog):
    output = tmp_path / "ratings.parquet"
    with caplog.at_level("INFO", logger="compute_trueskill"):
        rc = main([
            "--input", str(_SAMPLES),
            "--output", str(output),
            "--min-games", "1",
        ])
    assert rc == 0
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "decile share" in text
    # On 5 sample players nobody has tichu_calls >= 5, so we expect the skip path.
    assert "skipping spearman" in text.lower() or "spearman rho" in text.lower()


def test_cli_absurd_min_games_skips_rho_cleanly(tmp_path, caplog):
    output = tmp_path / "ratings.parquet"
    with caplog.at_level("WARNING", logger="compute_trueskill"):
        rc = main([
            "--input", str(_SAMPLES),
            "--output", str(output),
            "--min-games", "999999",
        ])
    assert rc == 0
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "skipping spearman" in text.lower()


def test_cli_default_min_games_is_20(tmp_path):
    """With only 2 sample games every player has n_games <= 2, so default --min-games=20 empties the table."""
    output = tmp_path / "ratings.parquet"
    assert main(["--input", str(_SAMPLES), "--output", str(output)]) == 0
    table = pq.read_table(output)
    assert table.num_rows == 0


def test_cli_min_games_one_keeps_all_sample_players(tmp_path):
    output = tmp_path / "ratings.parquet"
    assert main([
        "--input", str(_SAMPLES),
        "--output", str(output),
        "--min-games", "1",
    ]) == 0
    table = pq.read_table(output)
    assert table.num_rows == 5
    cols = set(table.column_names)
    assert "skill_decile" in cols
    assert {"tichu_calls", "tichu_success_rate",
            "grand_tichu_calls", "grand_tichu_success_rate"} <= cols
    deciles = table.column("skill_decile").to_pylist()
    assert all(0 <= d <= 9 for d in deciles)
    # Schema: skill_decile and tichu_calls must be int32.
    assert table.schema.field("skill_decile").type == pa.int32()
    assert table.schema.field("tichu_calls").type == pa.int32()
    assert table.schema.field("grand_tichu_calls").type == pa.int32()
    # Players who never called: counts are 0 and rate is null.
    rows = {h: i for i, h in enumerate(table.column("player_handle").to_pylist())}
    t_calls = table.column("tichu_calls").to_pylist()
    t_rates = table.column("tichu_success_rate").to_pylist()
    for handle in rows:
        i = rows[handle]
        if t_calls[i] == 0:
            assert t_rates[i] is None
        else:
            assert 0.0 <= t_rates[i] <= 1.0
