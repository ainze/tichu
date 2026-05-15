"""Smoke test for `train_calls` CLI."""

import csv
from pathlib import Path

from tichu_training.cli.train_calls import main


_CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def test_smoke_runs_both_networks_and_loss_decreases(tmp_path):
    rc = main([
        "--config", str(_CONFIGS / "calls_smoke.yaml"),
        "--run-dir", str(tmp_path),
    ])
    assert rc == 0

    for tag in ("grand", "tichu"):
        log_path = tmp_path / f"{tag}_step.csv"
        assert log_path.exists()
        rows = list(csv.DictReader(log_path.open("r", encoding="utf-8")))
        assert len(rows) > 1
        assert float(rows[-1]["loss"]) < float(rows[0]["loss"])
        # Final checkpoint exists.
        assert (tmp_path / "checkpoints" / f"{tag}_final.bin").exists()
        # Calling-rate CSV written per epoch.
        rate_files = list(tmp_path.glob(f"{tag}_calling_rate_by_decile_epoch*.csv"))
        assert rate_files
        rate_rows = list(csv.DictReader(rate_files[0].open("r", encoding="utf-8")))
        assert rate_rows and {"decile", "n", "calling_rate"} <= set(rate_rows[0].keys())

    # Config copied into run dir.
    assert (tmp_path / "calls_smoke.yaml").exists()


def test_only_flag_restricts_to_one_network(tmp_path):
    rc = main([
        "--config", str(_CONFIGS / "calls_smoke.yaml"),
        "--run-dir", str(tmp_path),
        "--only", "grand",
    ])
    assert rc == 0
    assert (tmp_path / "checkpoints" / "grand_final.bin").exists()
    assert not (tmp_path / "checkpoints" / "tichu_final.bin").exists()
