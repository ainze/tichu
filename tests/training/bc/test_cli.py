"""End-to-end smoke test for `train_bc` CLI."""

import csv
from pathlib import Path

from tichu_training.cli.train_bc import main


_CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def test_smoke_config_runs_and_loss_decreases(tmp_path):
    rc = main([
        "--config", str(_CONFIGS / "bc_smoke.yaml"),
        "--run-dir", str(tmp_path),
    ])
    assert rc == 0

    log_path = tmp_path / "step.csv"
    assert log_path.exists()
    with log_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) > 1
    first_total = float(rows[0]["loss_total"])
    last_total = float(rows[-1]["loss_total"])
    assert last_total < first_total

    # Config was copied into the run dir.
    assert (tmp_path / "bc_smoke.yaml").exists()
    # At least one checkpoint exists.
    ckpts = list((tmp_path / "checkpoints").glob("*.bin"))
    assert ckpts
