"""End-to-end smoke for `train_bc --refine-from <bc_ckpt>` AWR pipeline."""

import csv
from pathlib import Path

from tichu_training.bc.heads import BCModel
from tichu_training.bc.training import load_checkpoint
from tichu_training.cli.train_bc import main


_CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def _train_bc_checkpoint(tmp_path: Path) -> Path:
    """Run a BC smoke pass and return the resulting checkpoint path."""
    bc_run = tmp_path / "bc"
    bc_run.mkdir()
    rc = main([
        "--config", str(_CONFIGS / "bc_smoke.yaml"),
        "--run-dir", str(bc_run),
    ])
    assert rc == 0
    ckpts = sorted((bc_run / "checkpoints").glob("*.bin"))
    assert ckpts
    return ckpts[-1]


def test_awr_smoke_runs_and_writes_compatible_checkpoint(tmp_path):
    bc_ckpt = _train_bc_checkpoint(tmp_path)

    awr_run = tmp_path / "awr"
    awr_run.mkdir()
    rc = main([
        "--config", str(_CONFIGS / "awr_smoke.yaml"),
        "--run-dir", str(awr_run),
        "--refine-from", str(bc_ckpt),
    ])
    assert rc == 0

    # Config copied into the run dir.
    assert (awr_run / "awr_smoke.yaml").exists()

    # AWR step CSV has the awr_weight_mean column; loss is finite.
    # Note: we can't reliably assert that loss decreases here because BC has
    # already converged to a near-zero loss on the trivial synthetic dataset,
    # so AWR's reweighting only nudges the gradient near a fixed point. The
    # quality signal lives in the eval matrix (#011), not the loss curve.
    step_csv = awr_run / "step.csv"
    assert step_csv.exists()
    rows = list(csv.DictReader(step_csv.open("r", encoding="utf-8")))
    assert rows
    assert "awr_weight_mean" in rows[0]
    for r in rows:
        loss = float(r["loss_total"])
        assert loss == loss  # not NaN
        assert loss >= 0
        assert float(r["awr_weight_mean"]) > 0

    # Epoch-level win-rate proxy logged.
    epoch_csv = awr_run / "epoch.csv"
    assert epoch_csv.exists()
    epoch_rows = list(csv.DictReader(epoch_csv.open("r", encoding="utf-8")))
    assert epoch_rows
    assert {"epoch", "loss_total", "avg_weight", "win_rate_proxy"} <= set(epoch_rows[0].keys())

    # Final checkpoint is loadable by the BC contract (format compatibility).
    final_ckpts = sorted((awr_run / "checkpoints").glob("*.bin"))
    assert final_ckpts
    model = BCModel(
        feature_dim=32, skill_buckets=10, skill_dim=4,
        trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16,
    )
    step = load_checkpoint(final_ckpts[-1], model)
    assert step > 0


def test_bc_smoke_still_passes_without_refine_from(tmp_path):
    """The AWR work must not regress plain BC training."""
    rc = main([
        "--config", str(_CONFIGS / "bc_smoke.yaml"),
        "--run-dir", str(tmp_path),
    ])
    assert rc == 0
    assert (tmp_path / "step.csv").exists()
    ckpts = list((tmp_path / "checkpoints").glob("*.bin"))
    assert ckpts
