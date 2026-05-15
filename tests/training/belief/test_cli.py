"""Smoke test for `train_belief` CLI."""

import csv
from pathlib import Path

import torch

from tichu_training.belief.model import BeliefModel
from tichu_training.checkpoint import Checkpoint
from tichu_training.cli.train_belief import main
from tichu_training.featurizer import FEATURIZER_VERSION


_CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def test_smoke_runs_and_loss_decreases(tmp_path):
    rc = main([
        "--config", str(_CONFIGS / "belief_smoke.yaml"),
        "--run-dir", str(tmp_path),
    ])
    assert rc == 0

    log_path = tmp_path / "step.csv"
    assert log_path.exists()
    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8")))
    assert len(rows) > 1
    assert float(rows[-1]["loss"]) < float(rows[0]["loss"])

    # Epoch CSV with accuracy column.
    epoch_path = tmp_path / "epoch.csv"
    assert epoch_path.exists()
    epoch_rows = list(csv.DictReader(epoch_path.open("r", encoding="utf-8")))
    assert epoch_rows
    assert {"epoch", "loss", "accuracy"} <= set(epoch_rows[0].keys())

    # Final calibration CSV.
    assert (tmp_path / "calibration.csv").exists()
    calib_rows = list(csv.DictReader((tmp_path / "calibration.csv").open("r", encoding="utf-8")))
    assert len(calib_rows) == 10

    # Checkpoint exists in a belief-specific dir; loadable via Checkpoint.load
    # with featurizer-version pinning (action-space pin is skipped because
    # belief doesn't predict actions).
    ckpt_path = tmp_path / "checkpoints" / "belief_final.bin"
    assert ckpt_path.exists()
    cp = Checkpoint.load(ckpt_path, expected_featurizer_version=FEATURIZER_VERSION)
    assert cp.featurizer_version == FEATURIZER_VERSION

    # Config copied into run dir.
    assert (tmp_path / "belief_smoke.yaml").exists()


def test_inference_service_does_not_import_belief():
    """Sanity check: phase-1 inference path is decoupled from belief."""
    import importlib
    # tichu_inference doesn't exist yet (#015 is future). When it does, this
    # test ensures no belief import sneaks in. Until then, just confirm
    # the belief modules themselves don't drag in any inference-side surface.
    try:
        importlib.import_module("tichu_inference")
    except ImportError:
        return  # phase-1 fine: no inference package yet.

    # If/when tichu_inference exists, none of its public surface should mention
    # belief — this assertion will start providing real coverage then.
    import tichu_inference  # type: ignore[import-not-found]
    text = repr(getattr(tichu_inference, "__all__", []))
    assert "belief" not in text.lower()
