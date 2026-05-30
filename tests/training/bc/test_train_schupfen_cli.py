"""CLI smoke for `train_schupfen` — Slice 6 of the design pass.

End-to-end coverage: argparse → config load → synthetic dataset → train
loop → save_checkpoint. Asserts the resulting checkpoint loads with the
live featurizer + action-space version pins (Q6).
"""

from pathlib import Path

import yaml

from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.checkpoint import Checkpoint
from tichu_training.cli.train_schupfen import main as train_schupfen_main
from tichu_training.featurizer import FEATURIZER_VERSION


def test_smoke_run_produces_loadable_versioned_checkpoint(tmp_path):
    config = {
        "dataset": "synthetic",
        "dataset_kwargs": {"seed": 0, "n_examples": 64, "feature_dim": 32},
        "model": {"hidden": 16, "skill_dim": 4},
        "learning_rate": 5.0e-3,
        "batch_size": 16,
        "epochs": 2,
        "seed": 0,
    }
    config_path = tmp_path / "schupfen_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = tmp_path / "run"

    rc = train_schupfen_main([
        "--config", str(config_path), "--run-dir", str(run_dir),
    ])
    assert rc == 0

    ckpt_path = run_dir / "checkpoints" / "schupfen_final.bin"
    assert ckpt_path.exists()

    ckpt = Checkpoint.load(
        ckpt_path,
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    assert ckpt.featurizer_version == FEATURIZER_VERSION
    assert ckpt.action_space_version == ACTION_SPACE_VERSION

    # step.csv is the training log; CLI must have produced it.
    assert (run_dir / "step.csv").exists()
