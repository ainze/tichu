"""`train_schupfen` reading the materialised Schupfen bundle (`dataset: memmap`).

The memmap path lets `train_schupfen` consume the packed schupfen bundle from
the consolidated parse pass ([ADR-0020](../../../docs/adr/0020-one-parse-pass-many-task-bundles.md))
instead of re-parsing the archive to rebuild synthetic pre-schupfen states.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.schupfen_materialised import materialise_schupfen
from tichu_training.bc.schupfen_training import SyntheticSchupfenDataset
from tichu_training.checkpoint import Checkpoint
from tichu_training.cli.train_schupfen import main as train_schupfen_main
from tichu_training.featurizer import FEATURIZER_VERSION


def test_cli_trains_from_memmap_bundle(tmp_path):
    """End-to-end: `main` with `dataset: memmap` trains the Schupfen network
    off the packed bundle and saves a version-stamped checkpoint."""
    examples = list(
        SyntheticSchupfenDataset(seed=7, n_examples=48, binary_features=True)
    )
    bundle = tmp_path / "schupfen_bundle"
    materialise_schupfen(iter(examples), bundle, max_examples=len(examples))

    config = {
        "dataset": "memmap",
        "dataset_kwargs": {"data_dir": str(bundle)},
        "model": {"hidden": 16, "skill_dim": 4},
        "learning_rate": 5.0e-3,
        "batch_size": 16,
        "epochs": 2,
        "seed": 0,
    }
    config_path = tmp_path / "schupfen_memmap.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = tmp_path / "run"

    rc = train_schupfen_main(["--config", str(config_path), "--run-dir", str(run_dir)])
    assert rc == 0

    ckpt_path = run_dir / "checkpoints" / "schupfen_final.bin"
    assert ckpt_path.exists()
    ckpt = Checkpoint.load(
        ckpt_path,
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    assert ckpt.featurizer_version == FEATURIZER_VERSION
    assert (run_dir / "step.csv").exists()
