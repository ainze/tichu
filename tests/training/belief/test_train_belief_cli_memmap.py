"""`train_belief` reading the materialised Belief bundle (`dataset: memmap`).

The memmap path lets `train_belief` consume the packed belief bundle
([ADR-0021](../../../docs/adr/0021-belief-trains-on-a-replay-derived-bundle.md))
instead of the synthetic-only dataset — the first real-data path for belief.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from tichu_training.belief.belief_materialised import materialise_belief
from tichu_training.belief.dataset import SyntheticBeliefDataset
from tichu_training.belief.history import BELIEF_FEATURE_DIM
from tichu_training.checkpoint import Checkpoint
from tichu_training.cli.train_belief import main as train_belief_main
from tichu_training.featurizer import FEATURIZER_VERSION


def test_cli_trains_from_memmap_bundle(tmp_path):
    """End-to-end: `main` with `dataset: memmap` trains the belief model off
    the packed bundle and saves a version-stamped checkpoint."""
    src = list(SyntheticBeliefDataset(
        seed=0, n_examples=24, feature_dim=BELIEF_FEATURE_DIM,
        binary_features=True,
    ))
    bundle = tmp_path / "belief_bundle"
    materialise_belief(iter(src), bundle)

    config = {
        "dataset": "memmap",
        "dataset_kwargs": {"data_dir": str(bundle)},
        "model": {"hidden": 16},
        "learning_rate": 5.0e-3,
        "batch_size": 8,
        "epochs": 2,
        "seed": 0,
    }
    config_path = tmp_path / "belief_memmap.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = tmp_path / "run"

    rc = train_belief_main(["--config", str(config_path), "--run-dir", str(run_dir)])
    assert rc == 0

    ckpt_path = run_dir / "checkpoints" / "belief_final.bin"
    assert ckpt_path.exists()
    ckpt = Checkpoint.load(ckpt_path, expected_featurizer_version=FEATURIZER_VERSION)
    assert ckpt.featurizer_version == FEATURIZER_VERSION
    assert (run_dir / "epoch.csv").exists()


def test_cli_streaming_path_trains_tier_bcore(tmp_path):
    """The low-RAM `streaming: true` path trains a tier-sliced (B_core=251)
    model off the memmap without materialising the example list."""
    src = list(SyntheticBeliefDataset(
        seed=0, n_examples=40, feature_dim=BELIEF_FEATURE_DIM, binary_features=True,
    ))
    bundle = tmp_path / "belief_bundle"
    materialise_belief(iter(src), bundle)

    config = {
        "dataset": "memmap",
        "dataset_kwargs": {"data_dir": str(bundle)},
        "streaming": True,
        "tier": "B_core",
        "model": {"hidden": 16},
        "learning_rate": 5.0e-3,
        "batch_size": 8,
        "epochs": 2,
        "seed": 0,
    }
    config_path = tmp_path / "belief_stream.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = tmp_path / "run_stream"
    rc = train_belief_main(["--config", str(config_path), "--run-dir", str(run_dir)])
    assert rc == 0
    assert (run_dir / "checkpoints" / "belief_final.bin").exists()
    # B_core slices to 251 input columns.
    import torch
    from tichu_training.belief.history import TIER_DIMS
    state = torch.load(
        __import__("io").BytesIO(
            Checkpoint.load(
                run_dir / "checkpoints" / "belief_final.bin",
                expected_featurizer_version=FEATURIZER_VERSION,
            ).payload
        ),
        weights_only=True,
    )["model"]
    assert state["fc1.weight"].shape[1] == TIER_DIMS["B_core"]
