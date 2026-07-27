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
from tichu_training.belief.input_spec import BELIEF_FEATURE_DIM
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


def _stream_config(bundle: Path) -> dict:
    return {
        "dataset": "memmap",
        "dataset_kwargs": {"data_dir": str(bundle)},
        "streaming": True,
        "model": {"hidden": 16},
        "learning_rate": 5.0e-3,
        "batch_size": 8,
        "epochs": 2,
        "seed": 0,
    }


def test_cli_streaming_path_trains_at_the_bundle_width(tmp_path):
    """The low-RAM `streaming: true` path trains off the memmap without
    materialising the example list, at the bundle's own feature width — no
    tier prefix (ADR-0028's tiers died with featurizer v6, ADR-0038)."""
    src = list(SyntheticBeliefDataset(
        seed=0, n_examples=40, feature_dim=BELIEF_FEATURE_DIM, binary_features=True,
    ))
    bundle = tmp_path / "belief_bundle"
    materialise_belief(iter(src), bundle)

    config_path = tmp_path / "belief_stream.yaml"
    config_path.write_text(yaml.safe_dump(_stream_config(bundle)), encoding="utf-8")
    run_dir = tmp_path / "run_stream"
    rc = train_belief_main(["--config", str(config_path), "--run-dir", str(run_dir)])
    assert rc == 0
    assert (run_dir / "checkpoints" / "belief_final.bin").exists()

    import io

    import torch
    state = torch.load(
        io.BytesIO(
            Checkpoint.load(
                run_dir / "checkpoints" / "belief_final.bin",
                expected_featurizer_version=FEATURIZER_VERSION,
            ).payload
        ),
        weights_only=True,
    )["model"]
    assert state["fc1.weight"].shape[1] == BELIEF_FEATURE_DIM


def test_legacy_tier_key_is_rejected_not_ignored(tmp_path):
    """A leftover `tier:` from the ADR-0028 ablation must fail loudly rather
    than silently training on a mis-sliced prefix."""
    import pytest

    src = list(SyntheticBeliefDataset(
        seed=0, n_examples=8, feature_dim=BELIEF_FEATURE_DIM, binary_features=True,
    ))
    bundle = tmp_path / "belief_bundle"
    materialise_belief(iter(src), bundle)

    config = _stream_config(bundle) | {"tier": "B_core"}
    config_path = tmp_path / "belief_legacy.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="tier"):
        train_belief_main([
            "--config", str(config_path), "--run-dir", str(tmp_path / "run_legacy"),
        ])
