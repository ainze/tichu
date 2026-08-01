"""`train_belief` CLI — opponent-hand prediction model.

Belief checkpoints live separately from policy checkpoints and are NOT
loaded by the inference service in phase 1. The model is a phase-2
enabler for ISMCTS-style search.
"""

import argparse
import csv
import io
import logging
import shutil
import sys
from pathlib import Path

import torch
import yaml

from tichu_training.belief.belief_materialised import MemmapBeliefDataset
from tichu_training.belief.dataset import SyntheticBeliefDataset
from tichu_training.belief.model import BeliefModel, belief_accuracy, belief_loss
from tichu_training.belief.training import (
    train_one_belief_epoch,
    write_calibration_csv,
)
from tichu_training.checkpoint import Checkpoint
from tichu_training.cli._dataset_build import build_memmap_dataset
from tichu_training.featurizer import FEATURIZER_VERSION


log = logging.getLogger("train_belief")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Train the belief model (opponent hand prediction)."
    )
    p.add_argument("--config", required=True, metavar="FILE", help="YAML config file")
    p.add_argument("--run-dir", required=True, metavar="DIR")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_dir / config_path.name)

    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    _reject_legacy_tier(config)

    if config.get("streaming"):
        return _run_streaming(config, run_dir)

    examples = _build_dataset(config)
    feature_dim = examples[0].features.shape[0]
    m_cfg = config.get("model", {})
    model = BeliefModel(feature_dim=feature_dim, hidden=int(m_cfg.get("hidden", 256)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    step_log = run_dir / "step.csv"
    epoch_log = run_dir / "epoch.csv"
    epoch_fields = ["epoch", "loss", "accuracy"]
    new_epoch_log = not epoch_log.exists()

    with epoch_log.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=epoch_fields)
        if new_epoch_log:
            writer.writeheader()
        for epoch in range(int(config["epochs"])):
            loss = train_one_belief_epoch(
                model, examples, optimizer,
                batch_size=int(config["batch_size"]),
                log_path=step_log,
            )
            acc = _epoch_accuracy(model, examples)
            log.info("epoch %d loss=%.4f acc=%.4f", epoch, loss, acc)
            writer.writerow({"epoch": epoch, "loss": loss, "accuracy": acc})

    write_calibration_csv(model, examples, run_dir / "calibration.csv")

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _save_belief_checkpoint(model, ckpt_dir / "belief_final.bin")
    log.info("saved belief checkpoint to %s", ckpt_dir / "belief_final.bin")
    return 0


def _run_streaming(config, run_dir: Path) -> int:
    """Low-RAM scale path: stream batched arrays off the memmap bundle (no
    full-list materialisation), at the bundle's own feature width. Required at
    10k+ games where the in-memory list would OOM. Skips the (full-stack)
    calibration table — use scripts/belief_naive_baseline.py for held-out
    metrics."""
    from tichu_training.belief.belief_materialised import MemmapBeliefDataset
    from tichu_training.belief.training import (
        stream_accuracy,
        train_one_belief_epoch_stream,
    )

    if config.get("dataset") != "memmap":
        raise ValueError("streaming requires dataset: memmap")
    ds = MemmapBeliefDataset(config["dataset_kwargs"]["data_dir"])
    log.info("streaming belief: %s examples, feature_dim=%d, batch=%s",
             f"{len(ds):,}", ds.feature_dim, config["batch_size"])

    m_cfg = config.get("model", {})
    model = BeliefModel(feature_dim=ds.feature_dim, hidden=int(m_cfg.get("hidden", 256)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    step_log = run_dir / "step.csv"
    epoch_log = run_dir / "epoch.csv"
    new_epoch_log = not epoch_log.exists()
    with epoch_log.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "loss", "accuracy"])
        if new_epoch_log:
            writer.writeheader()
        for epoch in range(int(config["epochs"])):
            loss = train_one_belief_epoch_stream(
                model, ds, optimizer,
                batch_size=int(config["batch_size"]),
                log_path=step_log,
            )
            acc = stream_accuracy(model, ds, batch_size=int(config["batch_size"]))
            log.info("epoch %d loss=%.4f acc=%.4f", epoch, loss, acc)
            writer.writerow({"epoch": epoch, "loss": loss, "accuracy": acc})

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _save_belief_checkpoint(model, ckpt_dir / "belief_final.bin")
    log.info("saved belief checkpoint to %s", ckpt_dir / "belief_final.bin")
    return 0


def _reject_legacy_tier(config) -> None:
    """The ADR-0028 A / B_core / B_full tiers are gone: the ablation settled on
    B-core, and featurizer v6 (ADR-0038) folded B-core's channels into the
    policy Feature Vector itself, so the belief input is now just that vector.
    A leftover `tier:` key would otherwise be silently ignored — fail instead."""
    if "tier" in config:
        raise ValueError(
            f"config sets tier: {config['tier']!r}, but the ADR-0028 belief-input "
            "tiers no longer exist — v6 absorbed the B-core history channels "
            "(ADR-0038) and the belief input is the full policy Feature Vector. "
            "Drop the key."
        )


def _build_dataset(config):
    name = config["dataset"]
    kwargs = dict(config.get("dataset_kwargs", {}))
    if name == "synthetic":
        return list(SyntheticBeliefDataset(**kwargs))
    if name == "memmap":
        # Read the packed belief bundle (ADR-0021) — the first real-data path
        # for belief. force_materialize: the training loop stacks the whole
        # example list per epoch (no iter_batches fast path here).
        return build_memmap_dataset(
            MemmapBeliefDataset, kwargs, force_materialize=True,
        )
    raise ValueError(f"unknown dataset: {name!r}")


def _epoch_accuracy(model: BeliefModel, examples) -> float:
    if not examples:
        return 0.0
    import numpy as np
    features = torch.from_numpy(np.stack([e.features for e in examples]))
    labels = torch.from_numpy(np.stack([e.labels for e in examples]))
    mask = torch.from_numpy(np.stack([e.mask for e in examples]))
    with torch.no_grad():
        logits = model(features)
    return belief_accuracy(logits, labels, mask)


def _save_belief_checkpoint(model: BeliefModel, path: Path) -> None:
    buf = io.BytesIO()
    torch.save({"model": model.state_dict()}, buf)
    Checkpoint(
        featurizer_version=FEATURIZER_VERSION,
        action_space_version="",  # belief does not predict actions.
        payload=buf.getvalue(),
    ).save(path)


if __name__ == "__main__":
    sys.exit(main())
