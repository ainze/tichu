"""`train_bc` CLI.

Reads a YAML config, instantiates a dataset + model + optimizer, runs the
training loop, and writes checkpoints + a `step.csv` log into the run
directory. Resumable via `--resume`.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import torch
import yaml

from tichu_training.bc.dataset import ParquetBCDataset, SyntheticBCDataset
from tichu_training.bc.heads import BCModel
from tichu_training.bc.training import (
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


log = logging.getLogger("train_bc")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the behavioral cloning policy.")
    p.add_argument("--config", required=True, metavar="FILE", help="YAML config file")
    p.add_argument("--run-dir", required=True, metavar="DIR",
                   help="Directory to write checkpoints, logs, and config copy")
    p.add_argument("--resume", metavar="CHECKPOINT", help="Checkpoint to resume from")
    p.add_argument("--refine-from", metavar="CHECKPOINT",
                   help="BC checkpoint to refine with AWR (reserved for #012)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_dir / config_path.name)
    log.info("copied config to %s", run_dir / config_path.name)

    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)

    dataset_examples = _build_dataset(config)
    feature_dim = dataset_examples[0].features.shape[0]
    model = _build_model(feature_dim, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer)
        log.info("resumed from step %d", start_step)

    log_path = run_dir / "step.csv"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_every = int(config.get("checkpoint_every", 1000))
    head_weights = {k: float(v) for k, v in config.get("head_weights", {}).items()}

    step = start_step
    for epoch in range(int(config["epochs"])):
        loss = train_one_epoch(
            model,
            dataset_examples,
            optimizer,
            batch_size=int(config["batch_size"]),
            log_path=log_path,
            head_weights=head_weights or None,
        )
        log.info("epoch %d final batch loss=%.4f", epoch, loss)
        # Conservative checkpoint cadence: one per epoch is fine for smoke.
        step += 1
        if step % max(1, checkpoint_every) == 0 or epoch == int(config["epochs"]) - 1:
            ckpt_path = ckpt_dir / f"step_{step:06d}.bin"
            save_checkpoint(model, optimizer, step=step, path=ckpt_path)
            log.info("saved checkpoint %s", ckpt_path)

    return 0


def _build_dataset(config):
    name = config["dataset"]
    kwargs = dict(config.get("dataset_kwargs", {}))
    if name == "synthetic":
        return list(SyntheticBCDataset(**kwargs))
    if name == "parquet":
        ds = ParquetBCDataset(**kwargs)
        # Parquet → BCExample materialisation is a follow-up; refuse for now.
        raise NotImplementedError(
            "ParquetBCDataset example materialisation is deferred to a "
            "featurization-cache follow-up; use the synthetic dataset for "
            "the smoke test and for early experiments."
        )
    raise ValueError(f"unknown dataset: {name!r}")


def _build_model(feature_dim: int, config) -> BCModel:
    m = config.get("model", {})
    return BCModel(
        feature_dim=feature_dim,
        skill_buckets=10,
        skill_dim=int(m.get("skill_dim", 64)),
        trunk_hidden=int(m.get("trunk_hidden", 1024)),
        trunk_depth=int(m.get("trunk_depth", 4)),
        trunk_out_dim=int(m.get("trunk_out_dim", 512)),
        head_hidden=int(m.get("head_hidden", 256)),
    )


if __name__ == "__main__":
    sys.exit(main())
