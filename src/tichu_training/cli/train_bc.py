"""`train_bc` CLI.

Reads a YAML config, instantiates a dataset + model + optimizer, runs the
training loop, and writes checkpoints + a `step.csv` log into the run
directory. Resumable via `--resume`. AWR offline refinement (#012) is
activated by `--refine-from <bc_ckpt>`.
"""

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from tichu_training.awr.refine import awr_refine_epoch
from tichu_training.awr.value_baseline import ValueBaseline, fit_value_baseline
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
                   help="BC checkpoint to refine with AWR")
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
    if args.refine_from:
        start_step = load_checkpoint(args.refine_from, model)
        log.info("loaded BC checkpoint for AWR refinement: %s", args.refine_from)

    log_path = run_dir / "step.csv"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_every = int(config.get("checkpoint_every", 1000))
    head_weights = {k: float(v) for k, v in config.get("head_weights", {}).items()}

    if args.refine_from:
        return _run_awr_refinement(
            model, optimizer, dataset_examples, config, run_dir,
            log_path=log_path, ckpt_dir=ckpt_dir,
            head_weights=head_weights, start_step=start_step,
        )

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


def _run_awr_refinement(
    model, optimizer, dataset_examples, config, run_dir,
    *, log_path, ckpt_dir, head_weights, start_step,
):
    awr_cfg = config.get("awr") or {}
    beta = float(awr_cfg.get("beta", 1.0))
    max_weight = float(awr_cfg.get("max_weight", 20.0))
    held_out_fraction = float(awr_cfg.get("held_out_fraction", 0.1))

    feature_dim = dataset_examples[0].features.shape[0]
    baseline = ValueBaseline(
        feature_dim=feature_dim,
        hidden=int(awr_cfg.get("baseline_hidden", 128)),
    )

    split = max(1, int((1.0 - held_out_fraction) * len(dataset_examples)))
    train_set = dataset_examples[:split]
    held_out = dataset_examples[split:] or None

    features = np.stack([e.features for e in train_set])
    outcomes = np.array([e.round_outcome for e in train_set], dtype=np.float32)
    baseline_mse = fit_value_baseline(
        baseline, features, outcomes,
        batch_size=int(config["batch_size"]),
        epochs=int(awr_cfg.get("baseline_epochs", 3)),
        lr=float(awr_cfg.get("baseline_lr", 1.0e-3)),
    )
    log.info("baseline MSE after fit: %.4f", baseline_mse)

    epoch_log_path = run_dir / "epoch.csv"
    new_epoch_log = not epoch_log_path.exists()
    epoch_fields = ["epoch", "loss_total", "avg_weight", "win_rate_proxy"]

    step = start_step
    with epoch_log_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=epoch_fields)
        if new_epoch_log:
            writer.writeheader()
        for epoch in range(int(config["epochs"])):
            summary = awr_refine_epoch(
                model, train_set, baseline, optimizer,
                beta=beta, max_weight=max_weight,
                batch_size=int(config["batch_size"]),
                log_path=log_path,
                held_out_subset=held_out,
                head_weights=head_weights or None,
            )
            log.info(
                "[awr] epoch %d loss=%.4f avg_weight=%.3f win_rate_proxy=%s",
                epoch, summary["loss_total"], summary["avg_weight"],
                "n/a" if summary["win_rate_proxy"] is None else f"{summary['win_rate_proxy']:.3f}",
            )
            writer.writerow({
                "epoch": epoch,
                "loss_total": summary["loss_total"],
                "avg_weight": summary["avg_weight"],
                "win_rate_proxy": "" if summary["win_rate_proxy"] is None
                                  else summary["win_rate_proxy"],
            })
            step += 1

    ckpt_path = ckpt_dir / f"awr_final_step_{step:06d}.bin"
    save_checkpoint(model, optimizer, step=step, path=ckpt_path)
    log.info("saved AWR checkpoint %s", ckpt_path)
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
