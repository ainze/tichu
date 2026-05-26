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
from tqdm import tqdm

from tichu_training.awr.refine import awr_refine_epoch
from tichu_training.awr.streaming import (
    awr_refine_epoch_streaming,
    collect_held_out,
    fit_value_baseline_streaming,
)
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
    p.add_argument("--max-examples", type=int, default=None, metavar="N",
                   help="Cap each epoch to the first N examples yielded by the "
                        "dataset. Useful for fast smoke runs against the full "
                        "100k/2.4M parquet manifest without committing to a "
                        "full sweep. Default: no cap.")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="Where to run the model. `auto` picks cuda if "
                        "torch.cuda.is_available() else cpu. (default: auto)")
    p.add_argument("--workers", type=int, default=0, metavar="N",
                   help="Number of data-loader worker processes. 0 = "
                        "single-threaded (the legacy path). N>=1 spawns "
                        "ParallelParquetBCDataset workers, each handling "
                        "a hash-sharded slice of the manifest. Default 0. "
                        "Suggested for corpus runs: os.cpu_count() - 1.")
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

    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    dataset = _build_dataset(config, num_workers=args.workers)
    # Feature dim comes from the featurizer constant — no need to peek at the
    # first example, which would force materialisation of a streaming dataset
    # before training starts. SyntheticBCDataset uses a configurable smaller
    # dim for smoke; fall back to the constant when it's not overridden.
    if config["dataset"] == "synthetic":
        feature_dim = int(config.get("dataset_kwargs", {}).get(
            "feature_dim", FEATURIZER_OUTPUT_DIM,
        ))
    else:
        feature_dim = FEATURIZER_OUTPUT_DIM
    model = _build_model(feature_dim, config)
    device = _resolve_device(args.device)
    log.info("using device: %s", device)
    model = model.to(device)
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
    # Mid-epoch checkpointing: save every N fired batches inside an epoch.
    # 0 disables, matching the legacy "only at epoch boundary" behaviour.
    # Crucial for full-corpus epochs that take hours — otherwise AWR refine
    # and --resume have no recent target.
    checkpoint_every_batches = int(config.get("checkpoint_every_batches", 0))
    head_weights = {k: float(v) for k, v in config.get("head_weights", {}).items()}

    if args.refine_from:
        awr_cfg = config.get("awr") or {}
        if awr_cfg.get("streaming", False):
            # Streaming AWR: never materialise the full dataset. The
            # dataset is iterated fresh once per AWR epoch (plus one
            # held-out scan + one baseline-fit pass). Required for the
            # full 2.4M-game / ~1.5B-decision corpus where the
            # materialised version would need ~100 TB of RAM.
            return _run_awr_refinement_streaming(
                model, optimizer, dataset, args, config, run_dir,
                log_path=log_path, ckpt_dir=ckpt_dir,
                head_weights=head_weights, start_step=start_step,
            )

        # In-memory AWR: drain the iterable once and pass the list
        # through. Respect --max-examples — caller may want a quick AWR
        # smoke without committing to the full sweep.
        refine_iter = (
            _capped(dataset, args.max_examples)
            if args.max_examples is not None else dataset
        )
        # The drain takes minutes on real parquet — show a bar so the
        # user knows the run hasn't hung. n_rows is set on both
        # _CappedIterable and the parquet datasets.
        total = getattr(refine_iter, "n_rows", None)
        dataset_examples = list(tqdm(
            refine_iter, total=total, unit="ex", dynamic_ncols=True,
            desc="awr materialise",
        ))
        log.info("materialised %d examples for AWR refinement", len(dataset_examples))
        return _run_awr_refinement(
            model, optimizer, dataset_examples, config, run_dir,
            log_path=log_path, ckpt_dir=ckpt_dir,
            head_weights=head_weights, start_step=start_step,
        )

    step = start_step
    for epoch in range(int(config["epochs"])):
        # Cap the per-epoch example count if requested. islice gives a
        # bounded iterable that preserves the streaming nature of the
        # underlying dataset — no materialisation up front.
        epoch_iter = (
            _capped(dataset, args.max_examples)
            if args.max_examples is not None else dataset
        )
        def _save_mid_epoch(batch_step: int) -> None:
            ckpt_path = ckpt_dir / f"step_e{epoch:03d}_b{batch_step:08d}.bin"
            save_checkpoint(model, optimizer, step=batch_step, path=ckpt_path)
            log.info("saved mid-epoch checkpoint %s", ckpt_path)

        loss = train_one_epoch(
            model,
            epoch_iter,
            optimizer,
            batch_size=int(config["batch_size"]),
            log_path=log_path,
            head_weights=head_weights or None,
            checkpoint_every_batches=checkpoint_every_batches,
            checkpoint_fn=_save_mid_epoch if checkpoint_every_batches else None,
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

    # Stack features once and share with both fit_value_baseline and the
    # per-epoch awr_refine_epoch calls — re-stacking is the dominant
    # memory hit on multi-million-row training sets.
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
                features=features,
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


def _run_awr_refinement_streaming(
    model, optimizer, dataset, args, config, run_dir,
    *, log_path, ckpt_dir, head_weights, start_step,
):
    """Full-corpus AWR refinement that never materialises the dataset.

    The dataset is iterated fresh once per phase:
      * held-out scan (stops early once `max_held_out` is hit)
      * single baseline-fit pass
      * one fresh pass per AWR epoch

    With ParallelParquetBCDataset each pass spawns its worker pool
    again. Adds tens of seconds of startup per pass; for the full
    corpus that's negligible against the multi-hour stream.
    """
    awr_cfg = config.get("awr") or {}
    beta = float(awr_cfg.get("beta", 1.0))
    max_weight = float(awr_cfg.get("max_weight", 20.0))
    held_out_fraction = float(awr_cfg.get("held_out_fraction", 0.05))
    max_held_out = int(awr_cfg.get("max_held_out", 20_000))
    chunk_size = int(awr_cfg.get("chunk_size", 16_384))

    # Each call returns a fresh iterator. iter() on a list, ParquetBCDataset,
    # or ParallelParquetBCDataset all yield a new pass.
    def dataset_factory():
        capped = (
            _capped(dataset, args.max_examples)
            if args.max_examples is not None else dataset
        )
        return iter(capped)

    log.info("[awr/stream] phase 1: scanning for held-out subset "
             "(fraction=%.3f, cap=%d)", held_out_fraction, max_held_out)
    held_out, held_out_filter = collect_held_out(
        dataset_factory,
        fraction=held_out_fraction,
        max_held_out=max_held_out,
    )
    log.info("[awr/stream] held-out subset: %d examples", len(held_out))
    if not held_out:
        log.warning("[awr/stream] held-out subset is empty; win_rate_proxy "
                    "will be n/a for every epoch")

    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    feature_dim = (
        held_out[0].features.shape[0] if held_out else FEATURIZER_OUTPUT_DIM
    )
    baseline = ValueBaseline(
        feature_dim=feature_dim,
        hidden=int(awr_cfg.get("baseline_hidden", 128)),
    )

    log.info("[awr/stream] phase 2: fitting value baseline by stream")
    baseline_mse = fit_value_baseline_streaming(
        baseline, dataset_factory,
        batch_size=int(config["batch_size"]),
        lr=float(awr_cfg.get("baseline_lr", 1.0e-3)),
        chunk_size=chunk_size,
        sgd_steps_per_chunk=int(awr_cfg.get("baseline_sgd_steps_per_chunk", 3)),
        held_out_filter=held_out_filter,
    )
    log.info("[awr/stream] baseline MSE after fit: %.4f", baseline_mse)

    epoch_log_path = run_dir / "epoch.csv"
    new_epoch_log = not epoch_log_path.exists()
    epoch_fields = ["epoch", "loss_total", "avg_weight", "win_rate_proxy"]

    step = start_step
    with epoch_log_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=epoch_fields)
        if new_epoch_log:
            writer.writeheader()
        for epoch in range(int(config["epochs"])):
            log.info("[awr/stream] phase 3.%d: refine epoch", epoch)
            summary = awr_refine_epoch_streaming(
                model, dataset_factory, baseline, optimizer,
                beta=beta, max_weight=max_weight,
                batch_size=int(config["batch_size"]),
                log_path=log_path,
                head_weights=head_weights or None,
                held_out_examples=held_out or None,
                held_out_filter=held_out_filter,
                chunk_size=chunk_size,
            )
            log.info(
                "[awr/stream] epoch %d loss=%.4f avg_weight=%.3f win_rate_proxy=%s",
                epoch, summary["loss_total"], summary["avg_weight"],
                "n/a" if summary["win_rate_proxy"] is None
                else f"{summary['win_rate_proxy']:.3f}",
            )
            writer.writerow({
                "epoch": epoch,
                "loss_total": summary["loss_total"],
                "avg_weight": summary["avg_weight"],
                "win_rate_proxy": "" if summary["win_rate_proxy"] is None
                                  else summary["win_rate_proxy"],
            })
            step += 1
            # Per-epoch checkpoint so a long full-corpus run can be
            # resumed without retraining from scratch on a crash.
            ckpt_path = ckpt_dir / f"awr_epoch_{epoch:03d}_step_{step:06d}.bin"
            save_checkpoint(model, optimizer, step=step, path=ckpt_path)
            log.info("[awr/stream] saved epoch checkpoint %s", ckpt_path)

    ckpt_path = ckpt_dir / f"awr_final_step_{step:06d}.bin"
    save_checkpoint(model, optimizer, step=step, path=ckpt_path)
    log.info("saved AWR checkpoint %s", ckpt_path)
    return 0


def _resolve_device(choice: str) -> torch.device:
    """Resolve the `--device` flag to a torch.device.

    `auto` picks `cuda` when `torch.cuda.is_available()`, else `cpu`.
    Explicit `cuda` raises if CUDA isn't available, so silent CPU
    fallback never confuses a user who asked for GPU.
    """
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested but torch.cuda.is_available() is False"
            )
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _CappedIterable:
    """Bound an iterable to its first N items while preserving `n_rows`
    so the trainer's tqdm bar shows the right total. `n_rows` is the
    minimum of the underlying value (when present) and the cap.
    """

    def __init__(self, inner, cap: int) -> None:
        self._inner = inner
        self._cap = int(cap)
        underlying_n_rows = getattr(inner, "n_rows", None)
        self.n_rows = (
            min(underlying_n_rows, self._cap)
            if underlying_n_rows is not None else self._cap
        )

    def __iter__(self):
        from itertools import islice
        return islice(iter(self._inner), self._cap)


def _capped(inner, cap: int):
    return _CappedIterable(inner, cap)


def _build_dataset(config, *, num_workers: int = 0):
    """Construct the BC training dataset.

    `synthetic`: in-memory deterministic examples for smoke. Materialised
        as a list (small).
    `parquet`: ADR-0011 archive-driven replay-on-the-fly streaming
        dataset. With `num_workers == 0` (default), the sequential
        `ParquetBCDataset` is used. With `num_workers >= 1`, the
        `ParallelParquetBCDataset` shards the manifest across that many
        worker processes — typical speedup is near-linear on CPU-bound
        runs since the Python data-loading loop releases nothing to
        MKL's threadpool.
    """
    name = config["dataset"]
    kwargs = dict(config.get("dataset_kwargs", {}))
    if name == "synthetic":
        return list(SyntheticBCDataset(**kwargs))
    if name == "parquet":
        if num_workers >= 1:
            from tichu_training.bc.parallel_dataset import ParallelParquetBCDataset
            return ParallelParquetBCDataset(num_workers=num_workers, **kwargs)
        return ParquetBCDataset(**kwargs)
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
