"""`train_schupfen` CLI — trains the standalone Schupfen Network.

Mirrors `train_calls` (single-network shape — schupfen has no
grand/tichu split). Smoke path consumes `SyntheticSchupfenDataset`;
production path consumes `ParquetSchupfenDataset` (play-shard manifest
+ archive streaming). Output is a `Checkpoint` with the live
`FEATURIZER_VERSION` / `ACTION_SPACE_VERSION` stamped (ADR-0012 §3).
"""

import argparse
import csv
import logging
import random
import shutil
import sys
from pathlib import Path

import torch
import yaml

from tichu_training.bc.schupfen_materialised import MemmapSchupfenDataset
from tichu_training.bc.schupfen_model import SchupfenNetwork
from tichu_training.bc.schupfen_training import (
    ParquetSchupfenDataset,
    SyntheticSchupfenDataset,
    train_one_schupfen_epoch,
    train_one_schupfen_epoch_batched,
)
from tichu_training.bc.training import save_checkpoint
from tichu_training.cli._dataset_build import build_memmap_dataset


log = logging.getLogger("train_schupfen")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the standalone Schupfen Network.")
    p.add_argument("--config", required=True, metavar="FILE", help="YAML config file")
    p.add_argument("--run-dir", required=True, metavar="DIR")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="Where to run the network. `auto` picks cuda if "
                        "torch.cuda.is_available() else cpu. (default: auto)")
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
    device = _resolve_device(args.device)
    log.info("using device: %s", device)

    examples = _build_dataset(config)
    feature_dim = _peek_feature_dim(examples)
    m = config.get("model", {})
    net = SchupfenNetwork(
        feature_dim=feature_dim,
        skill_buckets=10,
        skill_dim=int(m.get("skill_dim", 64)),
        hidden=int(m.get("hidden", 256)),
        depth=int(m.get("depth", 4)),
        residual=bool(m.get("residual", False)),
    ).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(config["learning_rate"]))
    log_path = run_dir / "step.csv"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Per-epoch shuffle of the training pass. The materialised bundle is in
    # game_id order with a recency sample_weight that flips 0.5 -> 1.0 at the
    # cutoff; an unshuffled pass crosses that boundary in the same place every
    # epoch. Opt-in (full-corpus configs set it).
    shuffle = bool(config.get("shuffle", False))
    n_epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    # Game-level held-out fraction (online estimate, folded into the train
    # sweep). 0 keeps existing configs unchanged; full-corpus configs opt in.
    val_frac = float(config.get("val_frac", 0.0))
    val_log_path = run_dir / "val.csv"

    if hasattr(examples, "iter_batches") and not isinstance(examples, list):
        # Streaming memmap fast path — never materialise the slice (the
        # full-corpus schupfen slice is ~88M rows / ~100 GB as objects).
        if shuffle:
            log.info("per-epoch streaming shuffle enabled (seed=%d)", seed)
        if val_frac > 0:
            log.info("streaming %d rows, val_frac=%.3f (game_id hash split)",
                     len(examples), val_frac)
        total_batches = (len(examples) + batch_size - 1) // batch_size
        for epoch in range(n_epochs):
            batches = examples.iter_batches(
                batch_size, shuffle=shuffle, seed=seed + epoch,
            )
            loss, metrics = train_one_schupfen_epoch_batched(
                net, batches, optimizer,
                log_path=log_path,
                desc=f"epoch {epoch + 1}/{n_epochs}",
                total_batches=total_batches,
                val_frac=val_frac, val_seed=seed,
            )
            log.info("epoch %d final batch loss=%.4f", epoch, loss)
            if metrics is not None:
                log.info("epoch %d val: NLL=%.4f acc=%.4f (n=%d)",
                         epoch, metrics["loss"], metrics["accuracy"], int(metrics["n"]))
                _append_schupfen_val_row(val_log_path, epoch, metrics)
    else:
        # Legacy materialised path (synthetic / parquet smoke).
        rate_examples = examples if isinstance(examples, list) else list(examples)
        if shuffle:
            log.info("per-epoch training shuffle enabled (seed=%d)", seed)
        for epoch in range(n_epochs):
            if shuffle:
                random.Random(seed + epoch).shuffle(rate_examples)
            loss = train_one_schupfen_epoch(
                net, rate_examples, optimizer,
                batch_size=batch_size,
                log_path=log_path,
                desc=f"epoch {epoch + 1}/{n_epochs}",
            )
            log.info("epoch %d final batch loss=%.4f", epoch, loss)

    save_checkpoint(net, optimizer, step=n_epochs,
                    path=ckpt_dir / "schupfen_final.bin")
    log.info("saved final checkpoint")
    return 0


def _append_schupfen_val_row(path: Path, epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "n", "loss", "accuracy"])
        if new_file:
            writer.writeheader()
        writer.writerow({
            "epoch": epoch, "n": int(metrics["n"]),
            "loss": metrics["loss"], "accuracy": metrics["accuracy"],
        })


def _resolve_device(choice: str) -> torch.device:
    """Resolve `--device` to a torch.device. `auto` picks cuda when available
    else cpu; explicit `cuda` raises if unavailable. The schupfen net is a
    small MLP — historically CPU-only — so `auto` only moves to GPU when one
    is present."""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested but torch.cuda.is_available() is False"
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_dataset(config):
    name = config["dataset"]
    kwargs = dict(config.get("dataset_kwargs", {}))
    if name == "synthetic":
        return list(SyntheticSchupfenDataset(**kwargs))
    if name == "parquet":
        materialize = bool(kwargs.pop("materialize", False))
        ds = ParquetSchupfenDataset(**kwargs)
        return list(ds) if materialize else ds
    if name == "memmap":
        # Read the packed schupfen bundle (ADR-0020) instead of re-parsing the
        # archive. Never materialise — the trainer streams via iter_batches, so
        # `materialize` (if present) is dropped and the re-iterable dataset is
        # returned for the streaming fast path.
        kwargs.pop("materialize", None)
        return build_memmap_dataset(MemmapSchupfenDataset, kwargs)
    raise ValueError(f"unknown dataset: {name!r}")


def _peek_feature_dim(examples) -> int:
    if isinstance(examples, list):
        return int(examples[0].features.shape[0])
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    return FEATURIZER_OUTPUT_DIM


if __name__ == "__main__":
    sys.exit(main())
