"""`train_calls` CLI — trains the Tichu and Grand Tichu call networks."""

import argparse
import csv
import logging
import random
import shutil
import sys
from pathlib import Path

import torch
import yaml

from tichu_training.bc.call_materialised import MemmapCallDataset
from tichu_training.bc.call_model import GrandTichuCallNetwork, TichuCallNetwork
from tichu_training.bc.call_training import (
    ParquetCallDataset,
    SyntheticCallDataset,
    evaluate_call_examples,
    split_examples_by_game,
    train_one_call_epoch,
    train_one_call_epoch_batched,
    write_calling_rate_csv,
    write_calling_rate_rows,
)
from tichu_training.bc.training import save_checkpoint
from tichu_training.cli._dataset_build import build_memmap_dataset


log = logging.getLogger("train_calls")

_NETWORKS = {
    "grand": GrandTichuCallNetwork,
    "tichu": TichuCallNetwork,
}

# CLI tag → ParquetCallDataset call_type. The CLI uses short tags
# ("grand", "tichu") for filenames; the dataset uses the canonical
# Decision names from CONTEXT.md.
_TAG_TO_CALL_TYPE: dict[str, str] = {
    "grand": "grand_tichu",
    "tichu": "tichu",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train Tichu and Grand Tichu call networks.")
    p.add_argument("--config", required=True, metavar="FILE", help="YAML config file")
    p.add_argument("--run-dir", required=True, metavar="DIR")
    p.add_argument("--only", choices=("grand", "tichu"),
                   help="Train only one of the two networks (default: both)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="Where to run the networks. `auto` picks cuda if "
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

    # Default 0 keeps existing smoke configs (which lack a `val_frac` key)
    # working unchanged. Production configs opt in explicitly.
    val_frac = float(config.get("val_frac", 0.0))
    # Per-epoch shuffle of the training pass. The materialised bundle is in
    # game_id order with a recency sample_weight that flips 0.5 -> 1.0 at the
    # cutoff, so an unshuffled pass crosses that boundary in the same place
    # every epoch (doubling the weighted loss + effective LR on the tail).
    # Opt-in for parity with the BC config; full-corpus configs set it.
    shuffle = bool(config.get("shuffle", False))
    targets = ["grand", "tichu"] if args.only is None else [args.only]
    for tag in targets:
        examples = _build_dataset(config, _TAG_TO_CALL_TYPE[tag])
        feature_dim = _peek_feature_dim(examples)
        net_cls = _NETWORKS[tag]
        m = config.get("model", {})
        net = net_cls(
            feature_dim=feature_dim,
            skill_buckets=10,
            skill_dim=int(m.get("skill_dim", 64)),
            hidden=int(m.get("hidden", 256)),
            depth=int(m.get("depth", 4)),
            residual=bool(m.get("residual", False)),
        ).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=float(config["learning_rate"]))
        log_path = run_dir / f"{tag}_step.csv"
        val_log_path = run_dir / f"{tag}_val.csv"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        n_epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])

        def _log_val(epoch: int, metrics: dict) -> None:
            if val_frac <= 0:
                return
            log.info(
                "[%s] epoch %d val: loss=%.4f acc=%.4f auc=%.4f "
                "(n=%d, pos_frac=%.4f)",
                tag, epoch, metrics["loss"], metrics["accuracy"],
                metrics["auc"], int(metrics["n"]), metrics["pos_frac"],
            )
            _append_val_row(val_log_path, epoch, metrics)

        rate_path = lambda e: run_dir / f"{tag}_calling_rate_by_decile_epoch{e}.csv"

        if hasattr(examples, "iter_batches") and not isinstance(examples, list):
            # Streaming memmap fast path — never materialise the slice (the
            # full-corpus calls slice is ~88M rows / ~100 GB as objects). The
            # game-level val holdout is carved per-batch by a game_id hash
            # (fixed across epochs) instead of an up-front list split.
            if shuffle:
                log.info("[%s] per-epoch streaming shuffle enabled (seed=%d)", tag, seed)
            log.info("[%s] streaming %d rows, val_frac=%.3f (game_id hash split)",
                     tag, len(examples), val_frac)
            total_batches = (len(examples) + batch_size - 1) // batch_size
            for epoch in range(n_epochs):
                train_batches = examples.iter_batches(
                    batch_size, shuffle=shuffle, seed=seed + epoch,
                )
                # Val metrics + calling-rate are folded into this single sweep
                # (held-out rows forwarded no-grad) — no second pass over the
                # ~88M-row slice. Val is an online estimate; see the function.
                loss, metrics, rates, decile_counts = train_one_call_epoch_batched(
                    net, train_batches, optimizer, log_path=log_path,
                    desc=f"[{tag}] epoch {epoch + 1}/{n_epochs}",
                    total_batches=total_batches, val_frac=val_frac, val_seed=seed,
                )
                log.info("[%s] epoch %d final batch loss=%.4f", tag, epoch, loss)
                _log_val(epoch, metrics)
                write_calling_rate_rows(rate_path(epoch), rates, decile_counts)
        else:
            # Legacy materialised path (synthetic / parquet smoke). Split by
            # game_id so a game's rows never straddle the train/val boundary.
            all_examples = examples if isinstance(examples, list) else list(examples)
            train_examples, val_examples = split_examples_by_game(
                all_examples, val_frac=val_frac, seed=seed,
            )
            log.info(
                "[%s] split: train=%d val=%d (val_frac=%.3f)",
                tag, len(train_examples), len(val_examples), val_frac,
            )
            if val_examples:
                pos = sum(1 for e in val_examples if e.target == 1)
                log.info("[%s] val positive fraction: %.4f (%d / %d)",
                         tag, pos / len(val_examples), pos, len(val_examples))
            if shuffle:
                log.info("[%s] per-epoch training shuffle enabled (seed=%d)", tag, seed)
            for epoch in range(n_epochs):
                if shuffle:
                    random.Random(seed + epoch).shuffle(train_examples)
                loss = train_one_call_epoch(
                    net, train_examples, optimizer,
                    batch_size=batch_size,
                    log_path=log_path,
                    desc=f"[{tag}] epoch {epoch + 1}/{n_epochs}",
                )
                log.info("[%s] epoch %d final batch loss=%.4f", tag, epoch, loss)
                if val_examples:
                    _log_val(epoch, evaluate_call_examples(
                        net, val_examples, batch_size=batch_size,
                    ))
                write_calling_rate_csv(
                    net, train_examples, path=rate_path(epoch),
                )
        save_checkpoint(net, optimizer, step=n_epochs,
                        path=ckpt_dir / f"{tag}_final.bin")
        log.info("[%s] saved final checkpoint", tag)

    return 0


def _resolve_device(choice: str) -> torch.device:
    """Resolve `--device` to a torch.device. `auto` picks cuda when available
    else cpu; explicit `cuda` raises if unavailable (no silent CPU fallback).
    The call/schupfen nets are small MLPs — historically CPU-only — so `auto`
    only moves to GPU when one is present."""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested but torch.cuda.is_available() is False"
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _append_val_row(path: Path, epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    fields = ["epoch", "n", "loss", "accuracy", "auc", "pos_frac"]
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "epoch": epoch,
            "n": int(metrics["n"]),
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "auc": metrics["auc"],
            "pos_frac": metrics["pos_frac"],
        })


def _build_dataset(config, call_type: str):
    """Construct the per-tag dataset. Synthetic returns a list; parquet and
    memmap return a re-iterable dataset (or a materialised list if
    `materialize: true` is set).

    The memmap path reads the packed calls bundle (ADR-0020) instead of
    re-replaying the archive. The bundle's type keys carry a `call_` prefix
    (`CALL_TYPE_ORDER`) while `call_type` here is the bare Decision name, so we
    map `<call_type>` → `call_<call_type>`."""
    name = config["dataset"]
    kwargs = dict(config.get("dataset_kwargs", {}))
    if name == "synthetic":
        return list(SyntheticCallDataset(**kwargs))
    if name == "parquet":
        materialize = bool(kwargs.pop("materialize", False))
        ds = ParquetCallDataset(call_type=call_type, **kwargs)
        return list(ds) if materialize else ds
    if name == "memmap":
        # Never materialise — the trainer streams via iter_batches, so
        # `materialize` (if present) is dropped and the re-iterable dataset is
        # returned for the streaming fast path.
        kwargs.pop("materialize", None)
        return build_memmap_dataset(
            MemmapCallDataset, kwargs, call_type=f"call_{call_type}",
        )
    raise ValueError(f"unknown dataset: {name!r}")


def _peek_feature_dim(examples) -> int:
    """Get the feature dimension without consuming a streaming dataset.
    Materialised lists are peeked at index 0; ParquetCallDataset reports
    the featurizer's output dim via `FEATURIZER_OUTPUT_DIM`."""
    if isinstance(examples, list):
        return int(examples[0].features.shape[0])
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    return FEATURIZER_OUTPUT_DIM


if __name__ == "__main__":
    sys.exit(main())
