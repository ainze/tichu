"""`train_calls` CLI — trains the Tichu and Grand Tichu call networks."""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import torch
import yaml

from tichu_training.bc.call_model import GrandTichuCallNetwork, TichuCallNetwork
from tichu_training.bc.call_training import (
    ParquetCallDataset,
    SyntheticCallDataset,
    train_one_call_epoch,
    write_calling_rate_csv,
)
from tichu_training.bc.training import save_checkpoint


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
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=float(config["learning_rate"]))
        log_path = run_dir / f"{tag}_step.csv"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # For streaming datasets, the calling-rate CSV needs a materialised
        # snapshot of examples (it indexes into the data). Materialise once
        # per tag — the parquet adapter is re-iterable, but the rate CSV
        # asks for predictions on each example, not a fresh pass.
        rate_examples = examples if isinstance(examples, list) else list(examples)
        n_epochs = int(config["epochs"])
        for epoch in range(n_epochs):
            loss = train_one_call_epoch(
                net, rate_examples, optimizer,
                batch_size=int(config["batch_size"]),
                log_path=log_path,
                desc=f"[{tag}] epoch {epoch + 1}/{n_epochs}",
            )
            log.info("[%s] epoch %d final batch loss=%.4f", tag, epoch, loss)
            write_calling_rate_csv(
                net, rate_examples,
                path=run_dir / f"{tag}_calling_rate_by_decile_epoch{epoch}.csv",
            )
        save_checkpoint(net, optimizer, step=int(config["epochs"]),
                        path=ckpt_dir / f"{tag}_final.bin")
        log.info("[%s] saved final checkpoint", tag)

    return 0


def _build_dataset(config, call_type: str):
    """Construct the per-tag dataset. Synthetic returns a list; parquet
    returns a re-iterable `ParquetCallDataset` (or a materialised list
    if `materialize: true` is set on the parquet path)."""
    name = config["dataset"]
    kwargs = dict(config.get("dataset_kwargs", {}))
    if name == "synthetic":
        return list(SyntheticCallDataset(**kwargs))
    if name == "parquet":
        materialize = bool(kwargs.pop("materialize", False))
        ds = ParquetCallDataset(call_type=call_type, **kwargs)
        return list(ds) if materialize else ds
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
