"""`train_schupfen` CLI — trains the standalone Schupfen Network.

Mirrors `train_calls` (single-network shape — schupfen has no
grand/tichu split). Smoke path consumes `SyntheticSchupfenDataset`;
production path consumes `ParquetSchupfenDataset` (play-shard manifest
+ archive streaming). Output is a `Checkpoint` with the live
`FEATURIZER_VERSION` / `ACTION_SPACE_VERSION` stamped (ADR-0012 §3).
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import torch
import yaml

from tichu_training.bc.schupfen_model import SchupfenNetwork
from tichu_training.bc.schupfen_training import (
    ParquetSchupfenDataset,
    SyntheticSchupfenDataset,
    train_one_schupfen_epoch,
)
from tichu_training.bc.training import save_checkpoint


log = logging.getLogger("train_schupfen")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the standalone Schupfen Network.")
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

    examples = _build_dataset(config)
    feature_dim = _peek_feature_dim(examples)
    m = config.get("model", {})
    net = SchupfenNetwork(
        feature_dim=feature_dim,
        skill_buckets=10,
        skill_dim=int(m.get("skill_dim", 64)),
        hidden=int(m.get("hidden", 256)),
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=float(config["learning_rate"]))
    log_path = run_dir / "step.csv"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Materialise streaming datasets once for multi-epoch training (parquet
    # adapter is re-iterable, but re-streaming the BSW archive every epoch
    # is wasteful when the in-memory example list fits).
    rate_examples = examples if isinstance(examples, list) else list(examples)
    n_epochs = int(config["epochs"])
    for epoch in range(n_epochs):
        loss = train_one_schupfen_epoch(
            net, rate_examples, optimizer,
            batch_size=int(config["batch_size"]),
            log_path=log_path,
            desc=f"epoch {epoch + 1}/{n_epochs}",
        )
        log.info("epoch %d final batch loss=%.4f", epoch, loss)

    save_checkpoint(net, optimizer, step=n_epochs,
                    path=ckpt_dir / "schupfen_final.bin")
    log.info("saved final checkpoint")
    return 0


def _build_dataset(config):
    name = config["dataset"]
    kwargs = dict(config.get("dataset_kwargs", {}))
    if name == "synthetic":
        return list(SyntheticSchupfenDataset(**kwargs))
    if name == "parquet":
        materialize = bool(kwargs.pop("materialize", False))
        ds = ParquetSchupfenDataset(**kwargs)
        return list(ds) if materialize else ds
    raise ValueError(f"unknown dataset: {name!r}")


def _peek_feature_dim(examples) -> int:
    if isinstance(examples, list):
        return int(examples[0].features.shape[0])
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    return FEATURIZER_OUTPUT_DIM


if __name__ == "__main__":
    sys.exit(main())
