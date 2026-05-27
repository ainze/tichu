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

    examples = _build_dataset(config)
    feature_dim = examples[0].features.shape[0]

    targets = ["grand", "tichu"] if args.only is None else [args.only]
    for tag in targets:
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
        for epoch in range(int(config["epochs"])):
            loss = train_one_call_epoch(
                net, examples, optimizer,
                batch_size=int(config["batch_size"]),
                log_path=log_path,
            )
            log.info("[%s] epoch %d final batch loss=%.4f", tag, epoch, loss)
            write_calling_rate_csv(
                net, examples,
                path=run_dir / f"{tag}_calling_rate_by_decile_epoch{epoch}.csv",
            )
        save_checkpoint(net, optimizer, step=int(config["epochs"]),
                        path=ckpt_dir / f"{tag}_final.bin")
        log.info("[%s] saved final checkpoint", tag)

    return 0


def _build_dataset(config):
    name = config["dataset"]
    kwargs = dict(config.get("dataset_kwargs", {}))
    if name == "synthetic":
        return list(SyntheticCallDataset(**kwargs))
    if name == "parquet":
        raise NotImplementedError(
            "Parquet call-record path needs negative-example synthesis. "
            "Deferred to a follow-up; use the synthetic dataset for smoke runs."
        )
    raise ValueError(f"unknown dataset: {name!r}")


if __name__ == "__main__":
    sys.exit(main())
