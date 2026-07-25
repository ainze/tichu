"""One-shot setup for the vine v3 run (ADR-0040): materialise the warm-start
checkpoints from the wishfix run's rolling champion, and pre-seed the frozen-BC
gate opponent.

Why the pre-seed: train_cotrain derives `_bc_opponent.pt` from the warm_start
models at first launch — and v3 warm-starts from the CHAMPION, which would
silently turn the "bc" gate axis into a second champion axis. The trainer only
writes the file if missing, so copying the wishfix run's `_bc_opponent.pt`
first keeps the true wishfix-BC as the fixed human reference.

    py scripts/setup_vine_v3.py \
        --source-run C:/workbench/tichu/data/runs/cotrain_v6_pbrs_resid_wish_gated_wishfix \
        --config configs/cotrain_v6_vine_v3_gated.yaml
"""

import argparse
import shutil
from pathlib import Path

import torch
import yaml

from tichu_training.bc.training import save_checkpoint
from tichu_training.cli.train_cotrain import _NET_TYPES, _build_models


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", required=True,
                    help="run dir holding _champion.pt + _bc_opponent.pt (the wishfix run)")
    ap.add_argument("--config", required=True, help="the v3 config (arch + warm_start paths)")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    source = Path(args.source_run)
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    champion = torch.load(source / "_champion.pt", map_location="cpu", weights_only=False)
    models = _build_models(config)
    for dt in _NET_TYPES:
        models[dt].load_state_dict(champion["models"][dt])
        warm_path = Path(config["warm_start"][dt])
        warm_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(models[dt], torch.optim.Adam(models[dt].parameters()),
                        step=0, path=str(warm_path))
        print(f"warm[{dt}] <- champion  -> {warm_path}")

    bc_opp = run_dir / "_bc_opponent.pt"
    if bc_opp.exists():
        print(f"pre-seed skipped ({bc_opp} exists)")
    else:
        shutil.copyfile(source / "_bc_opponent.pt", bc_opp)
        print(f"pre-seeded {bc_opp} <- {source / '_bc_opponent.pt'} (true wishfix BC)")
    print("setup complete — launch with:\n"
          f"  py -m tichu_training.cli.train_cotrain --config {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
