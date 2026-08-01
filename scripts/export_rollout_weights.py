"""Export a rollout-weights `.pt` bundle to a four-net TorchScript directory.

The gate's opponents (`_bc_opponent.pt`, `_champion.pt`, `iter*_opponent.pt`) are
stored as `{"models": {net: state_dict}, "critic": ...}` — the format the spawned
rollout workers load. `eval_matrix` needs deployed TorchScript instead, and some of
these bundles are the ONLY surviving copy of a reference agent (the frozen BC of a
run whose BC checkpoints have since been deleted to save disk).

Thin wrapper over `greedy_gate.export_opponent`, which already rebuilds the arch
from a config and writes the four `MLAgent` files.

  py -m scripts.export_rollout_weights \
      --weights data/runs/<run>/_bc_opponent.pt \
      --config configs/cotrain_v6_pbrs_resid_wish_gated_wishfix.yaml \
      --out data/export/<name>

`--warm-start` instead exports the four `.bin` checkpoints named in the config's
`warm_start` block — i.e. the agent the run actually STARTED from.

  py -m scripts.export_rollout_weights --warm-start \
      --config configs/cotrain_v6_pbrs_resid_wish_gated_wishfix.yaml \
      --out data/export/<name>

Prefer `--warm-start` when you want "the BC": a run's `_bc_opponent.pt` is written
lazily (`if not exists`) at gate-setup time, so on a run where `also_beat_bc` was
enabled partway through — or on any resume after a promotion under
`reanchor_on_promote`, which restores `bc_models["play"]` from the bundle's
`play_anchor` — the file captures the ANCHOR AT THAT MOMENT, not the BC. Verified on
the wishfix run: its `_bc_opponent.pt` play net differs from the BC export by
max|logit diff| 812, so it is not the BC despite the filename.
"""

import argparse
from pathlib import Path

import yaml

_NET_TYPES = ("play", "schupfen", "tichu", "grand")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", help="rollout-weights .pt bundle")
    ap.add_argument("--warm-start", action="store_true",
                    help="export the config's warm_start .bin checkpoints instead")
    ap.add_argument("--config", required=True,
                    help="config supplying the model/schupfen_model/call_model/grand_model arch")
    ap.add_argument("--out", required=True, help="destination directory for the four .pt files")
    args = ap.parse_args(argv)
    if bool(args.weights) == bool(args.warm_start):
        ap.error("pass exactly one of --weights or --warm-start")

    from tichu_training.cli.train_cotrain import _arch_cfg, _build_models
    from tichu_training.ppo.greedy_gate import export_live_models, export_opponent

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    if args.warm_start:
        from tichu_training.bc.training import load_checkpoint

        warm = config["warm_start"]
        models = _build_models(config)
        for net in _NET_TYPES:
            src = Path(warm[net])
            if not src.is_file():
                raise FileNotFoundError(f"warm_start[{net}] not found: {src}")
            load_checkpoint(str(src), models[net])
            print(f"  loaded {net:9s} {src}")
        kwargs = export_live_models(models, Path(args.out))
        print("exported the config's warm_start checkpoints")
    else:
        src = Path(args.weights)
        if not src.is_file():
            raise FileNotFoundError(f"weights not found: {src}")
        kwargs = export_opponent(str(src), _arch_cfg(config), Path(args.out))
        print(f"exported {src}")

    for k, v in kwargs.items():
        print(f"  {k:17s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
