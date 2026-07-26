"""Ship check, drift leg: Move-Prediction top-1 of the wishfix champion vs the
served cpfix3328 export on human games (the Q9 human-plausibility leash).

Companion to `ship_check_wishfix_vs_cpfix3328.py` (the tournament leg), reusing
check_cotrain's validated `_move_prediction` (strided archive sample, play
decisions only, >3pp top-1 drop vs master = crater flag; flag-not-gate). The
candidate's TorchScript export is the one the tournament leg already wrote to
`_ship_check_export`.

    $env:PYTHONPATH = "<worktree>\\src"
    py scripts/ship_check_movepred.py --games 2000
"""

import argparse
from functools import partial
from pathlib import Path

import yaml

from tichu_training.cli.check_cotrain import _build_agent, _move_prediction

WISHFIX_RUN = Path(r"C:/workbench/tichu/data/runs/cotrain_v6_pbrs_resid_wish_gated_wishfix")
EXPORT = WISHFIX_RUN / "_ship_check_export"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cotrain_v6_vine_v3_gated.yaml")
    ap.add_argument("--held-out", default=r"C:/workbench/tichu/data/archive.zst")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--max-decisions", type=int, default=None)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    skill_decile = int(config["ppo"].get("skill_decile", 9))
    champ_kwargs = {
        "checkpoint_path": str(EXPORT / "policy.pt"),
        "schupfen_path": str(EXPORT / "schupfen.pt"),
        "tichu_call_path": str(EXPORT / "tichu_call.pt"),
        "grand_call_path": str(EXPORT / "grand_tichu_call.pt"),
    }
    builders = {
        "master": partial(_build_agent, "ml", skill_decile=skill_decile,
                          **config["eval"]["master"]),
        "wishfix_champion": partial(_build_agent, "ml", skill_decile=skill_decile,
                                    **champ_kwargs),
    }
    result = _move_prediction(builders, "wishfix_champion", args.held_out,
                              max_decisions=args.max_decisions,
                              max_games=args.games, progress=True)
    print(f"\nflagged: {result['flagged']}  (ship bar: within 3pp of master)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
