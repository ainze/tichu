"""Ship check: the wishfix rolling champion vs the SERVED cpfix3328 export.

The ADR-0040 kill (2026-07-26) closed the paired-playout family; the post-kill
audit showed the wishfix BC lineage already banks the wish-decline fix
(3f7548b), the residual-head capacity upgrade, and the schupfen
current_player fix — while the SERVED export (cpfix3328) predates all three.
This is the standing ship bar's tournament leg: seat-swapped greedy paired
deltas (the exact per-Position observations check_cotrain bootstraps), n=40k
deals, 95% CI. The move-prediction drift leg (top-1 within −3pp) runs
separately after.

The candidate is the promoted `_champion.pt` itself (gate-window iterations
are 128-multiples, snapshots 25-multiples, so no snapshot equals the champion
— check_cotrain's snapshot path can't target it). Same export + tournament
machinery as `diag_vine_v3_chimera_gate.py`, whose learner arm replicated the
live gate's reading (−2.32 vs −2.2) — harness validated 2026-07-26.

Ship readout: CI_lo > 0 clears the strict bar. A CI straddling zero but
excluding a regression still supports a correctness-motivated ship (the
served export over-wishes and gives Dog to the grand-caller's partner) —
owner's call, this script just reports.

    $env:PYTHONPATH = "<worktree>\\src"
    py scripts/ship_check_wishfix_vs_cpfix3328.py --deals 40000 --workers 10
"""

import argparse
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_eval.tournament import collect_pair_deltas
from tichu_training.cli.check_cotrain import _build_agent
from tichu_training.cli.train_cotrain import _NET_TYPES, _build_models
from tichu_training.ppo.greedy_gate import export_live_models

WISHFIX_RUN = Path(r"C:/workbench/tichu/data/runs/cotrain_v6_pbrs_resid_wish_gated_wishfix")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cotrain_v6_vine_v3_gated.yaml",
                    help="arch + eval.master (cpfix3328 TorchScript paths)")
    ap.add_argument("--deals", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=180_000_000,
                    help="fresh deal stream, >=100M, distinct from all diag streams")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=str(WISHFIX_RUN / "ship_check_vs_cpfix3328.npz"))
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    skill_decile = int(config["ppo"].get("skill_decile", 9))
    master_kwargs = dict(config["eval"]["master"])  # the served cpfix3328 export

    champ = torch.load(WISHFIX_RUN / "_champion.pt", map_location="cpu",
                       weights_only=False)
    models = _build_models(config)
    for dt in _NET_TYPES:
        models[dt].load_state_dict(champ["models"][dt])
        models[dt].eval()
    champ_kwargs = export_live_models(models, WISHFIX_RUN / "_ship_check_export")
    print(f"candidate: {WISHFIX_RUN / '_champion.pt'}", flush=True)
    print(f"master:    {master_kwargs['checkpoint_path']}", flush=True)

    positions = generate_full_position_pool(seed=args.seed, n=args.deals)
    print(f"{args.deals} deals (seed {args.seed}), {args.workers} workers", flush=True)

    t0 = time.time()
    done = [0]

    def _progress(n):
        done[0] += n
        if done[0] % 4096 < n:
            el = (time.time() - t0) / 60
            eta = el / max(done[0], 1) * (args.deals - done[0])
            print(f"  {done[0]}/{args.deals} deals  {el:.1f} min  (eta {eta:.0f} min)",
                  flush=True)

    builder_c = partial(_build_agent, "ml", skill_decile=skill_decile, **champ_kwargs)
    builder_m = partial(_build_agent, "ml", skill_decile=skill_decile, **master_kwargs)
    totals, bonuses = collect_pair_deltas(builder_c, builder_m, positions,
                                          workers=args.workers, progress=_progress)
    np.savez(args.out, totals=totals, bonuses=bonuses,
             deals=np.asarray([args.deals]), seed=np.asarray([args.seed]))

    rng = np.random.default_rng(0)
    boots = totals[rng.integers(0, totals.size, size=(10_000, totals.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\n=== SHIP CHECK: wishfix champion vs cpfix3328 (served) ===", flush=True)
    print(f"margin/round: {totals.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  "
          f"(n={totals.size} obs, {args.deals} deals)", flush=True)
    print(f"round record: win {(totals > 0).mean():.1%}  "
          f"loss {(totals < 0).mean():.1%}  tie {(totals == 0).mean():.1%}", flush=True)
    print(f"call-bonus component: {bonuses.mean():+.3f}/round", flush=True)
    verdict = ("CLEARS the strict CI>0 ship bar" if lo > 0 else
               "regression NOT excluded — do not ship" if hi < 0 else
               "CI straddles zero — strict bar not met; correctness-motivated "
               "ship is an owner call (regression excluded)" if lo > -1.0 else
               "CI straddles zero, wide — strict bar not met")
    print(f"verdict: {verdict}", flush=True)
    print(f"rows saved -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
