r"""Calibrate the Shrink-and-Perturb blend strength before training on it.

Arm P v1 of the Shrink-and-Perturb Probe used alpha=0.6, beta=0.4 untested, and
the blend wrecked the policy instead of perturbing it. Even after 128 iterations
the learner beat its own starting weights by +87/round, and it started 58 below the
v7 champion (docs/notes/2026-10-10-train-from-random-options-memo.md, §8). The probe
needs a start that has visibly lost ground but still holds most of what the
champion knows. This script measures that directly, before any training.

For each alpha, it blends the source champion's selected nets toward the SAME fresh
init (`p <- alpha*p + beta*p_fresh`, beta = 1 - alpha unless --beta is given) and
plays the blend against the unperturbed source champion. The tournament is greedy
and seat-swapped (`collect_pair_deltas`, the ship-check harness), with a paired
cluster bootstrap. Every alpha plays the SAME deals, so the alphas are compared on
common deals.

Pick rule: the smallest alpha (= strongest perturbation) whose mean margin lies in
the band [-hi, -lo] (default [-15, -3]). If none does, it says which side every
alpha fell on so you can bracket the next sweep.

Usage (worktree root, this worktree's src FIRST on PYTHONPATH; nothing else heavy
running):

  python scripts/calibrate_shrink_perturb.py `
      --weights C:\workbench\tichu\data\runs\cotrain_v7_gated\_champion.pt `
      --config  configs\cotrain_v7_snp_a060.yaml `
      --alphas 1.0 0.95 0.9 0.85 0.8 0.7 --deals 4096 --workers 10 `
      --out-dir C:\workbench\tichu\data\snp_v7\calibration
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml

from tichu_training.shrink_perturb import NET_TYPES as _NET_TYPES
from tichu_training.shrink_perturb import build_perturbed


def summarise(totals: np.ndarray, *, iters: int = 10_000) -> dict:
    """Mean margin and the PAIRED (seat-swap cluster) 95% CI of one alpha's deltas."""
    from tichu_eval.tournament import _bootstrap_ci

    _, lo, hi = _bootstrap_ci(totals, iters, np.random.default_rng(0), paired=True)
    return {"mean": float(np.mean(totals)), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_obs": int(np.size(totals))}


def pick_alpha(rows: list[dict], *, lo: float, hi: float) -> dict | None:
    """The smallest alpha whose mean margin is in [-hi, -lo], or None.

    Smallest alpha = the strongest shake that still passes. The band reads the
    point estimate, not the CI: the band edges are judgement calls, and the CI is
    printed next to every row so a borderline pick is visible.
    """
    if not 0 <= lo < hi:
        raise ValueError(f"need 0 <= lo < hi, got lo={lo}, hi={hi}")
    in_band = [r for r in rows if -hi <= r["mean"] <= -lo]
    return min(in_band, key=lambda r: r["alpha"]) if in_band else None


def calibrate(alphas: list[float], evaluate: Callable[[float], np.ndarray]) -> list[dict]:
    """Evaluate each alpha (`evaluate(alpha)` -> seat-swap deltas) and summarise."""
    rows = []
    for a in sorted(alphas, reverse=True):
        rows.append({"alpha": a, **summarise(evaluate(a))})
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="source champion rollout-weights .pt")
    p.add_argument("--config", required=True, help="cotrain YAML (arch blocks + ppo.skill_decile)")
    p.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.95, 0.9, 0.85, 0.8, 0.7],
                   help="alpha=1.0 is the harness check: the champion vs itself must read exactly 0")
    p.add_argument("--beta", type=float, default=None,
                   help="fixed fresh-init scale; default 1 - alpha (interpolation)")
    p.add_argument("--seed", type=int, default=0,
                   help="fresh-init seed; must match the seed you train with")
    p.add_argument("--nets", nargs="+", default=["play"], choices=_NET_TYPES)
    p.add_argument("--deals", type=int, default=4096)
    p.add_argument("--deal-seed", type=int, default=190_000_000,
                   help="deal stream, >=100M, distinct from the gate/ship-check streams")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--band", type=float, nargs=2, default=[3.0, 15.0], metavar=("LO", "HI"),
                   help="target deficit: mean margin in [-HI, -LO]")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    from tichu_eval.full_position_pool import generate_full_position_pool
    from tichu_eval.tournament import collect_pair_deltas
    from tichu_training.cli.check_cotrain import _build_agent
    from tichu_training.cli.train_cotrain import _build_models
    from tichu_training.ppo.greedy_gate import export_live_models

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    skill_decile = int(config.get("ppo", {}).get("skill_decile", 9))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = torch.load(args.weights, map_location="cpu", weights_only=False)
    champ = _build_models(config)
    for net in _NET_TYPES:
        champ[net].load_state_dict(src["models"][net])
        champ[net].eval()
    champ_kwargs = export_live_models(champ, out_dir / "_export_source")
    builder_champ = partial(_build_agent, "ml", skill_decile=skill_decile, **champ_kwargs)

    positions = generate_full_position_pool(seed=args.deal_seed, n=args.deals)
    print(f"source {args.weights}\n{args.deals} deals (seed {args.deal_seed}), "
          f"{args.workers} workers, nets={args.nets}", flush=True)

    def evaluate(alpha: float) -> np.ndarray:
        beta = (1.0 - alpha) if args.beta is None else args.beta
        models, stats = build_perturbed(Path(args.weights), config, alpha=alpha, beta=beta,
                                        seed=args.seed, nets=list(args.nets))
        for m in models.values():
            m.eval()
        kwargs = export_live_models(models, out_dir / f"_export_a{alpha:.3f}")
        t0 = time.time()
        totals, bonuses = collect_pair_deltas(
            partial(_build_agent, "ml", skill_decile=skill_decile, **kwargs),
            builder_champ, positions, workers=args.workers)
        np.savez(out_dir / f"deltas_a{alpha:.3f}.npz", totals=totals, bonuses=bonuses,
                 alpha=alpha, beta=beta, seed=args.seed, deal_seed=args.deal_seed)
        rel = stats[args.nets[0]]["rel_change"]
        print(f"  alpha={alpha:.3f} beta={beta:.3f}  rel_change={rel:.3f}  "
              f"margin {totals.mean():+.2f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        return totals

    rows = calibrate(args.alphas, evaluate)
    for r in rows:
        if r["alpha"] == 1.0 and r["mean"] != 0.0:
            print(f"WARNING: alpha=1.0 (champion vs itself) read {r['mean']:+.3f}, not 0 -- "
                  "the harness is not deterministic; distrust every row", flush=True)
    lo, hi = args.band
    pick = pick_alpha(rows, lo=lo, hi=hi)

    with (out_dir / "calibration.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "mean", "ci_lo", "ci_hi", "n_obs"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== blend vs source champion (target band [{-hi:+.0f}, {-lo:+.0f}]) ===")
    for r in rows:
        mark = "  <- pick" if pick is r else ("  in band" if -hi <= r["mean"] <= -lo else "")
        print(f"  alpha={r['alpha']:.3f}  {r['mean']:+7.2f}  "
              f"CI [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]{mark}")
    if pick is not None:
        print(f"PICK alpha={pick['alpha']:.3f} (beta="
              f"{(1 - pick['alpha']) if args.beta is None else args.beta:.3f})")
    else:
        sides = ["too gentle" if r["mean"] > -lo else "too strong" for r in rows]
        print(f"NO alpha in band: {dict(zip([r['alpha'] for r in rows], sides))}; "
              "bracket the next sweep between the last too-gentle and first too-strong alpha")
    print(f"wrote {out_dir / 'calibration.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
