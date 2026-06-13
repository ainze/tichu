r"""Pool pMCPA A/B shards into one exact bootstrap (ADR-0036).

Concatenates every `shard_*.npz` raw-delta file written by `scripts.run_pmcpa_ab`
and draws a single percentile bootstrap over the pooled per-Position deltas — the
exact CI a one-shot tournament would have produced, recovered from the resumable
sharded run. The headline read: mean A-minus-B (pMCPA minus opponent) with 95% CI
and per-Round win-rate. Ship bar: CI lower bound > 0.

  py -m scripts.pool_pmcpa_ab --out-dir C:\workbench\tichu\data\runs\pmcpa_probe\ab_offline
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Pool pMCPA A/B shards.")
    p.add_argument("--out-dir", required=True, help="dir of shard_*.npz files")
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    shards = sorted(out_dir.glob("shard_*.npz"))
    if not shards:
        p.error(f"no shard_*.npz files in {out_dir}")

    totals, bonuses = [], []
    for s in shards:
        with np.load(s) as z:
            totals.append(z["totals"])
            bonuses.append(z["bonuses"])
    totals = np.concatenate(totals)
    bonuses = np.concatenate(bonuses)

    from tichu_eval.tournament import _bootstrap_ci

    rng = np.random.default_rng(args.seed)
    mean, lo, hi = _bootstrap_ci(totals, args.bootstrap_iters, rng)
    n = len(totals)
    win = float((totals > 0).mean())
    tie = float((totals == 0).mean())
    cb = float(bonuses.mean())

    print(f"shards pooled : {len(shards)}  (n={n} seat-swap observations)")
    print(f"pMCPA - opp   : mean={mean:+.2f}  95% CI=[{lo:+.2f}, {hi:+.2f}]")
    print(f"call-bonus     : {cb:+.2f}/round")
    print(f"win / tie / loss: {win:.3f} / {tie:.3f} / {1 - win - tie:.3f}")
    verdict = ("SHIP (CI>0)" if lo > 0 else
               "REGRESSION (CI<0)" if hi < 0 else "FLAT (CI spans 0)")
    print(f"verdict        : {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
