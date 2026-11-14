r"""Seat-swap A/B of two export dirs (plain agents, no search).

  py -3.14 -m scripts.run_export_ab --a <export dir> --b <export dir> `
      --pool C:\workbench\tichu\data\full_position_pool_s0_n20000.parquet --n-deals 5000
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--n-deals", type=int, default=5000)
    p.add_argument("--workers", type=int, default=11)
    p.add_argument("--out", default=None)
    p.add_argument("--a-play-only", action="store_true",
                   help="A uses its policy only for on-turn Play Decisions, B's otherwise")
    args = p.parse_args(argv)

    from tichu_eval.full_position_pool import load_full_position_pool
    from tichu_eval.tournament import _bootstrap_ci, collect_pair_deltas
    from tichu_training.ppo.pmcpa import build_export_agent

    pool = load_full_position_pool(Path(args.pool))[: args.n_deals]
    from tichu_training.search.search_distill import build_on_turn_play_only

    builder_a = (partial(build_on_turn_play_only, tuned_dir=args.a, base_dir=args.b)
                 if args.a_play_only else partial(build_export_agent, export_dir=args.a))
    totals, bonuses = collect_pair_deltas(
        builder_a,
        partial(build_export_agent, export_dir=args.b),
        pool, workers=args.workers, max_chunk=max(1, len(pool) // (args.workers * 4)))
    rng = np.random.default_rng(0)
    mean, lo, hi = _bootstrap_ci(totals, 2000, rng, paired=True)
    bmean, blo, bhi = _bootstrap_ci(bonuses, 2000, rng, paired=True)
    res = {"a": args.a, "b": args.b, "deals": len(pool), "mean": mean, "ci": [lo, hi],
           "call_bonus_mean": bmean, "call_bonus_ci": [blo, bhi],
           "identical_deals": float((np.abs(totals.reshape(-1, 2)).sum(axis=1) == 0).mean())}
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
