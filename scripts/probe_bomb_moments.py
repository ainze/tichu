r"""Issue #110 probe: value of the bomb moments the Bomb Window contract omits.

Team A answers the extra windows, team B always passes in them; same export on
both sides, seat-swap, paired bootstrap CI.

  py -3.14 -m scripts.probe_bomb_moments --mode pass --n-deals 5000 --out <json>
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

CHAMPION = r"C:\workbench\tichu\data\runs\cotrain_v7_gated\export\iter_15360"
POOL = r"C:\workbench\tichu\data\full_position_pool_s0_n20000.parquet"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", default=CHAMPION)
    p.add_argument("--pool", default=POOL)
    p.add_argument("--mode", choices=("pass", "empty", "both"), required=True)
    p.add_argument("--n-deals", type=int, default=5000)
    p.add_argument("--count-deals", type=int, default=300,
                   help="deals played serially first, to count the extra windows")
    p.add_argument("--workers", type=int, default=11)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    from tichu_eval.full_position_pool import load_full_position_pool
    from tichu_eval.play_full import play_full_round
    from tichu_eval.tournament import _bootstrap_ci, collect_pair_deltas
    from tichu_training.search import bomb_moments as bm

    kw = {"export_dir": args.export, "after_pass": args.mode in ("pass", "both"),
          "empty": args.mode in ("empty", "both")}
    pool = load_full_position_pool(Path(args.pool))
    res = {"mode": args.mode, "export": args.export}

    if args.count_deals:
        # Everyone answers the extra windows: how often do they open, and how
        # often does the net bomb in them?
        agent = bm.build_user(**kw)
        bm.COUNTS.clear()
        for pos in pool[-args.count_deals:]:
            play_full_round([agent] * 4, pos.state, pos.grand_prefixes)
        res["counts_per_1000_rounds"] = {
            k: round(1000 * v / args.count_deals, 1) for k, v in sorted(bm.COUNTS.items())}
        print(json.dumps(res["counts_per_1000_rounds"], indent=2), flush=True)

    if args.n_deals:
        deals = pool[: args.n_deals]
        totals, bonuses = collect_pair_deltas(
            partial(bm.build_user, **kw), partial(bm.build_decliner, **kw), deals,
            workers=args.workers, max_chunk=max(1, len(deals) // (args.workers * 4)))
        rng = np.random.default_rng(0)
        mean, lo, hi = _bootstrap_ci(totals, 2000, rng, paired=True)
        res.update({
            "deals": len(deals), "mean": mean, "ci": [lo, hi],
            # Seat-swap deltas cancel exactly when the extra windows changed nothing.
            "unchanged_deals": float((totals.reshape(-1, 2).sum(axis=1) == 0).mean())})
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
