r"""Value Search A/B (test build) — RESUMABLE, sharded.

Seat-swap tournament of the export wrapped in Value Search against the same
export without it, over a slice of the Full Position Pool. Each shard writes its
raw per-arrangement deltas; a shard that exists is skipped. Ends by pooling
every shard in the out dir into a mean with a paired (per-deal) bootstrap CI.

  py -3.14 -m scripts.run_value_search_ab `
      --export-dir C:\workbench\tichu\data\runs\cotrain_v7_gated\export\iter_15360 `
      --value C:\workbench\tichu\data\runs\full_state_value_v1\fit\full_state_value.pt `
      --pool C:\workbench\tichu\data\full_position_pool_s0_n20000.parquet `
      --n-deals 1000 --n-shards 10 --workers 10 `
      --worlds 20 --top-k 4 --depth 0 --delta 10 --t 2.5 `
      --out-dir C:\workbench\tichu\data\runs\value_search_ab\d0_w20

`--depth -1` plays every world out (exact-playout reference; no value net).
"""

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def pool_result(out_dir: Path) -> dict:
    from tichu_eval.tournament import _bootstrap_ci

    totals, bonuses = [], []
    for f in sorted(out_dir.glob("shard_*.npz")):
        d = np.load(f)
        totals.append(d["totals"])
        bonuses.append(d["bonuses"])
    totals, bonuses = np.concatenate(totals), np.concatenate(bonuses)
    rng = np.random.default_rng(0)
    mean, lo, hi = _bootstrap_ci(totals, 2000, rng, paired=True)
    bmean, blo, bhi = _bootstrap_ci(bonuses, 2000, rng, paired=True)
    stats = {"decisions": 0, "deviations": 0, "gain_sum": 0.0}
    for f in (out_dir / "stats").glob("stats_*.json"):
        s = json.loads(f.read_text())
        for k in stats:
            stats[k] += s[k]
    return {
        "deals": int(len(totals) // 2), "mean": mean, "ci": [lo, hi],
        "call_bonus_mean": bmean, "call_bonus_ci": [blo, bhi],
        "identical_deals": float((np.abs(totals.reshape(-1, 2)).sum(axis=1) == 0).mean()),
        "search_stats_partial": stats,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export-dir", required=True)
    p.add_argument("--value", default=None)
    p.add_argument("--pool", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-deals", type=int, required=True)
    p.add_argument("--n-shards", type=int, default=10)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--worlds", type=int, default=20)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--delta", type=float, default=10.0)
    p.add_argument("--t", type=float, default=2.5)
    p.add_argument("--tau", type=float, default=None,
                   help="points per nat; switches to the policy+value rule")
    args = p.parse_args(argv)

    from tichu_eval.full_position_pool import load_full_position_pool
    from tichu_eval.tournament import collect_pair_deltas
    from tichu_training.ppo.pmcpa import build_export_agent
    from tichu_training.search.value_search import build_value_search_agent

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = load_full_position_pool(Path(args.pool))[: args.n_deals]
    depth = None if args.depth < 0 else args.depth
    builder_b = partial(build_export_agent, export_dir=args.export_dir)
    t0 = time.monotonic()
    for k in range(args.n_shards):
        out_path = out_dir / f"shard_{k:04d}.npz"
        if out_path.exists():
            continue
        base, extra = divmod(len(pool), args.n_shards)
        start = k * base + min(k, extra)
        positions = pool[start:start + base + (1 if k < extra else 0)]
        if not positions:
            continue
        builder_a = partial(
            build_value_search_agent, export_dir=args.export_dir,
            value_path=None if depth is None else args.value,
            worlds=args.worlds, top_k=args.top_k, depth=depth,
            delta_threshold=args.delta, t_threshold=args.t, mix_tau=args.tau, seed=k,
            stats_dir=str(out_dir / "stats"))
        max_chunk = max(1, len(positions) // (args.workers * 3))
        totals, bonuses = collect_pair_deltas(
            builder_a, builder_b, positions, workers=args.workers, max_chunk=max_chunk)
        tmp = out_path.with_name(out_path.stem + ".tmp.npz")
        np.savez(tmp, totals=totals, bonuses=bonuses)
        tmp.replace(out_path)
        res = pool_result(out_dir)
        print(f"shard {k + 1}/{args.n_shards} | {(time.monotonic() - t0) / 60:.1f} min | "
              f"deals {res['deals']} mean {res['mean']:+.2f} "
              f"CI [{res['ci'][0]:+.2f}, {res['ci'][1]:+.2f}]", flush=True)
    res = pool_result(out_dir)
    (out_dir / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
