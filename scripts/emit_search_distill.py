r"""Emit played-out move comparisons for search distillation — RESUMABLE.

See `tichu_training.search.search_distill.emit_decision_targets`.

  py -3.14 -m scripts.emit_search_distill `
      --export-dir C:\workbench\tichu\data\runs\cotrain_v7_gated\export\iter_15360 `
      --out-dir C:\workbench\tichu\data\runs\search_distill_v1\shards `
      --shards 200 --deals-per-shard 250 --workers 11
"""

import argparse
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Disjoint from the eval pools (0..49151) and the value corpora (10M.., 20M..).
SEED_BASE = 30_000_000

_AGENT = None


def _init(export_dir: str) -> None:
    global _AGENT
    import torch

    torch.set_num_threads(1)
    from tichu_training.ppo.pmcpa import build_export_agent

    _AGENT = build_export_agent(export_dir=export_dir)


def _emit_shard(task) -> tuple[int, int]:
    shard, deals, decisions, top_k, seed_base, out_dir = task
    import numpy as np

    from tichu_eval.full_position_pool import generate_full_position_pool
    from tichu_training.search.search_distill import emit_decision_targets

    out_path = Path(out_dir) / f"shard_{shard:05d}.npz"
    seed = seed_base + shard * deals
    rng = random.Random(seed)
    cols = {k: [] for k in ("features", "mask", "cand", "q", "phase", "deal")}
    for i, position in enumerate(generate_full_position_pool(seed, deals)):
        for item in emit_decision_targets(_AGENT, position, rng, decisions=decisions,
                                          top_k=top_k):
            for k, v in item.items():
                cols[k].append(v)
            cols["deal"].append(seed + i)
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez(
        tmp, features=np.stack(cols["features"]),
        mask=np.packbits(np.stack(cols["mask"]), axis=1),
        cand=np.stack(cols["cand"]), q=np.stack(cols["q"]),
        phase=np.asarray(cols["phase"], dtype=np.int8),
        deal=np.asarray(cols["deal"], dtype=np.int64))
    tmp.replace(out_path)
    return shard, len(cols["deal"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--deals-per-shard", type=int, default=250)
    p.add_argument("--decisions", type=int, default=6)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=SEED_BASE)
    p.add_argument("--workers", type=int, default=11)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(s, args.deals_per_shard, args.decisions, args.top_k, args.seed_base,
              str(out_dir))
             for s in range(args.shards)
             if not (out_dir / f"shard_{s:05d}.npz").exists()]
    print(f"{len(tasks)} of {args.shards} shards to emit", flush=True)
    t0 = time.monotonic()
    done = rows = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(args.export_dir,)) as pool:
        for shard, n in pool.imap_unordered(_emit_shard, tasks):
            done += 1
            rows += n
            el = time.monotonic() - t0
            print(f"shard {shard} done ({n} decisions) | {done}/{len(tasks)} | "
                  f"{rows:,} decisions | {el / 60:.1f} min | "
                  f"ETA {el / done * (len(tasks) - done) / 60:.0f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
