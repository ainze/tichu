r"""Emit same-state move comparisons for the Full-State Value net — RESUMABLE.

See `tichu_training.search.full_state_value.emit_siblings`. One npz per shard.

  py -3.14 -m scripts.emit_value_siblings `
      --export-dir C:\workbench\tichu\data\runs\cotrain_v7_gated\export\iter_15360 `
      --out-dir C:\workbench\tichu\data\runs\value_siblings_v1\shards `
      --shards 60 --deals-per-shard 500 --workers 11
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

# Disjoint from the eval pools (0..49151) and the v1 value corpus (10M..10.4M).
SEED_BASE = 20_000_000

_AGENT = None


def _init(export_dir: str) -> None:
    global _AGENT
    import torch

    torch.set_num_threads(1)
    from tichu_training.ppo.pmcpa import build_export_agent

    _AGENT = build_export_agent(export_dir=export_dir)


def _emit_shard(task) -> tuple[int, int, int]:
    shard, deals, decisions, top_k, plies, out_dir = task
    import numpy as np

    from tichu_eval.full_position_pool import generate_full_position_pool
    from tichu_training.search.full_state_value import emit_siblings

    out_path = Path(out_dir) / f"shard_{shard:05d}.npz"
    seed = SEED_BASE + shard * deals
    positions = generate_full_position_pool(seed, deals)
    rng = random.Random(seed)
    agents = [_AGENT] * 4
    x, meta, cands = [], [], []
    for i, position in enumerate(positions):
        out = emit_siblings(agents, position, rng, decisions=decisions,
                            top_k=top_k, plies=plies)
        deal = seed + i
        for feats, z, d, c, ply, sign, phase in out["rows"]:
            x.append(feats)
            meta.append((z, deal * 16 + d, c, ply, sign, phase, deal))
        for d, c, zd, n, phase in out["cands"]:
            cands.append((deal * 16 + d, c, zd, n, phase, deal))
    m = np.asarray(meta, dtype=np.float64).reshape(-1, 7)
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez(
        tmp, x=np.stack(x).astype(np.float16), z=m[:, 0].astype(np.float32),
        dec=m[:, 1].astype(np.int64), cand=m[:, 2].astype(np.int8),
        ply=m[:, 3].astype(np.int8), sign=m[:, 4].astype(np.int8),
        phase=m[:, 5].astype(np.int8), deal=m[:, 6].astype(np.int64),
        cands=np.asarray(cands, dtype=np.float64).reshape(-1, 6))
    tmp.replace(out_path)
    return shard, len(x), len(cands)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--deals-per-shard", type=int, default=500)
    p.add_argument("--decisions", type=int, default=4)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--plies", type=int, default=5)
    p.add_argument("--workers", type=int, default=11)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(s, args.deals_per_shard, args.decisions, args.top_k, args.plies, str(out_dir))
             for s in range(args.shards)
             if not (out_dir / f"shard_{s:05d}.npz").exists()]
    print(f"{len(tasks)} of {args.shards} shards to emit", flush=True)
    t0 = time.monotonic()
    done = rows = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(args.export_dir,)) as pool:
        for shard, n, nc in pool.imap_unordered(_emit_shard, tasks):
            done += 1
            rows += n
            el = time.monotonic() - t0
            print(f"shard {shard} done ({n} rows, {nc} candidates) | {done}/{len(tasks)} | "
                  f"{rows:,} rows | {el / 60:.1f} min | "
                  f"ETA {el / done * (len(tasks) - done) / 60:.0f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
