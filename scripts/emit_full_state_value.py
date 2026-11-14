r"""Emit Full-State Value training data — RESUMABLE, sharded.

Greedy self-play of one export on fresh deals (seeds disjoint from the eval
pools), one counterfactual branch per deal; see
`tichu_training.search.full_state_value.emit_deal`. Each shard is one npz; a
shard that already exists is skipped.

  py -3.14 -m scripts.emit_full_state_value `
      --export-dir C:\workbench\tichu\data\runs\cotrain_v7_gated\export\iter_15360 `
      --out-dir C:\workbench\tichu\data\runs\full_state_value_v1\shards `
      --shards 200 --deals-per-shard 1000 --workers 10
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

# Eval pools use seeds 0..49151; training deals start far above them.
SEED_BASE = 10_000_000

_AGENT = None


def _init(export_dir: str) -> None:
    global _AGENT
    import torch

    torch.set_num_threads(1)
    from tichu_training.ppo.pmcpa import build_export_agent

    _AGENT = build_export_agent(export_dir=export_dir)


def _emit_shard(task) -> tuple[int, int]:
    shard, deals, rows_per_traj, out_dir = task
    import numpy as np

    from tichu_eval.full_position_pool import generate_full_position_pool
    from tichu_training.search.full_state_value import emit_deal

    out_path = Path(out_dir) / f"shard_{shard:05d}.npz"
    if out_path.exists():
        return shard, 0
    seed = SEED_BASE + shard * deals
    positions = generate_full_position_pool(seed, deals)
    rng = random.Random(seed)
    agents = [_AGENT] * 4
    x, z, deal, traj, turn, phase, live = [], [], [], [], [], [], []
    pairs = []
    for i, position in enumerate(positions):
        out = emit_deal(agents, position, rng, rows_per_traj=rows_per_traj)
        offset = len(z)
        for feats, zz, tr, tn, ph, lv in out["rows"]:
            x.append(feats); z.append(zz); deal.append(seed + i)
            traj.append(tr); turn.append(tn); phase.append(ph); live.append(lv)
        p = out["pair"]
        if p is not None:
            pairs.append((
                p["base_row"] + offset if p["base_row"] >= 0 else -1,
                p["alt_row"] + offset if p["alt_row"] >= 0 else -1,
                p["z_base"], p["z_alt"], p["base_sign"], p["alt_sign"], p["phase"],
                seed + i,
            ))
    pairs_arr = np.asarray(pairs, dtype=np.float64).reshape(-1, 8)
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez(
        tmp,
        x=np.stack(x).astype(np.float16), z=np.asarray(z, dtype=np.float32),
        deal=np.asarray(deal, dtype=np.int64), traj=np.asarray(traj, dtype=np.int8),
        turn=np.asarray(turn, dtype=np.int16), phase=np.asarray(phase, dtype=np.int8),
        live=np.asarray(live, dtype=bool), pairs=pairs_arr,
    )
    tmp.replace(out_path)
    return shard, len(z)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--deals-per-shard", type=int, default=1000)
    p.add_argument("--rows-per-traj", type=int, default=12)
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(s, args.deals_per_shard, args.rows_per_traj, str(out_dir))
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
            eta = el / done * (len(tasks) - done) / 60
            print(f"shard {shard} done ({n} rows) | {done}/{len(tasks)} | "
                  f"{rows:,} rows | {el / 60:.1f} min | ETA {eta:.0f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
