r"""piKL A/B shard runner (ADR-0037) — RESUMABLE.

Runs ONE shard of the seat-swap piKL-vs-un-searched-export tournament over a
contiguous slice of the Starting-Position Pool and writes that slice's raw
per-Position deltas. Because τ **is** the iter_06225 export, the mechanism read
(with-vs-without piKL) and the ship bar (vs shipped iter_06225) are the *same*
comparison — one opponent. A shard whose output already exists is SKIPPED, so a
crashed/killed run resumes by re-launching the same command. Pool the shards with
`scripts.pool_pmcpa_ab` (identical npz schema).

  py -m scripts.run_pikl_ab \
      --export-dir C:\workbench\tichu\data\runs\cotrain_wish_v5\export\iter_06225 \
      --pool C:\workbench\tichu\data\full_position_pool_s0_n20000.parquet \
      --n-deals 4000 --n-shards 20 --shard 3 --workers 10 \
      --worlds 20 --k 8 --lam 0.1 --q-scale 30 \
      --out-dir C:\workbench\tichu\data\runs\pikl_probe\lam_0p1
"""

import argparse
import sys
import threading
import time
from pathlib import Path

# Self-bootstrap THIS worktree's src onto sys.path (inherited by spawn workers) so
# `import tichu_*` resolves here, not in a sibling worktree's editable install.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _progress_reporter(total: int, shard_k: int, n_shards: int, *, every_s: float = 60.0):
    """Heartbeat reporter for one shard — `(cb, stop)`. `cb(n)` accumulates the
    Position count; a daemon thread prints liveness + ETA every `every_s` even
    when no chunk has completed (piKL's per-Position cold start is minutes)."""
    state = {"done": 0}
    lock = threading.Lock()
    t0 = time.monotonic()
    stop = threading.Event()

    def cb(n: int) -> None:
        with lock:
            state["done"] += n

    def beat() -> None:
        while not stop.wait(every_s):
            with lock:
                done = state["done"]
            el = max(time.monotonic() - t0, 1e-9)
            rate = done / el
            eta = (total - done) / rate / 60 if rate > 0 else float("nan")
            print(f"    [shard {shard_k}/{n_shards}] {done}/{total} positions "
                  f"| {el / 60:.1f} min | {rate * 60:.1f}/min | ETA {eta:.0f} min",
                  flush=True)

    threading.Thread(target=beat, daemon=True).start()
    return cb, stop


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Run the resumable piKL A/B: all shards by default (each "
                    "self-skips if done), or one shard with --shard.")
    p.add_argument("--export-dir", required=True, help="iter_06225 TorchScript export (= τ)")
    p.add_argument("--pool", required=True, help="full_position_pool parquet")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shard", type=int, default=None,
                   help="run ONLY this shard; omit to run all 0..n-shards-1")
    p.add_argument("--n-shards", type=int, required=True)
    p.add_argument("--n-deals", type=int, default=None,
                   help="cap total Positions before sharding (default: whole Pool)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--skill-decile", type=int, default=9)
    # piKL hyperparameters (ADR-0037 D/E/G).
    p.add_argument("--worlds", type=int, default=20)
    p.add_argument("--k", type=int, default=5)  # gate: max alt-rank 4 → k=5 covers all 663
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--q-scale", type=float, default=22.0)  # calibrated median spread
    args = p.parse_args(argv)

    if args.shard is not None and not 0 <= args.shard < args.n_shards:
        p.error(f"--shard must be in [0, {args.n_shards})")

    from tichu_eval.full_position_pool import load_full_position_pool
    from tichu_training.search.pikl import run_pikl_ab_shard

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = load_full_position_pool(Path(args.pool))
    total = len(pool) if args.n_deals is None else min(int(args.n_deals), len(pool))

    shards = [args.shard] if args.shard is not None else range(args.n_shards)
    hp = dict(worlds=args.worlds, k=args.k, lam=args.lam, q_scale=args.q_scale)
    for kk in shards:
        out_path = out_dir / f"shard_{kk:04d}.npz"
        if out_path.exists():
            print(f"shard {kk}/{args.n_shards}: already done — skipping", flush=True)
            continue
        base, extra = divmod(total, args.n_shards)
        start = kk * base + min(kk, extra)
        size = base + (1 if kk < extra else 0)
        positions = pool[start:start + size]
        if not positions:
            print(f"shard {kk}/{args.n_shards}: empty slice", flush=True)
            continue
        print(f"shard {kk}/{args.n_shards}: positions [{start}:{start + size}) "
              f"({len(positions)}), workers={args.workers}, worlds={args.worlds} "
              f"k={args.k} lam={args.lam} q_scale={args.q_scale}", flush=True)
        cb, stop = _progress_reporter(len(positions), kk, args.n_shards)
        try:
            run_pikl_ab_shard(
                export_dir=args.export_dir, positions=positions, out_path=out_path,
                workers=args.workers, skill_decile=args.skill_decile,
                hp=dict(hp, seed=kk),  # disjoint world-sampling stream per shard
                progress=cb,
            )
        finally:
            stop.set()
        print(f"shard {kk}/{args.n_shards}: wrote {out_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
