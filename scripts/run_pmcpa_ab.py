r"""pMCPA A/B shard runner (ADR-0036) — RESUMABLE.

Runs ONE shard of the seat-swap adapted-vs-opponent tournament over a contiguous
slice of the Starting-Position Pool and writes that slice's raw per-Position
deltas. A shard whose output already exists is SKIPPED, so a crashed or killed
run resumes by re-launching the same command — only missing shards recompute.
Pool the shards into one exact bootstrap with `scripts.pool_pmcpa_ab`.

  # adapted vs non-adapted theta_o (the mechanism A/B), shard 3 of 20:
  py -m scripts.run_pmcpa_ab \
      --theta-o-dir C:\workbench\tichu\data\runs\cotrain_vine_v1\warm \
      --opponent offline --pool C:\workbench\tichu\data\full_position_pool_s0_n20000.parquet \
      --n-deals 8600 --n-shards 20 --shard 3 --workers 10 \
      --worlds 128 --steps 3 --out-dir C:\workbench\tichu\data\runs\pmcpa_probe\ab_offline

  # vs the shipped iter_06225 export (the ship bar):
  py -m scripts.run_pmcpa_ab ... --opponent export \
      --export-dir C:\workbench\tichu\data\runs\cotrain_wish_v5\export\iter_06225 ...
"""

import argparse
import sys
import threading
import time
from pathlib import Path


def _progress_reporter(total: int, shard_k: int, n_shards: int, *, every_s: float = 60.0):
    """A heartbeat progress reporter for one shard. Returns `(cb, stop)`:

      * `cb(n)` is the `collect_pair_deltas` callback — fires on chunk completion
        and just accumulates the Position count (no printing).
      * a daemon thread prints a status line every `every_s` seconds REGARDLESS of
        completions, so the ~per-position cold start (no chunk done for ~15+ min)
        still shows liveness + elapsed instead of going dark.
      * `stop` (an Event) ends the thread; the caller sets it when the shard returns.
    """
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
            rate = done / el  # positions/sec
            eta = (total - done) / rate / 60 if rate > 0 else float("nan")
            print(f"    [shard {shard_k}/{n_shards}] {done}/{total} positions "
                  f"| {el / 60:.1f} min elapsed | {rate * 60:.1f}/min | ETA {eta:.0f} min",
                  flush=True)

    threading.Thread(target=beat, daemon=True).start()
    return cb, stop

# Self-bootstrap THIS worktree's src onto sys.path (and thus onto every spawn
# worker, which inherits it) so `import tichu_*` resolves here, not in a sibling
# worktree's editable install — no PYTHONPATH needed to launch.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Run the resumable pMCPA A/B: all shards by default (each "
                    "self-skips if done), or one shard with --shard.",
    )
    p.add_argument("--theta-o-dir", required=True,
                   help="dir of recovered theta_o .bin weights ({net}_{tag}.bin)")
    p.add_argument("--tag", default="iter06225", help=".bin filename tag")
    p.add_argument("--opponent", choices=("offline", "export"), default="offline",
                   help="'offline' = non-adapted theta_o control; 'export' = ship bar")
    p.add_argument("--export-dir", default=None, help="TorchScript export dir (opponent=export)")
    p.add_argument("--pool", required=True, help="full_position_pool parquet")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shard", type=int, default=None,
                   help="run ONLY this shard (for manual cross-process fan-out); "
                        "omit to run all shards 0..n-shards-1 sequentially")
    p.add_argument("--n-shards", type=int, required=True)
    p.add_argument("--n-deals", type=int, default=None,
                   help="cap total Positions before sharding (default: whole Pool)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--skill-decile", type=int, default=9)
    # pMCPA hyperparameters (ADR-0036 Decisions C/D/E/F).
    p.add_argument("--worlds", type=int, default=128)
    p.add_argument("--decisions-per-world", type=int, default=4)
    p.add_argument("--branches", type=int, default=4)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.01)  # diag: 0.05 craters play (ADR-0036)
    p.add_argument("--kl-coef", type=float, default=1.0)
    p.add_argument("--clip-eps", type=float, default=0.1)
    p.add_argument("--emit-branches", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-abs-advantage", type=float, default=0.0)
    p.add_argument("--adapt-trunk", action="store_true",
                   help="adapt Trunk+play head (the capacity escalation); default head-only")
    p.add_argument("--oracle", action="store_true",
                   help="DIAGNOSTIC: adapt on the TRUE world (danger-(b)=0 ceiling, ADR-0036)")
    args = p.parse_args(argv)

    if args.shard is not None and not 0 <= args.shard < args.n_shards:
        p.error(f"--shard must be in [0, {args.n_shards})")

    from tichu_eval.full_position_pool import load_full_position_pool
    from tichu_training.ppo.pmcpa import run_pmcpa_ab_shard

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = load_full_position_pool(Path(args.pool))
    total = len(pool) if args.n_deals is None else min(int(args.n_deals), len(pool))

    shards = [args.shard] if args.shard is not None else range(args.n_shards)
    hp_common = dict(
        worlds=args.worlds, decisions_per_world=args.decisions_per_world,
        branches=args.branches, steps=args.steps, lr=args.lr, kl_coef=args.kl_coef,
        clip_eps=args.clip_eps, emit_branches=args.emit_branches,
        min_abs_advantage=args.min_abs_advantage, adapt_trunk=args.adapt_trunk,
        oracle=args.oracle,
    )
    for k in shards:
        out_path = out_dir / f"shard_{k:04d}.npz"
        if out_path.exists():
            print(f"shard {k}/{args.n_shards}: already done ({out_path.name}) — skipping",
                  flush=True)
            continue
        # Contiguous, near-equal, non-overlapping slices covering [0, total).
        base, extra = divmod(total, args.n_shards)
        start = k * base + min(k, extra)
        size = base + (1 if k < extra else 0)
        positions = pool[start:start + size]
        if not positions:
            print(f"shard {k}/{args.n_shards}: empty slice — nothing to do", flush=True)
            continue
        print(f"shard {k}/{args.n_shards}: positions [{start}:{start + size}) "
              f"({len(positions)}) vs {args.opponent}, workers={args.workers}, "
              f"K={args.worlds} steps={args.steps}", flush=True)
        cb, stop = _progress_reporter(len(positions), k, args.n_shards)
        try:
            run_pmcpa_ab_shard(
                theta_o_dir=args.theta_o_dir, opponent=args.opponent, positions=positions,
                out_path=out_path, workers=args.workers, export_dir=args.export_dir,
                skill_decile=args.skill_decile, tag=args.tag,
                hp=dict(hp_common, seed=k),  # disjoint world-sampling stream per shard
                progress=cb,
            )
        finally:
            stop.set()  # end this shard's heartbeat thread
        print(f"shard {k}/{args.n_shards}: wrote {out_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
