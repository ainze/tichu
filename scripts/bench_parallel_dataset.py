r"""Throughput benchmark for ParallelParquetBCDataset.

Strips away AWR / training so the signal is the raw IPC-bound producer
to consumer rate. Reports ex/s sampled over a short window after
warmup, so the worker-spawn cost is excluded from the measurement.

Usage (PowerShell):

  py -3.14 scripts\bench_parallel_dataset.py `
      --shards-dir C:\workbench\tichu\data\parquet_100k_v3 `
      --archive C:\workbench\tichu\data\archive.zst `
      --ratings C:\workbench\tichu\data\ratings_100k.parquet `
      --workers 10 --warmup 5000 --measure 30000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.parallel_dataset import ParallelParquetBCDataset
from tichu_training.featurizer import FEATURIZER_VERSION


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shards-dir", required=True)
    p.add_argument("--archive", required=True)
    p.add_argument("--ratings", default=None)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5000,
                   help="Examples to consume before starting the timer (excludes worker spawn).")
    p.add_argument("--measure", type=int, default=30000,
                   help="Examples to consume inside the timed window.")
    p.add_argument("--progress-every", type=int, default=5000,
                   help="Print instantaneous ex/s every N examples (0 = off).")
    p.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Forward a single keyword argument to ParallelParquetBCDataset.__init__. "
            "Repeatable. Value is parsed as int if it looks like one, else passed as "
            "str. Use this to probe knobs added by a future prototype (e.g. "
            "`--extra put_batch_size=64`) without having to edit this script."
        ),
    )
    args = p.parse_args()

    extra: dict[str, object] = {}
    for kv in args.extra:
        if "=" not in kv:
            raise SystemExit(f"--extra wants KEY=VALUE, got: {kv!r}")
        k, v = kv.split("=", 1)
        try:
            extra[k] = int(v)
        except ValueError:
            extra[k] = v

    ds = ParallelParquetBCDataset(
        args.shards_dir,
        archive_path=args.archive,
        ratings_path=args.ratings,
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
        num_workers=args.workers,
        **extra,
    )
    print(f"config: workers={args.workers} queue_maxsize={ds.queue_maxsize} extra={extra}")

    it = iter(ds)

    print(f"warmup: consuming {args.warmup} examples (worker spawn + ramp)...")
    t_warm_start = time.perf_counter()
    for i in range(args.warmup):
        next(it)
    t_warm = time.perf_counter() - t_warm_start
    print(f"warmup done: {args.warmup} ex in {t_warm:.2f}s ({args.warmup / t_warm:.0f} ex/s incl spawn)")

    print(f"measure: consuming {args.measure} examples...")
    t0 = time.perf_counter()
    last_probe = t0
    last_n = 0
    for i in range(args.measure):
        next(it)
        n = i + 1
        if args.progress_every and n % args.progress_every == 0:
            now = time.perf_counter()
            inst = (n - last_n) / max(1e-9, now - last_probe)
            print(f"  ... {n}/{args.measure}  inst={inst:.0f} ex/s")
            last_probe = now
            last_n = n
    elapsed = time.perf_counter() - t0
    rate = args.measure / elapsed
    print(f"\nMEASUREMENT: {args.measure} ex in {elapsed:.2f}s = {rate:.0f} ex/s")
    # Make sure the dataset shuts down workers cleanly.
    try:
        del it
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
