r"""Generate a fixed, seeded Starting-Position Pool parquet for the tournament eval.

The tournament eval (`eval_matrix --mode tournament`) consumes a pool of
post-schupfen starting `GameState`s identified exactly by `(seed, n)`
(see `tichu_eval.starting_position_pool`). There is no other CLI that
materialises it, so this script wraps the two library functions.

Usage (PowerShell, from the worktree with `$env:PYTHONPATH = "src"`):

    py -3.14 scripts/gen_starting_position_pool.py `
      --seed 0 --n 500 `
      --output C:\workbench\tichu\data\starting_position_pool_s0_n500.parquet

Same `(seed, n)` always produces a byte-identical file, so the pool is
reproducible across machines and runs.
"""

import argparse
import sys
from pathlib import Path

from tichu_eval.starting_position_pool import (
    generate_starting_position_pool,
    save_starting_position_pool,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0, help="Pool seed (default: 0).")
    p.add_argument("--n", type=int, required=True, help="Number of starting positions.")
    p.add_argument("--output", required=True, metavar="FILE", help="Output parquet path.")
    args = p.parse_args(argv)

    deals = generate_starting_position_pool(seed=args.seed, n=args.n)
    out = Path(args.output)
    save_starting_position_pool(deals, out)
    print(f"wrote {out} ({len(deals)} deals, seed={args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
