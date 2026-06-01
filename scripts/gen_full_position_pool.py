r"""Generate a fixed, seeded Full-strength Starting-Position Pool parquet.

The Full-strength tournament (`eval_matrix --mode tournament`, the default
`variant: full_strength`) consumes a pool of pre-Schupfen, deal-time
`GameState`s with order-preserving hands, so each seat's Grand-Tichu Prefix
(first 8 cards) is reconstructible. Identity is exactly `(seed, n)`
(see `tichu_eval.full_position_pool`). See ADR-0025.

Usage (PowerShell, from the worktree with `$env:PYTHONPATH = "src"`):

    py -3.14 scripts/gen_full_position_pool.py `
      --seed 0 --n 2000 `
      --output C:\workbench\tichu\data\full_position_pool_s0_n2000.parquet

Same `(seed, n)` always produces a byte-identical file, so the pool is
reproducible across machines and runs.
"""

import argparse
import sys
from pathlib import Path

from tichu_eval.full_position_pool import (
    generate_full_position_pool,
    save_full_position_pool,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0, help="Pool seed (default: 0).")
    p.add_argument("--n", type=int, required=True, help="Number of starting positions.")
    p.add_argument("--output", required=True, metavar="FILE", help="Output parquet path.")
    args = p.parse_args(argv)

    pool = generate_full_position_pool(seed=args.seed, n=args.n)
    out = Path(args.output)
    save_full_position_pool(pool, out)
    print(f"wrote {out} ({len(pool)} positions, seed={args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
