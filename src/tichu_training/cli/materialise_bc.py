r"""`materialise_bc` CLI.

One-shot pre-featurisation pass: drives `ParallelParquetBCDataset` to
stream BCExamples through the engine replay, then writes the per-type
memmap bundle (`MemmapBCDataset`-readable) to disk. Replaces the
~75%-of-CPU `replay_round` cost at training time with a
~1.0 GB/s sequential memmap read.

The bundle is version-pinned (featurizer + action-space + schema). The
matching reader (`tichu_training.bc.materialised.MemmapBCDataset`)
refuses to load on mismatch, so a stale bundle never silently corrupts
a training run.

Usage:

  python -m tichu_training.cli.materialise_bc \
      --shards-dir DATA/parquet_100k_v3 \
      --archive    DATA/archive.zst \
      --ratings    DATA/ratings_100k.parquet \
      --out-dir    DATA/materialised_100k \
      --max-examples 6_200_000 \
      --workers 10

See [ADR-0014](../../../docs/adr/0014-pre-featurise-bc-corpus.md) for
the cost/benefit analysis vs ADR-0011's replay-on-the-fly path.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.materialised import materialise
from tichu_training.featurizer import FEATURIZER_VERSION


log = logging.getLogger("materialise_bc")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Materialise a BCExample slice to per-type numpy memmaps "
            "(features / legal_mask / meta) plus order.dat + manifest.json. "
            "Output is readable via `MemmapBCDataset`."
        )
    )
    p.add_argument("--shards-dir", required=True, metavar="DIR",
                   help="Parquet shards directory (output of parse_bsw).")
    p.add_argument("--archive", required=True, metavar="PATH",
                   help="BSW indexed zstd archive (.zst + .idx).")
    p.add_argument("--ratings", default=None, metavar="PATH",
                   help="Optional TrueSkill ratings parquet for live "
                        "skill_decile join. Without it, all rows get the "
                        "neutral decile.")
    p.add_argument("--out-dir", required=True, metavar="DIR",
                   help="Where the manifest + memmaps land. Created if "
                        "absent. Existing files are overwritten.")
    p.add_argument("--max-examples", type=int, required=True, metavar="N",
                   help="Cap on BCExamples to write. Use a small N for "
                        "smoke (250k = ~17 GB), or the full corpus row "
                        "count for production.")
    p.add_argument("--workers", type=int, default=10, metavar="N",
                   help="ParallelParquetBCDataset worker processes "
                        "(default 10). Each owns a hash-sharded slice of "
                        "the game-id space.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deferred import: ParallelParquetBCDataset triggers a heavy chain
    # (engine modules, ratings load, manifest build). Keep argparse fast.
    from tichu_training.bc.parallel_dataset import ParallelParquetBCDataset

    log.info("opening ParallelParquetBCDataset (workers=%d) ...", args.workers)
    ds = ParallelParquetBCDataset(
        args.shards_dir,
        archive_path=args.archive,
        ratings_path=args.ratings,
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
        num_workers=args.workers,
    )

    counts = materialise(ds, out_dir, max_examples=args.max_examples)
    log.info("done: counts=%s, out_dir=%s", counts, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
