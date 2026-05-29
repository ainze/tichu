r"""Phase B materialisation — write a BC corpus slice to disk as memmaps.

Streams BCExamples through `ParallelParquetBCDataset`, buckets them by
decision_type (play / wish / dragon_assignment), and writes one memmap
per (type, field) plus a small order index that preserves the original
emission order. A `manifest.json` records counts + version pins so the
reader can refuse to load a mismatched bundle.

Layout:

  <out_dir>/
    manifest.json
    play_features.dat              (N_play, 16624)  float32
    play_legal_mask.dat            (N_play, 1809)   uint8 (raw, not packed)
    play_meta.dat                  (N_play,)        structured
    wish_features.dat              (N_wish, 16624)  float32
    wish_legal_mask.dat            (N_wish, 14)     uint8
    wish_meta.dat                  (N_wish,)        structured
    dragon_assignment_features.dat (N_da, 16624)    float32
    dragon_assignment_legal_mask.dat (N_da, 2)      uint8
    dragon_assignment_meta.dat     (N_da,)          structured
    order.dat                      (N_total, 2)     uint32: (type_idx, row_idx)

Meta dtype:
  ("target", "u2"),
  ("sample_weight", "f4"),
  ("skill_decile", "u1"),
  ("round_outcome", "f4"),
  ("game_won", "i1"),  # -1=None, 0=False, 1=True

For ~100k examples the on-disk footprint is ~6.8 GB. For ~250k it's
~17 GB, which is what Phase B benches against (50k warmup + 200k
measure in one pass).

Usage (PowerShell):

  $env:PYTHONPATH = (Resolve-Path src).Path
  py -3.14 scripts\materialise_bc_subset.py `
      --shards-dir C:\workbench\tichu\data\parquet_100k_v3 `
      --archive C:\workbench\tichu\data\archive.zst `
      --ratings C:\workbench\tichu\data\ratings_100k.parquet `
      --out-dir C:\workbench\tichu\data\materialised_smoke `
      --max-examples 250000 --workers 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.dataset import BCExample
from tichu_training.bc.heads import HEAD_LOGIT_DIMS
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM, FEATURIZER_VERSION


META_DTYPE = np.dtype([
    ("target", "u2"),
    ("sample_weight", "f4"),
    ("skill_decile", "u1"),
    ("round_outcome", "f4"),
    ("game_won", "i1"),
])

# Stable ordering for type-index encoding in order.dat. Must match the
# reader's interpretation; bake into manifest.json so the reader doesn't
# rely on import order of HEAD_LOGIT_DIMS to recover the mapping.
TYPE_ORDER = ["play", "wish", "dragon_assignment"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shards-dir", required=True)
    p.add_argument("--archive", required=True)
    p.add_argument("--ratings", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-examples", type=int, required=True)
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deferred import — ParallelParquetBCDataset triggers a heavy chain
    # of imports + spawns workers on iter(). Keep argparse fast.
    from tichu_training.bc.parallel_dataset import ParallelParquetBCDataset

    ds = ParallelParquetBCDataset(
        args.shards_dir,
        archive_path=args.archive,
        ratings_path=args.ratings,
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
        num_workers=args.workers,
    )

    # Collect into per-type Python lists. For 250k examples the transient
    # peak is ~17 GB (real features, no pickle copy) — comfortable on a
    # 64 GB box. List-based collection avoids the headache of resizing
    # memmaps when the per-type split isn't known up front.
    per_type: dict[str, list[BCExample]] = {h: [] for h in TYPE_ORDER}
    order_pairs: list[tuple[int, int]] = []  # (type_idx, row_idx_within_type)
    type_idx = {h: i for i, h in enumerate(TYPE_ORDER)}

    print(f"streaming up to {args.max_examples} examples via "
          f"ParallelParquetBCDataset (workers={args.workers}) ...")
    t0 = time.perf_counter()
    n = 0
    last_t = t0
    last_n = 0
    for ex in ds:
        type_name = ex.decision_type
        if type_name not in per_type:
            # Unexpected decision_type — skip silently. Shouldn't happen
            # given HEAD_LOGIT_DIMS contracts but guard anyway.
            continue
        bucket = per_type[type_name]
        order_pairs.append((type_idx[type_name], len(bucket)))
        bucket.append(ex)
        n += 1
        if n >= args.max_examples:
            break
        if n - last_n >= 25_000:
            now = time.perf_counter()
            print(f"  ... {n}/{args.max_examples}  "
                  f"({(n - last_n) / max(1e-9, now - last_t):.0f} ex/s)")
            last_t = now
            last_n = n
    elapsed = time.perf_counter() - t0
    print(f"streamed {n} ex in {elapsed:.1f}s "
          f"({n / max(1e-9, elapsed):.0f} ex/s)")
    counts = {h: len(per_type[h]) for h in TYPE_ORDER}
    print(f"per-type counts: {counts}")

    # Write each type's three files. Pre-stacking via np.stack on the
    # list once is much cheaper than per-row memmap writes; the OS will
    # flush the resulting contiguous buffer to disk via the normal write
    # path.
    for type_name in TYPE_ORDER:
        examples = per_type[type_name]
        if not examples:
            print(f"  {type_name}: 0 examples (skipping write)")
            continue
        mask_dim = HEAD_LOGIT_DIMS[type_name]
        t_write = time.perf_counter()
        print(f"  {type_name}: writing {len(examples)} rows "
              f"(features {FEATURIZER_OUTPUT_DIM} f32, mask {mask_dim} u8) ...")
        feat = np.stack([e.features for e in examples])
        mask = np.stack(
            [np.asarray(e.legal_mask, dtype=np.uint8) for e in examples]
        )
        meta = np.zeros(len(examples), dtype=META_DTYPE)
        for i, e in enumerate(examples):
            meta[i]["target"] = e.target
            meta[i]["sample_weight"] = e.sample_weight
            meta[i]["skill_decile"] = e.skill_decile
            meta[i]["round_outcome"] = e.round_outcome
            meta[i]["game_won"] = (
                -1 if e.game_won is None else (1 if e.game_won else 0)
            )
        feat.tofile(out_dir / f"{type_name}_features.dat")
        mask.tofile(out_dir / f"{type_name}_legal_mask.dat")
        meta.tofile(out_dir / f"{type_name}_meta.dat")
        # Free the Python references right after writing so the next
        # type's stack doesn't compete for RAM. The list still holds
        # references; clear it.
        per_type[type_name] = []
        del feat, mask, meta, examples
        dt = time.perf_counter() - t_write
        bytes_written = (
            (FEATURIZER_OUTPUT_DIM * 4 + mask_dim + META_DTYPE.itemsize)
            * counts[type_name]
        )
        print(f"    wrote {bytes_written / 1e9:.2f} GB in {dt:.1f}s "
              f"({bytes_written / 1e9 / max(1e-9, dt):.2f} GB/s)")

    # order.dat: (N_total, 2) uint32. Preserves original stream order so
    # the reader can re-emit a mixed-type sequence that matches what the
    # parallel dataset produced.
    order_arr = np.array(order_pairs, dtype=np.uint32)
    order_arr.tofile(out_dir / "order.dat")
    print(f"  order.dat: {order_arr.nbytes / 1e6:.2f} MB ({len(order_arr)} rows)")

    # Manifest: everything the reader needs to (a) sanity-check version
    # pins, (b) memmap each file with the right shape/dtype, (c) walk
    # `order.dat` to recover the original sequence.
    manifest = {
        "schema_version": 1,
        "featurizer_version": FEATURIZER_VERSION,
        "action_space_version": ACTION_SPACE_VERSION,
        "feature_dim": FEATURIZER_OUTPUT_DIM,
        "head_logit_dims": dict(HEAD_LOGIT_DIMS),
        "type_order": TYPE_ORDER,
        "meta_dtype": [(n, str(t)) for n, t in META_DTYPE.descr],
        "counts": counts,
        "total": n,
        "files": {
            type_name: {
                "features": f"{type_name}_features.dat",
                "legal_mask": f"{type_name}_legal_mask.dat",
                "meta": f"{type_name}_meta.dat",
                "shape_features": [counts[type_name], FEATURIZER_OUTPUT_DIM],
                "shape_legal_mask": [counts[type_name],
                                     HEAD_LOGIT_DIMS[type_name]],
            }
            for type_name in TYPE_ORDER if counts[type_name] > 0
        },
        "order_file": "order.dat",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest.json written. Total {n} examples, out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
