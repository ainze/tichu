r"""Phase A spike — measure the BC training consumer-side throughput ceiling.

Pre-load N BCExamples into RAM via the sequential dataset, then run a
BC-shaped training-loop consumer over the cache repeatedly. The
measurement answers a single question:

  "If pre-featurised data were instantly available, how fast does the
  BC training loop actually go at this model size?"

That ceiling is the upper bound a disk-backed `MemmapBCDataset` could
chase. Gap between (ceiling) and (current ~6,500 ex/s replay-on-the-fly
baseline) is the realistic prize for pre-featurise. If the gap is
small (<2×), pre-featurise isn't worth the disk/ADR cost.

The consumer here mirrors `tichu_training.bc.training.train_one_epoch`
step-for-step: route every example into a per-decision-type buffer,
fire a step when a buffer reaches batch_size, drain partial buffers at
the end. No AWR baseline/advantage compute. No early-stop, no CSV
logging — just the hot path.

Two arch shapes the BC pipeline actually runs at:
  --arch current  → trunk 1024×4, head 256, B=1024  (bc_full_100k.yaml)
  --arch wider    → trunk 2048×6, head 512, B=1024  (queued wider-BC followup)

Usage (PowerShell):

  $env:PYTHONPATH = (Resolve-Path src).Path
  py -3.14 scripts\spike_bc_ceiling.py `
      --shards-dir C:\workbench\tichu\data\parquet_100k_v3 `
      --archive C:\workbench\tichu\data\archive.zst `
      --ratings C:\workbench\tichu\data\ratings_100k.parquet `
      --device cuda --load 50000 --measure 200000 --arch current
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import torch

from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.awr.refine import _to_tensors
from tichu_training.bc.dataset import BCExample, ParquetBCDataset
from tichu_training.bc.heads import BCModel, HEAD_LOGIT_DIMS
from tichu_training.bc.loss import masked_cross_entropy
from tichu_training.featurizer import FEATURIZER_VERSION


# Arch presets matching the actual training configs in this repo.
ARCH_PRESETS = {
    # bc_full_100k.yaml — current "full" BC training arch.
    "current": dict(
        trunk_hidden=1024, trunk_depth=4, trunk_out_dim=512,
        skill_dim=64, head_hidden=256, batch_size=1024,
        lr=3.0e-4,
    ),
    # Queued wider-BC followup — handoff-bc-wider-trunk.md.
    "wider": dict(
        trunk_hidden=2048, trunk_depth=6, trunk_out_dim=512,
        skill_dim=64, head_hidden=512, batch_size=1024,
        lr=1.5e-4,
    ),
    # configs/awr_smoke_100k_game.yaml — the size the original (wrong)
    # Phase A spike measured. Kept here only for cross-reference; not
    # representative of the actual BC training compute.
    "smoke": dict(
        trunk_hidden=256, trunk_depth=2, trunk_out_dim=128,
        skill_dim=32, head_hidden=64, batch_size=256,
        lr=1.0e-4,
    ),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shards-dir", required=True)
    p.add_argument("--archive", required=True)
    p.add_argument("--ratings", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--load", type=int, default=50_000,
                   help="Examples to pre-load into RAM (sequential, single-thread).")
    p.add_argument("--measure", type=int, default=200_000,
                   help="Examples to consume in the timed window (cycles over the cache).")
    p.add_argument("--arch", choices=sorted(ARCH_PRESETS), default="current",
                   help="Which BC arch preset to instantiate. 'current' = bc_full_100k.yaml; "
                        "'wider' = queued 2048×6 followup; 'smoke' = the cross-reference smoke arch.")
    # Optional per-knob overrides — for sweeping a single dimension.
    p.add_argument("--trunk-hidden", type=int)
    p.add_argument("--trunk-depth", type=int)
    p.add_argument("--trunk-out-dim", type=int)
    p.add_argument("--skill-dim", type=int)
    p.add_argument("--head-hidden", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    args = p.parse_args()

    arch = dict(ARCH_PRESETS[args.arch])
    for k in ("trunk_hidden", "trunk_depth", "trunk_out_dim",
              "skill_dim", "head_hidden", "batch_size", "lr"):
        v = getattr(args, k)
        if v is not None:
            arch[k] = v
    print(f"arch={args.arch}  {arch}")

    device = torch.device(args.device)

    # -----------------------------------------------------------------
    # Step 1 — pre-load the cache via the sequential dataset.
    # -----------------------------------------------------------------
    print(f"pre-loading {args.load} examples via sequential dataset ...")
    ds = ParquetBCDataset(
        args.shards_dir,
        archive_path=args.archive,
        ratings_path=args.ratings,
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    t_load_start = time.perf_counter()
    cache: list[BCExample] = []
    for ex in ds:
        cache.append(ex)
        if len(cache) >= args.load:
            break
    t_load = time.perf_counter() - t_load_start
    n = len(cache)
    print(f"  loaded {n} ex in {t_load:.1f}s "
          f"({n / max(1e-9, t_load):.0f} ex/s sequential)")
    if n < arch["batch_size"] * 4:
        raise SystemExit(
            f"cache too small ({n}) for batch_size={arch['batch_size']} — "
            f"bump --load to at least 4 batches"
        )
    feature_dim = cache[0].features.shape[0]

    # -----------------------------------------------------------------
    # Step 2 — build the model. Match BCModel constructor exactly.
    # Random init is fine; we want the throughput, not the loss curve.
    # -----------------------------------------------------------------
    model = BCModel(
        feature_dim=feature_dim,
        skill_buckets=11,
        skill_dim=arch["skill_dim"],
        trunk_hidden=arch["trunk_hidden"],
        trunk_depth=arch["trunk_depth"],
        trunk_out_dim=arch["trunk_out_dim"],
        head_hidden=arch["head_hidden"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params on {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=arch["lr"])
    head_weights = {h: 1.0 for h in HEAD_LOGIT_DIMS}
    batch_size = arch["batch_size"]

    head_buffers: dict[str, list[BCExample]] = defaultdict(list)

    def _fire(head: str, batch: list[BCExample]) -> None:
        tensors = _to_tensors(batch)
        if device.type != "cpu":
            tensors = {k: v.to(device, non_blocking=True) for k, v in tensors.items()}
        out = model(tensors["features"], tensors["skill_decile"])
        logits = out[head]
        per_head_loss = masked_cross_entropy(
            logits, tensors["target"], tensors["legal_mask"],
            tensors["sample_weight"],
        )
        total = head_weights[head] * per_head_loss
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

    # -----------------------------------------------------------------
    # Step 3 — measurement loop. Cycle the cache through the BC consumer.
    # -----------------------------------------------------------------
    print(f"measuring ceiling: {args.measure} ex through BC consumer "
          f"(cache cycles every {n} ex)")
    consumed = 0
    cache_idx = 0
    last_n = 0
    last_t = time.perf_counter()
    t0 = last_t
    progress_every = max(batch_size * 4, args.measure // 10)

    while consumed < args.measure:
        ex = cache[cache_idx]
        cache_idx += 1
        if cache_idx >= n:
            cache_idx = 0
        head_buf = head_buffers[ex.decision_type]
        head_buf.append(ex)
        consumed += 1
        if len(head_buf) >= batch_size:
            _fire(ex.decision_type, head_buf)
            head_buf.clear()
        if consumed - last_n >= progress_every:
            now = time.perf_counter()
            inst = (consumed - last_n) / max(1e-9, now - last_t)
            print(f"  ... {consumed}/{args.measure}  inst={inst:.0f} ex/s")
            last_n = consumed
            last_t = now
    # Drain partial buffers — mirrors `train_one_epoch` end-of-epoch.
    for head, head_buf in list(head_buffers.items()):
        if head_buf:
            _fire(head, head_buf)
            head_buf.clear()
    # On CUDA the optimizer step is async; sync before stopping the timer.
    if device.type != "cpu":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    rate = args.measure / max(1e-9, elapsed)
    print(f"\nCEILING ({args.arch}): {args.measure} ex in {elapsed:.2f}s = {rate:.0f} ex/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
