r"""Phase A spike — measure the consumer-side throughput ceiling.

Pre-load N BCExamples into RAM via the sequential dataset, then run an
AWR-shaped training-loop consumer over the cache repeatedly. The
measurement answers a single question:

  "If pre-featurised data were instantly available, how fast does the
  AWR training consumer actually go?"

That ceiling is the upper bound a disk-backed `MemmapBCDataset` could
chase. Gap between (ceiling) and (current ~6,500 ex/s replay-on-the-fly
baseline) is the realistic prize for the pre-featurise path. If the gap
is small (<2x), pre-featurise is not worth the disk/ADR cost regardless
of what the storage looks like.

The consumer here mirrors `awr_refine_epoch_streaming` step-for-step:
chunk to chunk_size, run baseline forward → advantages → AWR weights →
distribute into per-head buffers → fire weighted-BC step when a head's
buffer fills. Same operations, same batch size, same hyperparameters as
`configs/awr_smoke_100k_game.yaml`. The only difference is the data
source: instead of a `ParallelParquetBCDataset` queue, we cycle the
in-RAM `cache` list.

Usage (PowerShell):

  $env:PYTHONPATH = (Resolve-Path src).Path
  py -3.14 scripts\spike_pre_featurize_ceiling.py `
      --shards-dir C:\workbench\tichu\data\parquet_100k_v3 `
      --archive C:\workbench\tichu\data\archive.zst `
      --ratings C:\workbench\tichu\data\ratings_100k.parquet `
      --device cuda --load 50000 --measure 200000
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import torch

from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.awr.refine import _to_tensors
from tichu_training.awr.targets import target_value
from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.awr.weights import awr_weights
from tichu_training.bc.dataset import BCExample, ParquetBCDataset
from tichu_training.bc.heads import BCModel, HEAD_LOGIT_DIMS
from tichu_training.bc.loss import masked_cross_entropy
from tichu_training.featurizer import FEATURIZER_VERSION


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
    # Match configs/awr_smoke_100k_game.yaml so the ceiling is comparable
    # to what production AWR refine actually puts through this loop.
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=16_384)
    p.add_argument("--baseline-hidden", type=int, default=64)
    p.add_argument("--trunk-hidden", type=int, default=256)
    p.add_argument("--trunk-depth", type=int, default=2)
    p.add_argument("--trunk-out-dim", type=int, default=128)
    p.add_argument("--skill-dim", type=int, default=32)
    p.add_argument("--head-hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--max-weight", type=float, default=20.0)
    p.add_argument("--value-target", default="game")
    args = p.parse_args()

    device = torch.device(args.device)

    # -----------------------------------------------------------------
    # Step 1 — pre-load the cache via the sequential dataset.
    # Sequential is slow (~few hundred ex/s) but single-threaded and
    # avoids the worker-spawn cost; for a one-time fill that's fine.
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
    if n < args.batch_size * 4:
        raise SystemExit(
            f"cache too small ({n}) — bump --load to at least 4 batches"
        )

    feature_dim = cache[0].features.shape[0]

    # -----------------------------------------------------------------
    # Step 2 — build the model + baseline matching awr_smoke_100k_game.
    # Random init is fine for a throughput measurement; we just need the
    # forward/backward to do the same work it does in real training.
    # -----------------------------------------------------------------
    model = BCModel(
        feature_dim=feature_dim,
        skill_buckets=11,
        skill_dim=args.skill_dim,
        trunk_hidden=args.trunk_hidden,
        trunk_depth=args.trunk_depth,
        trunk_out_dim=args.trunk_out_dim,
        head_hidden=args.head_hidden,
    ).to(device)
    baseline = ValueBaseline(
        feature_dim=feature_dim, hidden=args.baseline_hidden,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    head_weights = {"play": 1.0, "wish": 0.3, "dragon_assignment": 0.2}

    per_head_buffer: dict[str, list[BCExample]] = defaultdict(list)

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

    def _process_chunk(buf: list[BCExample]) -> None:
        target_vals: list[float] = []
        keep: list[BCExample] = []
        for e in buf:
            t = target_value(
                round_outcome=e.round_outcome,
                game_won=e.game_won,
                kind=args.value_target,
            )
            if t is not None:
                target_vals.append(t)
                keep.append(e)
        if not keep:
            return
        features = np.stack([e.features for e in keep])
        outcomes = np.array(target_vals, dtype=np.float32)
        with torch.no_grad():
            preds = baseline(
                torch.from_numpy(features).to(device)
            ).cpu().numpy()
        advantages = outcomes - preds
        std = float(advantages.std())
        if std > 1e-8:
            advantages = (advantages - float(advantages.mean())) / std
        weights = awr_weights(
            advantages, beta=args.beta, max_weight=args.max_weight,
        )
        for ex, w in zip(keep, weights):
            new_ex = BCExample(
                decision_type=ex.decision_type,
                features=ex.features,
                target=ex.target,
                legal_mask=ex.legal_mask,
                sample_weight=float(ex.sample_weight * float(w)),
                skill_decile=ex.skill_decile,
                round_outcome=ex.round_outcome,
                game_won=ex.game_won,
            )
            head_buf = per_head_buffer[ex.decision_type]
            head_buf.append(new_ex)
            if len(head_buf) >= args.batch_size:
                _fire(ex.decision_type, head_buf)
                head_buf.clear()

    # -----------------------------------------------------------------
    # Step 3 — measurement loop. Cycle through the cache. Each yielded
    # example feeds the chunk buffer; every chunk_size examples we flush
    # the chunk through _process_chunk (which itself fires _fire on
    # each per-head buffer full).
    # -----------------------------------------------------------------
    print(f"measuring ceiling: {args.measure} ex through the AWR consumer "
          f"(cache cycles every {n} ex)")
    chunk: list[BCExample] = []
    consumed = 0
    cache_idx = 0
    last_n = 0
    last_t = time.perf_counter()
    t0 = last_t
    progress_every = max(args.chunk_size, args.measure // 10)

    while consumed < args.measure:
        ex = cache[cache_idx]
        cache_idx += 1
        if cache_idx >= n:
            cache_idx = 0
        chunk.append(ex)
        consumed += 1
        if len(chunk) >= args.chunk_size:
            _process_chunk(chunk)
            chunk.clear()
            if consumed - last_n >= progress_every:
                now = time.perf_counter()
                inst = (consumed - last_n) / max(1e-9, now - last_t)
                print(f"  ... {consumed}/{args.measure}  inst={inst:.0f} ex/s")
                last_n = consumed
                last_t = now
    if chunk:
        _process_chunk(chunk)
    for head, head_buf in list(per_head_buffer.items()):
        if head_buf:
            _fire(head, head_buf)
            head_buf.clear()
    # On CUDA the optimizer step is async; sync before stopping the
    # timer so we measure the actual completion, not the launch.
    if device.type != "cpu":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    rate = args.measure / max(1e-9, elapsed)
    print(f"\nCEILING: {args.measure} ex in {elapsed:.2f}s = {rate:.0f} ex/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
