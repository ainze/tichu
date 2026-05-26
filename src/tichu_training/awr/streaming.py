"""Streaming AWR pipeline — never materialises the full dataset.

The non-streaming path (`tichu_training.awr.refine.awr_refine_epoch`)
holds every BCExample in RAM. On a 2.4M-game / ~1.5B-decision corpus
that would need ~100 TB. This module fixes that with a two-pass
streaming pipeline:

  Pass 1: `collect_held_out`  →  hash-based partition into held-out
          (capped) vs train. Single pass over the dataset.
  Pass 2: `fit_value_baseline_streaming`  →  SGD fit on the train
          partition only, multiple SGD steps per chunk so the small
          baseline still gets enough updates without re-streaming.
  Pass N+: `awr_refine_epoch_streaming` (once per AWR epoch)  →  fresh
          pass through the dataset; per chunk, run the baseline forward
          to get advantages, compute AWR weights with per-chunk
          max-subtract, route examples into per-head buffers, and fire
          a weighted-BC step when each buffer fills.

The held-out partition predicate is deterministic across passes: a
given example always lands in the same bucket. So even though we only
*store* `max_held_out` examples for win_rate_proxy, every example
matching the predicate is excluded from training on every pass — the
model never sees them.

Per-chunk max-subtract is a slight semantic shift vs the in-memory
`awr_weights` (which subtracts the dataset-wide max). Relative weights
within a chunk are identical; absolute scaling differs across chunks
by a multiplicative constant. With `chunk_size >> batch_size`, the
practical effect on the gradient is small — particularly with the
`max_weight` clip already in play.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from tichu_training.awr.refine import (
    _CSV_FIELDS,
    _append_csv,
    _next_step,
    _play_head_top1,
    _to_tensors,
)
from tichu_training.awr.weights import awr_weights
from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.bc.dataset import BCExample
from tichu_training.bc.heads import BCModel, HEAD_LOGIT_DIMS
from tichu_training.bc.loss import masked_cross_entropy


DatasetFactory = Callable[[], Iterable[BCExample]]
HeldOutFilter = Callable[[BCExample], bool]


# ---------------------------------------------------------------------------
# Held-out partitioning
# ---------------------------------------------------------------------------


def _partition_key(ex: BCExample) -> int:
    """Cheap, stable hash. Mixes decision_type + target + a slice of the
    feature vector (32 floats = 128 bytes) for diversity.

    Hashing the full 16k-float feature vector would cost ~70 μs each
    (×64M examples = an hour). The slice is fast (~100 ns) and still
    gives essentially zero collision probability for game states that
    differ at all.
    """
    return hash((ex.decision_type, ex.target, ex.features[:32].tobytes()))


def make_held_out_filter(
    *, fraction: float, salt: int = 0,
) -> tuple[HeldOutFilter, int]:
    """Return a (predicate, bucket_count) pair. The predicate is `True`
    iff an example lands in the held-out partition. `salt` lets you
    rotate which examples are held out across runs (e.g. cross-fold).

    `fraction` must be in [0, 1]. Internally quantised to per-mille
    (0.001) precision so an int comparison is enough on the hot path.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0,1], got {fraction}")
    threshold = int(fraction * 1000)

    def is_held_out(ex: BCExample) -> bool:
        return ((_partition_key(ex) + salt) & 0x7FFFFFFF) % 1000 < threshold

    return is_held_out, threshold


def collect_held_out(
    dataset_factory: DatasetFactory,
    *,
    fraction: float,
    max_held_out: int,
    salt: int = 0,
    show_progress: bool = True,
) -> tuple[list[BCExample], HeldOutFilter]:
    """Pass 1: scan the stream and capture held-out examples (capped).

    Returns the captured subset *and* a predicate to filter held-out
    examples on subsequent passes — even examples beyond `max_held_out`
    that match the predicate are skipped during training, so the
    model never trains on something that could be in the win_rate_proxy
    eval pool.
    """
    is_held_out, _ = make_held_out_filter(fraction=fraction, salt=salt)
    held: list[BCExample] = []

    bar = tqdm(
        unit="ex", dynamic_ncols=True,
        desc="awr held-out scan", disable=not show_progress,
    )
    try:
        for ex in dataset_factory():
            bar.update(1)
            if len(held) >= max_held_out:
                # We've got enough for the eval pool. The predicate
                # remains stable, so we can stop scanning now — any
                # additional held-out matches will still be skipped
                # during training passes via the same hash predicate.
                break
            if is_held_out(ex):
                held.append(ex)
    finally:
        bar.close()

    return held, is_held_out


# ---------------------------------------------------------------------------
# Streaming value-baseline fit
# ---------------------------------------------------------------------------


def fit_value_baseline_streaming(
    baseline: ValueBaseline,
    dataset_factory: DatasetFactory,
    *,
    batch_size: int,
    lr: float,
    chunk_size: int = 16_384,
    sgd_steps_per_chunk: int = 3,
    held_out_filter: HeldOutFilter | None = None,
    show_progress: bool = True,
) -> float:
    """SGD-fit the value baseline by streaming the dataset once.

    Each chunk of `chunk_size` examples is converted to tensors once
    and trained over `sgd_steps_per_chunk` mini-epochs (so the small
    baseline still sees enough updates per example without re-streaming
    the whole corpus). Returns the final batch MSE.
    """
    optimizer = torch.optim.Adam(baseline.parameters(), lr=lr)
    final_mse = float("inf")
    chunk: list[BCExample] = []
    skip = held_out_filter or (lambda _e: False)

    bar = tqdm(
        unit="ex", dynamic_ncols=True,
        desc="value baseline (stream)", disable=not show_progress,
    )

    def _fit_chunk(buf: list[BCExample]) -> None:
        nonlocal final_mse
        feats = torch.from_numpy(np.stack([e.features for e in buf]))
        outs = torch.from_numpy(np.array([e.round_outcome for e in buf], dtype=np.float32))
        n = len(buf)
        for _ in range(sgd_steps_per_chunk):
            for i in range(0, n, batch_size):
                preds = baseline(feats[i : i + batch_size])
                loss = F.mse_loss(preds, outs[i : i + batch_size])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                final_mse = float(loss.detach())

    try:
        for ex in dataset_factory():
            bar.update(1)
            if skip(ex):
                continue
            chunk.append(ex)
            if len(chunk) >= chunk_size:
                _fit_chunk(chunk)
                chunk.clear()
                bar.set_postfix(last_mse=f"{final_mse:.4f}", refresh=False)
        if chunk:
            _fit_chunk(chunk)
    finally:
        bar.close()

    return final_mse


# ---------------------------------------------------------------------------
# Streaming AWR refine epoch
# ---------------------------------------------------------------------------


def awr_refine_epoch_streaming(
    model: BCModel,
    dataset_factory: DatasetFactory,
    baseline: ValueBaseline,
    optimizer: torch.optim.Optimizer,
    *,
    beta: float,
    max_weight: float,
    batch_size: int,
    log_path: Path,
    head_weights: dict[str, float] | None = None,
    held_out_examples: Sequence[BCExample] | None = None,
    held_out_filter: HeldOutFilter | None = None,
    chunk_size: int = 16_384,
    show_progress: bool = True,
) -> dict[str, float | None]:
    """One streaming AWR refinement pass over the dataset.

    Per chunk: baseline forward → advantages → per-chunk AWR weights →
    distribute examples into per-head buffers → fire a weighted-BC
    step when a head's buffer fills. Partial buffers drain at
    end-of-stream so no examples are silently dropped.

    `held_out_filter` (if set) is applied to every streamed example;
    matches are skipped. `held_out_examples` is the captured eval pool
    used to compute `win_rate_proxy` after the pass.
    """
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")
    head_weights = head_weights or {h: 1.0 for h in HEAD_LOGIT_DIMS}
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    skip = held_out_filter or (lambda _e: False)

    last_total = float("inf")
    step = _next_step(log_path)
    weight_running_total = 0.0
    weight_running_count = 0
    model_device = next(model.parameters()).device

    per_head_buffer: dict[str, list[BCExample]] = defaultdict(list)

    def _fire(head: str, batch: list[BCExample]) -> None:
        nonlocal last_total, step, weight_running_total, weight_running_count
        tensors = _to_tensors(batch)
        if model_device.type != "cpu":
            tensors = {k: v.to(model_device, non_blocking=True) for k, v in tensors.items()}
        out = model(tensors["features"], tensors["skill_decile"])
        logits = out[head]
        per_head_loss = masked_cross_entropy(
            logits, tensors["target"], tensors["legal_mask"], tensors["sample_weight"],
        )
        total = head_weights[head] * per_head_loss
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        with torch.no_grad():
            pred = logits.masked_fill(~tensors["legal_mask"], float("-inf")).argmax(dim=-1)
            acc = (pred == tensors["target"]).float().mean().item()
        batch_awr_mean = float(tensors["sample_weight"].mean())
        weight_running_total += batch_awr_mean * len(batch)
        weight_running_count += len(batch)

        row = {f: 0.0 for f in _CSV_FIELDS}
        row["step"] = step
        row["loss_total"] = float(total.detach())
        row[f"loss_{head}"] = float(per_head_loss.detach())
        row[f"acc_{head}"] = acc
        row["awr_weight_mean"] = batch_awr_mean
        _append_csv(log_path, [row])
        last_total = row["loss_total"]
        step += 1

    def _process_chunk(buf: list[BCExample]) -> None:
        features = np.stack([e.features for e in buf])
        outcomes = np.array([e.round_outcome for e in buf], dtype=np.float32)
        with torch.no_grad():
            preds = baseline(torch.from_numpy(features)).cpu().numpy()
        advantages = outcomes - preds
        weights = awr_weights(advantages, beta=beta, max_weight=max_weight)
        for ex, w in zip(buf, weights):
            new_ex = BCExample(
                decision_type=ex.decision_type,
                features=ex.features,
                target=ex.target,
                legal_mask=ex.legal_mask,
                sample_weight=float(ex.sample_weight * float(w)),
                skill_decile=ex.skill_decile,
                round_outcome=ex.round_outcome,
            )
            head_buf = per_head_buffer[ex.decision_type]
            head_buf.append(new_ex)
            if len(head_buf) >= batch_size:
                _fire(ex.decision_type, head_buf)
                head_buf.clear()

    chunk: list[BCExample] = []
    bar = tqdm(
        unit="ex", dynamic_ncols=True,
        desc="awr refine (stream)", disable=not show_progress,
    )
    try:
        for ex in dataset_factory():
            bar.update(1)
            if skip(ex):
                continue
            chunk.append(ex)
            if len(chunk) >= chunk_size:
                _process_chunk(chunk)
                chunk.clear()
                bar.set_postfix(loss=f"{last_total:.3f}", refresh=False)
        if chunk:
            _process_chunk(chunk)
        # Drain per-head partial buffers — every example pulled from
        # the stream must contribute exactly one row, even if its head's
        # last batch ran short.
        for head, head_buf in per_head_buffer.items():
            if head_buf:
                _fire(head, head_buf)
                head_buf.clear()
    finally:
        bar.close()

    summary: dict[str, float | None] = {
        "loss_total": last_total,
        "avg_weight": (weight_running_total / weight_running_count) if weight_running_count else 0.0,
        "win_rate_proxy": None,
    }
    if held_out_examples:
        summary["win_rate_proxy"] = _play_head_top1(model, held_out_examples)
    return summary
