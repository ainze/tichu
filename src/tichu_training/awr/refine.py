"""AWR refinement training step.

`awr_refine_epoch` is the AWR-side analogue of `train_one_epoch`: it
computes per-example advantages from a value baseline, derives clipped
AWR weights, multiplies them into each example's `sample_weight`, and
runs one weighted-BC pass through the data. The per-step CSV gains an
`awr_weight_mean` column; the function returns an epoch-level summary
including an optional `win_rate_proxy` (top-1 play-head accuracy on a
held-out subset).
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.awr.weights import awr_weights
from tichu_training.bc.dataset import BCExample
from tichu_training.bc.heads import BCModel, HEAD_LOGIT_DIMS
from tichu_training.bc.loss import masked_cross_entropy


_CSV_FIELDS = (
    ["step", "loss_total"]
    + [f"loss_{h}" for h in HEAD_LOGIT_DIMS]
    + [f"acc_{h}" for h in HEAD_LOGIT_DIMS]
    + ["awr_weight_mean"]
)


def awr_refine_epoch(
    model: BCModel,
    examples: Sequence[BCExample],
    baseline: ValueBaseline,
    optimizer: torch.optim.Optimizer,
    *,
    beta: float,
    max_weight: float,
    batch_size: int,
    log_path: Path,
    held_out_subset: Sequence[BCExample] | None = None,
    head_weights: dict[str, float] | None = None,
    features: np.ndarray | None = None,
) -> dict[str, float | None]:
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")
    head_weights = head_weights or {h: 1.0 for h in HEAD_LOGIT_DIMS}
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    weighted_examples = _apply_awr_weights(
        examples, baseline, beta=beta, max_weight=max_weight, features=features,
    )
    awr_weight_for = {id(orig): float(new.sample_weight / max(orig.sample_weight, 1e-12))
                      for orig, new in zip(examples, weighted_examples)}

    by_head: dict[str, list[BCExample]] = defaultdict(list)
    for e in weighted_examples:
        by_head[e.decision_type].append(e)

    last_total = float("inf")
    step = _next_step(log_path)
    weight_running_total = 0.0
    weight_running_count = 0
    model_device = next(model.parameters()).device

    for head, ex_list in by_head.items():
        for i in range(0, len(ex_list), batch_size):
            batch = ex_list[i : i + batch_size]
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
            # Append immediately so a crash mid-epoch leaves valid partial
            # progress on disk and live monitors can tail the file.
            _append_csv(log_path, [row])
            last_total = row["loss_total"]
            step += 1

    summary: dict[str, float | None] = {
        "loss_total": last_total,
        "avg_weight": (weight_running_total / weight_running_count) if weight_running_count else 0.0,
        "win_rate_proxy": None,
    }
    if held_out_subset is not None:
        summary["win_rate_proxy"] = _play_head_top1(model, held_out_subset)
    return summary


def _apply_awr_weights(
    examples: Sequence[BCExample],
    baseline: ValueBaseline,
    *,
    beta: float,
    max_weight: float,
    features: np.ndarray | None = None,
    chunk_size: int = 16_384,
) -> list[BCExample]:
    """Compute per-example AWR weights and rewrap with new sample_weight.

    `features` may be precomputed (e.g. the caller already stacked them
    to fit the value baseline) — avoids a second np.stack pass over a
    multi-GB dataset. When None, stacks from `examples`.

    `chunk_size` bounds the baseline forward pass — predictions are
    concatenated after the loop, so the downstream AWR math (max-subtract,
    exp, clip) still sees the full advantage vector and is numerically
    identical to the unchunked version. This is a memory-only knob; raise
    it on machines with more GPU RAM.
    """
    if not examples:
        return []
    if features is None:
        features = np.stack([e.features for e in examples])
    outcomes = np.array([e.round_outcome for e in examples], dtype=np.float32)

    feats_t = torch.from_numpy(features)
    pred_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(feats_t), chunk_size):
            chunk = feats_t[start : start + chunk_size]
            pred_chunks.append(baseline(chunk).cpu().numpy())
    preds = np.concatenate(pred_chunks) if pred_chunks else np.array([], dtype=np.float32)

    advantages = outcomes - preds
    w = awr_weights(advantages, beta=beta, max_weight=max_weight)
    out: list[BCExample] = []
    for e, weight in zip(examples, w):
        out.append(BCExample(
            decision_type=e.decision_type,
            features=e.features,
            target=e.target,
            legal_mask=e.legal_mask,
            sample_weight=float(e.sample_weight * float(weight)),
            skill_decile=e.skill_decile,
            round_outcome=e.round_outcome,
        ))
    return out


def _to_tensors(batch: Sequence[BCExample]) -> dict[str, torch.Tensor]:
    return {
        "features": torch.from_numpy(np.stack([e.features for e in batch])),
        "target": torch.tensor([e.target for e in batch], dtype=torch.long),
        "legal_mask": torch.from_numpy(np.stack([e.legal_mask for e in batch])),
        "sample_weight": torch.tensor(
            [e.sample_weight for e in batch], dtype=torch.float32
        ),
        "skill_decile": torch.tensor(
            [e.skill_decile for e in batch], dtype=torch.long
        ),
    }


def _play_head_top1(model: BCModel, examples: Iterable[BCExample]) -> float:
    """Top-1 accuracy of the play head as the per-epoch win-rate proxy."""
    play = [e for e in examples if e.decision_type == "play"]
    if not play:
        return 0.0
    tensors = _to_tensors(play)
    model_device = next(model.parameters()).device
    if model_device.type != "cpu":
        tensors = {k: v.to(model_device, non_blocking=True) for k, v in tensors.items()}
    with torch.no_grad():
        logits = model(tensors["features"], tensors["skill_decile"])["play"]
        pred = logits.masked_fill(~tensors["legal_mask"], float("-inf")).argmax(dim=-1)
        return float((pred == tensors["target"]).float().mean().item())


def _next_step(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    with log_path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh) - 1


def _append_csv(log_path: Path, rows: Iterable[dict[str, float]]) -> None:
    new = not log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_CSV_FIELDS))
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
