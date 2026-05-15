"""Belief training loop + reliability-table calibration."""

import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from tichu_training.belief.dataset import BeliefExample
from tichu_training.belief.model import BeliefModel, belief_accuracy, belief_loss


_NUM_BUCKETS = 10
_STEP_FIELDS = ("step", "loss", "accuracy")
_CALIB_FIELDS = (
    "bucket_index", "p_low", "p_high", "predicted_mean", "empirical_freq", "n"
)


def _stack(batch: Sequence[BeliefExample]) -> dict[str, torch.Tensor]:
    return {
        "features": torch.from_numpy(np.stack([e.features for e in batch])),
        "labels": torch.from_numpy(np.stack([e.labels for e in batch])),
        "mask": torch.from_numpy(np.stack([e.mask for e in batch])),
    }


def train_one_belief_epoch(
    model: BeliefModel,
    examples: Sequence[BeliefExample],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    log_path: Path,
) -> float:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = not log_path.exists()
    step = _next_step(log_path)
    last_loss = float("inf")
    rows: list[dict] = []
    for i in range(0, len(examples), batch_size):
        batch = list(examples[i : i + batch_size])
        if not batch:
            continue
        tensors = _stack(batch)
        logits = model(tensors["features"])
        loss = belief_loss(logits, tensors["labels"], tensors["mask"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = belief_accuracy(logits, tensors["labels"], tensors["mask"])
        rows.append({"step": step, "loss": float(loss.detach()), "accuracy": acc})
        last_loss = float(loss.detach())
        step += 1

    with log_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_STEP_FIELDS))
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return last_loss


def calibration_table(
    model: BeliefModel, examples: Sequence[BeliefExample]
) -> list[dict]:
    if not examples:
        return _empty_table()
    tensors = _stack(examples)
    with torch.no_grad():
        probs = torch.sigmoid(model(tensors["features"])).cpu().numpy()
    labels = tensors["labels"].cpu().numpy()
    mask = tensors["mask"].cpu().numpy()

    flat_p = probs[mask]
    flat_y = labels[mask]
    table: list[dict] = []
    edges = np.linspace(0.0, 1.0, _NUM_BUCKETS + 1)
    for b in range(_NUM_BUCKETS):
        lo, hi = float(edges[b]), float(edges[b + 1])
        if b == _NUM_BUCKETS - 1:
            in_bucket = (flat_p >= lo) & (flat_p <= hi)
        else:
            in_bucket = (flat_p >= lo) & (flat_p < hi)
        n = int(in_bucket.sum())
        predicted_mean = float(flat_p[in_bucket].mean()) if n else 0.0
        empirical_freq = float(flat_y[in_bucket].mean()) if n else 0.0
        table.append({
            "bucket_index": b,
            "p_low": lo,
            "p_high": hi,
            "predicted_mean": predicted_mean,
            "empirical_freq": empirical_freq,
            "n": n,
        })
    return table


def write_calibration_csv(
    model: BeliefModel, examples: Sequence[BeliefExample], path: Path
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = calibration_table(model, examples)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_CALIB_FIELDS))
        writer.writeheader()
        for row in table:
            writer.writerow(row)


def _empty_table() -> list[dict]:
    edges = np.linspace(0.0, 1.0, _NUM_BUCKETS + 1)
    return [
        {
            "bucket_index": b,
            "p_low": float(edges[b]),
            "p_high": float(edges[b + 1]),
            "predicted_mean": 0.0,
            "empirical_freq": 0.0,
            "n": 0,
        }
        for b in range(_NUM_BUCKETS)
    ]


def _next_step(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    with log_path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh) - 1
