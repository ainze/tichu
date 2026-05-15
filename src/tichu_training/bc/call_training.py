"""Synthetic dataset + training loop for the Tichu/Grand Tichu call networks.

Negative examples for the parquet path are not directly present in the
shards from #007 (the parser emits only positive call events). A future
parquet adapter will synthesise negatives per non-calling player per call
opportunity. The smoke test trains on `SyntheticCallDataset` which produces
both classes with a configurable balance.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tichu_training.bc.call_model import CallNetwork


@dataclass
class CallExample:
    features: np.ndarray
    target: int  # 0 = no-call, 1 = call
    skill_decile: int
    sample_weight: float = 1.0


class SyntheticCallDataset(Iterable[CallExample]):
    """Deterministic call examples; mixes positives at the requested rate."""

    def __init__(
        self,
        *,
        seed: int = 0,
        n_examples: int = 200,
        positive_rate: float = 0.3,
        feature_dim: int | None = None,
        skill_buckets: int = 10,
    ) -> None:
        if feature_dim is None:
            from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
            feature_dim = FEATURIZER_OUTPUT_DIM
        self.seed = seed
        self.n_examples = n_examples
        self.positive_rate = positive_rate
        self.feature_dim = feature_dim
        self.skill_buckets = skill_buckets

    def __iter__(self) -> Iterator[CallExample]:
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_examples):
            target = int(rng.random() < self.positive_rate)
            features = rng.standard_normal(self.feature_dim).astype(np.float32)
            # 10% cold-start; otherwise rated.
            if rng.random() < 0.1:
                skill = self.skill_buckets
            else:
                skill = int(rng.integers(0, self.skill_buckets))
            yield CallExample(features=features, target=target, skill_decile=skill)


def _stack(examples: Sequence[CallExample]) -> dict[str, torch.Tensor]:
    return {
        "features": torch.from_numpy(np.stack([e.features for e in examples])),
        "target": torch.tensor([e.target for e in examples], dtype=torch.long),
        "skill_decile": torch.tensor([e.skill_decile for e in examples], dtype=torch.long),
        "sample_weight": torch.tensor([e.sample_weight for e in examples], dtype=torch.float32),
    }


def train_one_call_epoch(
    network: CallNetwork,
    examples: Sequence[CallExample],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    log_path: Path,
) -> float:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists()
    step = _next_step(log_path)
    last_loss = float("inf")
    rows: list[dict[str, float]] = []
    for i in range(0, len(examples), batch_size):
        batch = list(examples[i : i + batch_size])
        if not batch:
            continue
        tensors = _stack(batch)
        logits = network(tensors["features"], tensors["skill_decile"])
        per_sample = F.cross_entropy(logits, tensors["target"], reduction="none")
        loss = (per_sample * tensors["sample_weight"]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = (pred == tensors["target"]).float().mean().item()
        rows.append({"step": step, "loss": float(loss.detach()), "accuracy": acc})
        last_loss = float(loss.detach())
        step += 1

    with log_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["step", "loss", "accuracy"])
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return last_loss


def _next_step(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    with log_path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh) - 1


def calling_rate_by_decile(
    network: CallNetwork, examples: Sequence[CallExample]
) -> dict[int, float]:
    """Predicted P(call) > 0.5 → 1, grouped by skill_decile."""
    if not examples:
        return {}
    tensors = _stack(examples)
    with torch.no_grad():
        probs = torch.softmax(network(tensors["features"], tensors["skill_decile"]), dim=-1)
        call_prob = probs[:, 1]
        calls = (call_prob > 0.5).long()
    by_decile: dict[int, list[int]] = defaultdict(list)
    deciles = tensors["skill_decile"].tolist()
    for d, c in zip(deciles, calls.tolist()):
        by_decile[int(d)].append(int(c))
    return {d: sum(v) / len(v) for d, v in by_decile.items()}


def write_calling_rate_csv(
    network: CallNetwork, examples: Sequence[CallExample], path: Path
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rates = calling_rate_by_decile(network, examples)
    by_decile_n: dict[int, int] = defaultdict(int)
    for e in examples:
        by_decile_n[e.skill_decile] += 1
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["decile", "n", "calling_rate"])
        writer.writeheader()
        for decile in sorted(rates.keys()):
            writer.writerow({
                "decile": decile,
                "n": by_decile_n[decile],
                "calling_rate": rates[decile],
            })
