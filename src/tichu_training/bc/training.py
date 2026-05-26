"""BC training loop, CSV logging, and version-pinned checkpointing.

The trainer consumes an `Iterable[BCExample]` (per ADR-0011), so it works
with both:
  - the in-memory `SyntheticBCDataset` (a list-like iterable for smoke),
  - the streaming `ParquetBCDataset` (archive-driven replay-on-the-fly).

Batches are formed by per-head fill-then-emit buffers: each example is
routed into a buffer keyed by `decision_type`, and a step fires the
moment a buffer reaches `batch_size`. At the end of the iterable, any
partial buffers are flushed as final batches.
"""

import csv
import io
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from tqdm import tqdm

from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.dataset import BCExample
from tichu_training.bc.heads import BCModel, HEAD_LOGIT_DIMS
from tichu_training.bc.loss import masked_cross_entropy
from tichu_training.checkpoint import Checkpoint
from tichu_training.featurizer import FEATURIZER_VERSION


_CSV_FIELDS = (
    ["step", "loss_total"]
    + [f"loss_{h}" for h in HEAD_LOGIT_DIMS]
    + [f"acc_{h}" for h in HEAD_LOGIT_DIMS]
)


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


def train_one_epoch(
    model: BCModel,
    examples: Iterable[BCExample],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    log_path: Path,
    head_weights: dict[str, float] | None = None,
    show_progress: bool = True,
) -> float:
    """Train for one epoch over `examples`. Returns the final total loss.

    Accepts any `Iterable[BCExample]` — a list, a generator, or a streaming
    dataset. Examples are routed into per-head buffers; a buffer fires a
    training step the moment it fills to `batch_size`. Partial buffers at
    the end of the iterable fire one final step each.

    A tqdm progress bar is shown by default. When `examples` exposes
    `n_rows` (the case for ParquetBCDataset), the bar is bounded; otherwise
    it counts up. The postfix shows the last loss and per-head batch
    counts so you can see all three heads making progress.
    """
    head_weights = head_weights or {h: 1.0 for h in HEAD_LOGIT_DIMS}
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    head_buffers: dict[str, list[BCExample]] = defaultdict(list)
    head_batch_counts: dict[str, int] = {h: 0 for h in HEAD_LOGIT_DIMS}
    last_total = float("inf")
    step = _next_step(log_path)

    # Snapshot the model's device once. Tensors built on the CPU side from
    # numpy arrays must be moved here before the forward pass when the
    # model is on CUDA.
    model_device = next(model.parameters()).device

    def _fire(head: str, batch: Sequence[BCExample]) -> None:
        nonlocal last_total, step
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
        row = {f: 0.0 for f in _CSV_FIELDS}
        row["step"] = step
        row["loss_total"] = float(total.detach())
        row[f"loss_{head}"] = float(per_head_loss.detach())
        row[f"acc_{head}"] = acc
        # Append immediately so a crash mid-epoch leaves valid partial progress
        # on disk and live monitors can tail the file during long runs.
        _append_csv(log_path, [row])
        last_total = row["loss_total"]
        head_batch_counts[head] = head_batch_counts.get(head, 0) + 1
        step += 1

    total_known = getattr(examples, "n_rows", None) if show_progress else None
    bar = tqdm(
        total=total_known,
        unit="ex",
        dynamic_ncols=True,
        desc="train",
        disable=not show_progress,
    ) if show_progress else None

    try:
        try:
            for example in examples:
                buf = head_buffers[example.decision_type]
                buf.append(example)
                if len(buf) >= batch_size:
                    _fire(example.decision_type, buf)
                    buf.clear()
                    if bar is not None:
                        bar.set_postfix(
                            loss=f"{last_total:.3f}",
                            **{f"b_{h}": head_batch_counts[h] for h in HEAD_LOGIT_DIMS},
                            refresh=False,
                        )
                if bar is not None:
                    bar.update(1)
        finally:
            # Drain partial buffers — runs on normal completion AND on
            # KeyboardInterrupt / dataset exception so already-pulled
            # examples still produce a row. _fire writes per-batch, so
            # rows fired before any crash are already on disk.
            for head, buf in head_buffers.items():
                if buf:
                    _fire(head, buf)
                    buf.clear()
        if bar is not None:
            bar.set_postfix(
                loss=f"{last_total:.3f}",
                **{f"b_{h}": head_batch_counts[h] for h in HEAD_LOGIT_DIMS},
            )
    finally:
        if bar is not None:
            bar.close()

    return last_total


def _next_step(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    with log_path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh) - 1  # subtract header


def _append_csv(log_path: Path, rows: Iterable[dict[str, float]]) -> None:
    new = not log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_CSV_FIELDS))
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str | Path,
) -> None:
    buf = io.BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        buf,
    )
    Checkpoint(
        featurizer_version=FEATURIZER_VERSION,
        action_space_version=ACTION_SPACE_VERSION,
        payload=buf.getvalue(),
    ).save(path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    cp = Checkpoint.load(
        path,
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    state = torch.load(io.BytesIO(cp.payload), weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return int(state["step"])
