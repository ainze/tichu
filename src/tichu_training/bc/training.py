"""BC training loop, CSV logging, and version-pinned checkpointing."""

import csv
import io
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import torch

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


def _batch_by_head(examples: Sequence[BCExample]) -> dict[str, list[BCExample]]:
    by_head: dict[str, list[BCExample]] = defaultdict(list)
    for e in examples:
        by_head[e.decision_type].append(e)
    return by_head


def _to_tensors(batch: Sequence[BCExample]) -> dict[str, torch.Tensor]:
    return {
        "features": torch.from_numpy(
            __import__("numpy").stack([e.features for e in batch])
        ),
        "target": torch.tensor([e.target for e in batch], dtype=torch.long),
        "legal_mask": torch.from_numpy(
            __import__("numpy").stack([e.legal_mask for e in batch])
        ),
        "sample_weight": torch.tensor(
            [e.sample_weight for e in batch], dtype=torch.float32
        ),
        "skill_decile": torch.tensor(
            [e.skill_decile for e in batch], dtype=torch.long
        ),
    }


def train_one_epoch(
    model: BCModel,
    examples: Sequence[BCExample],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    log_path: Path,
    head_weights: dict[str, float] | None = None,
) -> float:
    """Train for one epoch over `examples`. Returns the final total loss."""
    head_weights = head_weights or {h: 1.0 for h in HEAD_LOGIT_DIMS}
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    by_head = _batch_by_head(examples)
    # Build batches per head; flatten into a (decision_type, batch) work list.
    work: list[tuple[str, Sequence[BCExample]]] = []
    for head, ex_list in by_head.items():
        for i in range(0, len(ex_list), batch_size):
            work.append((head, ex_list[i : i + batch_size]))

    rows: list[dict[str, float]] = []
    last_total = float("inf")
    step = _next_step(log_path)
    for head, batch in work:
        tensors = _to_tensors(batch)
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
        rows.append(row)
        last_total = row["loss_total"]
        step += 1

    _append_csv(log_path, rows)
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
