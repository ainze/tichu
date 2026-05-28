"""Value baseline V(state) for AWR advantage estimation.

A small MLP fit by MSE on a per-row scalar target. The target is
selected by the caller — currently either `round_outcome` (team-0 minus
team-1 Ergebnis) or `game_won` cast to 0.0/1.0 — via the
`awr.value_target` config knob; see `tichu_training.awr.targets` for the
extractor and ADR-0013 / CONTEXT.md for the design.

Lives separately from `BCModel`: no trunk sharing in v1 — that is a
follow-up.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ValueBaseline(nn.Module):
    def __init__(self, feature_dim: int, *, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(features))
        return self.fc2(h).squeeze(-1)


def fit_value_baseline(
    baseline: ValueBaseline,
    features: np.ndarray,
    outcomes: np.ndarray,
    *,
    batch_size: int,
    epochs: int,
    lr: float,
    log_path: Path | None = None,
    show_progress: bool = True,
) -> float:
    """MSE fit on `(features, outcomes)`. Returns final MSE."""
    feats_t = torch.from_numpy(np.asarray(features, dtype=np.float32))
    outcomes_t = torch.from_numpy(np.asarray(outcomes, dtype=np.float32))
    optimizer = torch.optim.Adam(baseline.parameters(), lr=lr)
    n = feats_t.shape[0]
    final_mse = float("inf")
    bar = tqdm(
        total=epochs, unit="epoch", dynamic_ncols=True,
        desc="value baseline", disable=not show_progress,
    )
    try:
        for _ in range(epochs):
            for i in range(0, n, batch_size):
                batch_feats = feats_t[i : i + batch_size]
                batch_out = outcomes_t[i : i + batch_size]
                preds = baseline(batch_feats)
                loss = F.mse_loss(preds, batch_out)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                final_mse = float(loss.detach())
            bar.update(1)
            bar.set_postfix(last_mse=f"{final_mse:.4f}", refresh=False)
    finally:
        bar.close()
    # Re-evaluate MSE on the trained baseline in the same mini-batches
    # used during training. The full-dataset forward pass would OOM the
    # GPU on multi-million-row training sets.
    with torch.no_grad():
        sse = 0.0
        for i in range(0, n, batch_size):
            preds = baseline(feats_t[i : i + batch_size])
            sse += float(((preds - outcomes_t[i : i + batch_size]) ** 2).sum())
        final_mse = sse / max(1, n)
    return final_mse
