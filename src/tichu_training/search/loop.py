"""Generation loop for the search+learning experiment (ADR-0031).

The self-play collection step: run one full Round under four SelfPlaySearchAgents
(`play_full_round`, so the full call/schupfen/play stack and the call-bonus-inclusive
outcome are honoured — ADR-0031 Decision A), then turn every recorded `(features, π)`
target into a training `Sample` tagged with that seat's team-relative `round_outcome` z.
"""

from typing import NamedTuple

import numpy as np
import torch

from tichu_eval.play_full import play_full_round
from tichu_training.bc.loss import masked_kl_divergence
from tichu_training.search.selfplay import play_target_vectors


class Sample(NamedTuple):
    """One play-decision training row for the search+learning loop."""

    features: np.ndarray      # (FEATURIZER_OUTPUT_DIM,) float32
    target_probs: np.ndarray  # (ACTION_SPACE_SIZE,) float32 — the visit-distribution π
    legal_mask: np.ndarray    # (ACTION_SPACE_SIZE,) bool
    z: float                  # team-relative round_outcome (/100, call bonus included)


def _team_relative(total, seat: int) -> float:
    t = seat % 2
    return (total[t] - total[1 - t]) / 100.0


def samples_from_records(agents, total) -> list[Sample]:
    """Pair each seat's recorded `(features, π)` with that seat's team-relative outcome,
    vectorising π into the play-head target. Pure (no engine) — the z-attribution and
    target-shaping step, separated from the round run for direct testing."""
    out: list[Sample] = []
    for seat, agent in enumerate(agents):
        z = _team_relative(total, seat)
        for features, pi in agent.records:
            probs, mask = play_target_vectors(pi)
            out.append(Sample(features, probs, mask, z))
    return out


def collect_round(agents, position) -> list[Sample]:
    """Run one full Round under the four search agents and return its training Samples.
    Each agent's record buffer is cleared first, so agents may be reused across rounds."""
    for agent in agents:
        agent.records.clear()
    result = play_full_round(agents, position.state, position.grand_prefixes)
    return samples_from_records(agents, result.total)


def train_play_head(
    model,
    samples: list[Sample],
    *,
    skill_decile: int = 9,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 1024,
    seed: int = 0,
) -> list[float]:
    """Train the play head toward the visit-distribution targets via masked KL
    (ADR-0031 Decisions C, H). Backprop runs through the whole `BCModel`, but the loss
    touches only the ``play`` head, so the shared trunk + play head move while the
    wish/dragon heads (no gradient) are left as-is — option (ii). Conditions on
    ``skill_decile`` (the tier the self-play was generated at). Returns the per-epoch
    mean KL so the caller can see the loss come down."""
    feats = torch.from_numpy(np.stack([s.features for s in samples]))
    targets = torch.from_numpy(np.stack([s.target_probs for s in samples]))
    masks = torch.from_numpy(np.stack([s.legal_mask for s in samples]))
    skill = torch.full((len(samples),), int(skill_decile), dtype=torch.long)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    n = len(samples)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=gen)
        epoch_loss, seen = 0.0, 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            logits = model(feats[idx], skill[idx])["play"]
            loss = masked_kl_divergence(
                logits, targets[idx], masks[idx], torch.ones(len(idx))
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach()) * len(idx)
            seen += len(idx)
        history.append(epoch_loss / seen)
    return history
