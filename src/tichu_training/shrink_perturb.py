"""Shrink-and-Perturb (Ash & Adams 2020): blend trained weights toward a fresh
initialisation, keeping most of what the net knows while restoring some of its
ability to keep learning.

    p <- alpha * p_trained + beta * p_fresh        (every floating Parameter)

`alpha=1, beta=0` is the identity. `alpha + beta = 1` interpolates toward a fresh
init (the BBF form). `beta` small with `alpha < 1` is the original
shrink-plus-noise form. Buffers are left alone: they are running statistics or
constants, not learned weights, and a fresh init's buffers say nothing about the
trained net's inputs.

This is the deciding experiment of the 2026-10-10 train-from-random options memo
(docs/notes/2026-10-10-train-from-random-options-memo.md). Training from random
only helps if the co-train plateau comes from WHERE training started (a basin, or
lost plasticity) rather than from the learning signal. Perturbing the champion and
re-training tests that claim at a fraction of a from-random run's cost.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

NET_TYPES = ("play", "schupfen", "tichu", "grand")


def shrink_perturb_(trained: nn.Module, fresh: nn.Module, *, alpha: float,
                    beta: float) -> dict[str, float]:
    """Blend `trained` toward `fresh` IN PLACE; return parameter-space stats.

    The two modules must be the same architecture (same parameter names and
    shapes). `fresh` is only read. The stats are the global L2 norms before and
    after, and `rel_change = ||after - before|| / ||before||` — a parameter-space
    measure only; how far the POLICY moved is read off the gate's first window.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if beta < 0.0:
        raise ValueError(f"beta must be >= 0, got {beta}")
    fresh_params = dict(fresh.named_parameters())
    trained_names = {n for n, _ in trained.named_parameters()}
    if trained_names != set(fresh_params):
        missing = sorted(trained_names ^ set(fresh_params))
        raise ValueError(f"architecture mismatch; differing parameters: {missing[:5]}")

    sq_before = sq_after = sq_delta = 0.0
    with torch.no_grad():
        for name, p in trained.named_parameters():
            if not p.is_floating_point():
                continue
            q = fresh_params[name]
            if q.shape != p.shape:
                raise ValueError(f"shape mismatch at {name}: {tuple(p.shape)} vs {tuple(q.shape)}")
            before = p.detach().clone()
            p.mul_(alpha).add_(q.to(p.device, p.dtype), alpha=beta)
            sq_before += float(before.pow(2).sum())
            sq_after += float(p.pow(2).sum())
            sq_delta += float((p - before).pow(2).sum())
    norm_before = sq_before ** 0.5
    return {
        "norm_before": norm_before,
        "norm_after": sq_after ** 0.5,
        "rel_change": (sq_delta ** 0.5) / norm_before if norm_before else 0.0,
    }


def build_perturbed(weights: Path, config: dict, *, alpha: float, beta: float,
                    seed: int, nets: list[str]) -> tuple[dict, dict]:
    """Load a rollout-weights stack and shrink-and-perturb `nets`.

    Returns (models, stats): all four co-train nets built from the config's arch
    blocks, with `nets` blended toward a fresh init seeded by `seed`, plus
    `shrink_perturb_`'s stats per perturbed net. Nets not in `nets` are the
    source's, bit for bit.
    """
    from tichu_training.cli.train_cotrain import _build_models

    unknown = set(nets) - set(NET_TYPES)
    if unknown:
        raise ValueError(f"unknown nets {sorted(unknown)}; choose from {NET_TYPES}")
    payload = torch.load(weights, weights_only=False, map_location="cpu")
    models = _build_models(config)
    for net in NET_TYPES:
        models[net].load_state_dict(payload["models"][net])
    # The fresh init is seeded and built AFTER loading, so it depends only on
    # `seed` and the arch blocks, never on the source weights.
    torch.manual_seed(seed)
    fresh = _build_models(config)
    stats = {net: shrink_perturb_(models[net], fresh[net], alpha=alpha, beta=beta)
             for net in nets}
    return models, stats
