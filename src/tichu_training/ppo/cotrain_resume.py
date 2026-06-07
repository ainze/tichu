"""Resume Bundle — whole-training-state checkpoint for co-training (ADR-0034).

Distinct from a serving-export Checkpoint (one net's weights for inference): the
Resume Bundle holds *everything* a multi-day co-training run needs to continue
losslessly — all four policy nets, the shared Perfect-Info Critic, optimizer
(Adam) state, the per-net KL coefficients, the iteration counter, the league, and
RNG state. Written atomically (tmp -> fsync -> rename) keeping the last 2, so a
Ctrl-C at any moment loses at most one iteration and never corrupts the bundle.
"""

import os
import random

import numpy as np
import torch

PREV_SUFFIX = ".prev"  # the one-before bundle kept for corruption recovery (keep-2)


def capture_rng() -> dict:
    """Snapshot the global RNG of torch / numpy / python so a resume is bit-identical."""
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def save_resume_bundle(
    path: str,
    *,
    models: dict,
    critic,
    optimizer,
    kl_coefs: dict,
    iteration: int,
    rng_state: dict,
    league=None,
) -> None:
    """Atomically write the whole training state to `path`, keeping the prior bundle
    at `path + PREV_SUFFIX`. Write order: serialize to a `.tmp`, fsync, move the
    current live bundle aside to `.prev`, then atomically rename `.tmp` into place —
    so an interrupt at any point leaves either the new or a recoverable old bundle."""
    payload = {
        "version": 1,
        "models": {dt: m.state_dict() for dt, m in models.items()},
        "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "kl_coefs": dict(kl_coefs),
        "iteration": int(iteration),
        "rng": rng_state,
        "league": league if league is not None else [],
    }
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    if os.path.exists(path):
        os.replace(path, path + PREV_SUFFIX)
    os.replace(tmp, path)


def load_resume_bundle(path: str, *, models: dict, critic, optimizer) -> dict:
    """Load a Resume Bundle: restore every policy net, the critic, and the optimizer
    in place, and return the payload so the caller can restore the KL coefficients,
    iteration counter, league, and RNG (`restore_rng(payload["rng"])`)."""
    payload = torch.load(path, weights_only=False)
    for dt, m in models.items():
        m.load_state_dict(payload["models"][dt])
    critic.load_state_dict(payload["critic"])
    optimizer.load_state_dict(payload["optimizer"])
    return payload
