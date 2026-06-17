"""Critic-bootstrap leaf evaluator (ADR-0030 Phase B).

`CriticValue` wraps a frozen `ValueBaseline` behind the
``leaf_fn(state, root, rng) -> float`` interface `EngineWorld` expects, returning
``V(featurize(state.private_view(root)))`` — one forward instead of a full rollout
(the prototype showed rollouts dominate cost). The Value Baseline is fit on master
self-play `round_outcome` (team-relative, normalized /100) by
`scripts/fit_search_critic.py`; this module is the inference + persistence side.
"""

import pathlib

import torch

from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.featurizer import FEATURIZER_VERSION, featurize


class CriticValue:
    """Frozen-critic leaf: ``leaf_fn(state, root, rng) -> float``."""

    def __init__(self, baseline: ValueBaseline) -> None:
        self._baseline = baseline.eval()

    def __call__(self, state, root: int, rng) -> float:
        feats = torch.from_numpy(featurize(state.private_view(root))).unsqueeze(0)
        with torch.no_grad():
            return float(self._baseline(feats).reshape(-1)[0].item())


def save_critic(
    baseline: ValueBaseline,
    path,
    *,
    feature_dim: int,
    hidden: int,
    note: str = "",
) -> None:
    """Persist a trained Value Baseline with the metadata `load_critic` needs."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": baseline.state_dict(),
            "feature_dim": int(feature_dim),
            "hidden": int(hidden),
            # Derived from the model so a deeper critic reloads at the right depth.
            "depth": len(baseline.blocks) + 1,
            "featurizer_version": FEATURIZER_VERSION,
            "note": note,
        },
        path,
    )


def load_critic(path) -> CriticValue:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("featurizer_version") != FEATURIZER_VERSION:
        raise ValueError(
            f"critic featurizer_version {blob.get('featurizer_version')!r} "
            f"!= harness {FEATURIZER_VERSION!r}"
        )
    baseline = ValueBaseline(blob["feature_dim"], hidden=blob["hidden"], depth=blob.get("depth", 1))
    baseline.load_state_dict(blob["state_dict"])
    return CriticValue(baseline)
