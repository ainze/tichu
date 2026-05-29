"""MLAgent — exported TS model + version assertion + fallback."""

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from tichu_engine.legality import legal_actions_for
from tichu_engine.state import deal_initial_state

from tichu_export.torchscript import export_torchscript
from tichu_inference.ml_agent import MLAgent, VersionMismatchError as MLVersionMismatch


# Track the live module constants so the fixture stays current across
# featurizer / action-space version bumps. Hard-coding "v1" here used
# to silently work because MLAgent's expected-version pin also tracked
# the live constants — both halves moved together. Now we keep the
# fixture explicit about that contract.
from tichu_training.action_space import ACTION_SPACE_VERSION as _ACTION_SPACE_VERSION
from tichu_training.featurizer import FEATURIZER_VERSION as _FEATURIZER_VERSION


class _DummyPolicy(torch.nn.Module):
    """Bare-minimum BCModel-shaped module — uses real featurizer output dim."""

    def __init__(self, feature_dim: int, action_space_size: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(feature_dim, action_space_size)
        self.pass_card_head = torch.nn.Linear(feature_dim, 3)
        self.wish_head = torch.nn.Linear(feature_dim, 14)
        self.dragon_head = torch.nn.Linear(feature_dim, 2)

    def forward(self, features, skill_decile):
        return {
            "play": self.fc(features),
            "schupfen": self.pass_card_head(features),
            "wish": self.wish_head(features),
            "dragon_assignment": self.dragon_head(features),
        }


def _export_dummy(tmp_path: Path, *, feature_dim: int, action_space_size: int,
                  featurizer_version: str = _FEATURIZER_VERSION,
                  action_space_version: str = _ACTION_SPACE_VERSION) -> Path:
    torch.manual_seed(0)
    model = _DummyPolicy(feature_dim, action_space_size)
    out = tmp_path / "policy.pt"
    export_torchscript(
        model,
        example_inputs=(torch.randn(1, feature_dim), torch.tensor([0], dtype=torch.long)),
        featurizer_version=featurizer_version,
        action_space_version=action_space_version,
        output_path=out,
    )
    return out


def test_act_returns_legal_action_on_deal_initial(tmp_path):
    from tichu_training.action_space import ACTION_SPACE_SIZE
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    artifact = _export_dummy(tmp_path,
                             feature_dim=FEATURIZER_OUTPUT_DIM,
                             action_space_size=ACTION_SPACE_SIZE)
    agent = MLAgent(artifact)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    action = agent.act(ps)
    assert action in legal_actions_for(ps)


def test_version_mismatch_raises_at_construction(tmp_path):
    from tichu_training.action_space import ACTION_SPACE_SIZE
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    bad = _export_dummy(tmp_path,
                        feature_dim=FEATURIZER_OUTPUT_DIM,
                        action_space_size=ACTION_SPACE_SIZE,
                        featurizer_version="v999")
    with pytest.raises(MLVersionMismatch):
        MLAgent(bad)


def test_fallback_fires_on_model_exception(tmp_path, caplog):
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    # Broken artifact — wrong output dim breaks the action-space lookup.
    artifact = _export_dummy(tmp_path, feature_dim=FEATURIZER_OUTPUT_DIM, action_space_size=42)
    # Bypass version check to load it: stamp matching versions then deliberately
    # poke the agent's internal module to raise.
    agent = MLAgent(artifact)
    # Monkey-patch the loaded module to always raise.
    def boom(*_a, **_k):
        raise RuntimeError("synthetic failure")
    agent._module = type("Broken", (), {"__call__": boom})()
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    with caplog.at_level(logging.ERROR, logger="tichu_inference.ml_agent"):
        action = agent.act(ps)
    assert action in legal_actions_for(ps)
    assert agent.last_fallback_used is True
    assert any("synthetic failure" in r.getMessage() or "fallback" in r.getMessage().lower()
               for r in caplog.records)


def test_fallback_fires_on_nan_logits(tmp_path):
    from tichu_training.action_space import ACTION_SPACE_SIZE
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    artifact = _export_dummy(tmp_path,
                             feature_dim=FEATURIZER_OUTPUT_DIM,
                             action_space_size=ACTION_SPACE_SIZE)
    agent = MLAgent(artifact)

    nan_size = ACTION_SPACE_SIZE
    class NanModel:
        def __call__(self, features, skill_decile):
            n = features.shape[0]
            return {
                "play": torch.full((n, nan_size), float("nan")),
                "schupfen": torch.zeros(n, 3),
                "wish": torch.zeros(n, 14),
                "dragon_assignment": torch.zeros(n, 2),
            }
    agent._module = NanModel()

    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    action = agent.act(ps)
    assert action in legal_actions_for(ps)
    assert agent.last_fallback_used is True


def test_rank_actions_returns_legal_actions(tmp_path):
    from tichu_training.action_space import ACTION_SPACE_SIZE
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    artifact = _export_dummy(tmp_path,
                             feature_dim=FEATURIZER_OUTPUT_DIM,
                             action_space_size=ACTION_SPACE_SIZE)
    agent = MLAgent(artifact)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    ranked = agent.rank_actions(ps)
    assert ranked is not None and len(ranked) > 0
    legal = legal_actions_for(ps)
    for a in ranked:
        assert a in legal
