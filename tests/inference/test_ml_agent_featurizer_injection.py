"""MLAgent featurizer injection — run a v5-trained export inside a v6 process.

The cross-version tournament (cotrain_v6 vs cotrain_wish_v5) needs the SAME
process to drive a v6 net (591-dim features) and a v5 net (224-dim features). The
featurizer is therefore injected per agent: default = the live v6 module; pass
`featurizer_v5_frozen` to feed a v5-stamped export its 224-dim features and load it
past the version guard. These tests pin both halves of that contract.
"""

import pytest
import torch

from tichu_engine.legality import legal_actions_for
from tichu_engine.state import deal_initial_state
from tichu_export.torchscript import export_torchscript
from tichu_inference.ml_agent import MLAgent, VersionMismatchError
from tichu_training import featurizer_v5_frozen
from tichu_training.action_space import (
    ACTION_SPACE_SIZE,
    ACTION_SPACE_VERSION,
)


class _DummyPolicy(torch.nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(feature_dim, ACTION_SPACE_SIZE)
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


def _export(tmp_path, *, feature_dim: int, featurizer_version: str):
    torch.manual_seed(0)
    out = tmp_path / "policy.pt"
    export_torchscript(
        _DummyPolicy(feature_dim),
        example_inputs=(torch.randn(1, feature_dim), torch.tensor([0], dtype=torch.long)),
        featurizer_version=featurizer_version,
        action_space_version=ACTION_SPACE_VERSION,
        output_path=out,
    )
    return out


def _first_legal_act(agent):
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    action = agent.act(ps)
    assert action in legal_actions_for(ps)
    return action


def test_v5_frozen_agent_loads_and_acts(tmp_path):
    """A v5-stamped, 224-dim export loads and plays a real deal when built with the
    frozen v5 featurizer — the whole point of the injection."""
    artifact = _export(tmp_path, feature_dim=224, featurizer_version="v5")
    agent = MLAgent(artifact, featurizer=featurizer_v5_frozen)
    _first_legal_act(agent)


def test_injected_featurizer_drives_feature_dim(tmp_path):
    """The injected featurizer (not the process-global v6) decides the feature
    width fed to the net: 224 for the v5 agent."""
    artifact = _export(tmp_path, feature_dim=224, featurizer_version="v5")
    agent = MLAgent(artifact, featurizer=featurizer_v5_frozen)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    features, _skill = agent._inputs(ps)
    assert features.shape == (1, 224)


def test_default_agent_rejects_v5_export(tmp_path):
    """Without the injection, the default (v6) agent must still reject a v5 export —
    the version guard is intact, not bypassed."""
    artifact = _export(tmp_path, feature_dim=224, featurizer_version="v5")
    with pytest.raises(VersionMismatchError):
        MLAgent(artifact)


def test_default_v6_path_unchanged(tmp_path):
    """Default construction (no featurizer kwarg) still loads + plays a v6 export —
    the injection is additive, the existing path byte-compatible."""
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM, FEATURIZER_VERSION

    artifact = _export(tmp_path, feature_dim=FEATURIZER_OUTPUT_DIM,
                       featurizer_version=FEATURIZER_VERSION)
    agent = MLAgent(artifact)
    _first_legal_act(agent)
