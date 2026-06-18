"""Spawn-safety of the cross-version tournament builder.

`run_full_tournament(workers>1)` rebuilds each agent inside a process-pool worker,
so the builder must PICKLE. Passing the featurizer module directly fails
(`cannot pickle 'module' object`); the harness instead carries a string tag and
imports the module inside the worker. This test pins that: the bound builder
pickles, and the round-tripped builder produces a working v5-featurizer agent.
"""

import pickle
from functools import partial

import torch

from tichu_engine.legality import legal_actions_for
from tichu_engine.state import deal_initial_state
from tichu_export.torchscript import export_torchscript
from tichu_training.action_space import ACTION_SPACE_SIZE, ACTION_SPACE_VERSION
from tichu_training.cli.cross_version_tournament import build_versioned_ml_agent


class _DummyPolicy(torch.nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(feature_dim, ACTION_SPACE_SIZE)
        self.pass_card_head = torch.nn.Linear(feature_dim, 3)
        self.wish_head = torch.nn.Linear(feature_dim, 14)
        self.dragon_head = torch.nn.Linear(feature_dim, 2)

    def forward(self, features, skill_decile):
        return {"play": self.fc(features), "schupfen": self.pass_card_head(features),
                "wish": self.wish_head(features), "dragon_assignment": self.dragon_head(features)}


def _export_v5_dummy(tmp_path):
    torch.manual_seed(0)
    out = tmp_path / "policy.pt"
    export_torchscript(
        _DummyPolicy(224),
        example_inputs=(torch.randn(1, 224), torch.tensor([0], dtype=torch.long)),
        featurizer_version="v5", action_space_version=ACTION_SPACE_VERSION,
        output_path=out,
    )
    return out


def test_v5_builder_pickles_and_round_trips_to_working_agent(tmp_path):
    artifact = _export_v5_dummy(tmp_path)
    builder = partial(build_versioned_ml_agent, "v5_frozen",
                      skill_decile=9, checkpoint_path=str(artifact))

    # The whole point: this must NOT raise "cannot pickle 'module' object".
    revived = pickle.loads(pickle.dumps(builder))

    agent = revived()
    assert agent._fz.FEATURIZER_VERSION == "v5"
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    assert agent.act(ps) in legal_actions_for(ps)


def test_v6_builder_selects_live_featurizer():
    builder = partial(build_versioned_ml_agent, "v6", skill_decile=9, checkpoint_path="unused")
    revived = pickle.loads(pickle.dumps(builder))  # pickles before it ever touches disk
    assert revived.func is build_versioned_ml_agent
    assert revived.args == ("v6",)
