"""Tests for the critic-bootstrap leaf evaluator (ADR-0030 Phase B).

CriticValue wraps a frozen ValueBaseline behind the leaf_fn(state, root, rng) ->
float interface EngineWorld expects, evaluating V(featurize(root's view)). Uses
torch (in-harness) with a tiny randomly-initialised ValueBaseline — no checkpoint.
"""

import random

import pytest
import torch

from tichu_engine.state import deal_initial_state
from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM, featurize
from tichu_training.search.critic import CriticValue, load_critic, save_critic


def _state_root():
    state = deal_initial_state(seed=0)
    return state, state.public.current_player


def test_critic_leaf_returns_a_float():
    critic = CriticValue(ValueBaseline(FEATURIZER_OUTPUT_DIM, hidden=8))
    state, root = _state_root()
    assert isinstance(critic(state, root, random.Random(0)), float)


def test_critic_value_matches_a_manual_forward():
    torch.manual_seed(0)
    baseline = ValueBaseline(FEATURIZER_OUTPUT_DIM, hidden=8)
    state, root = _state_root()
    feats = torch.from_numpy(featurize(state.private_view(root))).unsqueeze(0)
    with torch.no_grad():
        expected = float(baseline(feats).reshape(-1)[0].item())
    assert CriticValue(baseline)(state, root, random.Random(0)) == pytest.approx(expected)


def test_save_load_round_trip(tmp_path):
    torch.manual_seed(1)
    baseline = ValueBaseline(FEATURIZER_OUTPUT_DIM, hidden=8)
    path = tmp_path / "critic.bin"
    save_critic(baseline, path, feature_dim=FEATURIZER_OUTPUT_DIM, hidden=8)

    loaded = load_critic(path)
    state, root = _state_root()
    assert loaded(state, root, random.Random(0)) == pytest.approx(
        CriticValue(baseline)(state, root, random.Random(0))
    )
