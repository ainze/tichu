"""Tests for SelfPlaySearchAgent (ADR-0031 Phase 1).

The self-play data-generation agent: on a Play Decision it runs PIMC with root
Dirichlet noise (Decision G) and τ=1 opponent sampling (Decision F), records the
``(features, π)`` policy-improvement target (Decision C), and advances the game by
sampling the action from the visit distribution. Pending wish/dragon/schupfen and
the Tichu/Grand calls delegate to the wrapped policy. Tested torch-free with the
real engine + a deterministic stub policy and the rollout leaf.
"""

from dataclasses import replace

from tichu_engine.legality import legal_actions_for
from tichu_engine.state import MahjongWishPending, PrivateState, deal_initial_state
from tichu_training.action_space import ACTION_SPACE_SIZE
from tichu_training.featurizer import (
    FEATURIZER_OUTPUT_DIM,
    _combination_to_action_index,
)
from tichu_training.search.engine_world import fast_rollout_leaf
from tichu_training.search.selfplay import SelfPlaySearchAgent, play_target_vectors


def test_play_target_vectors_place_mass_at_action_indices():
    # π over legal ConcreteActions must vectorize into a (ACTION_SPACE_SIZE,) target +
    # legal mask: mass lands at each action's intent index, the result is renormalised
    # over the representable (mapped) legal actions, and mass only ever sits on legal slots.
    state = deal_initial_state(seed=5)
    pv = state.private_view(state.public.current_player)
    legal = list(legal_actions_for(pv))
    pi = {a: 1.0 / len(legal) for a in legal}

    probs, mask = play_target_vectors(pi)

    assert probs.shape == (ACTION_SPACE_SIZE,)
    assert mask.shape == (ACTION_SPACE_SIZE,) and mask.dtype == bool
    assert abs(float(probs.sum()) - 1.0) < 1e-6
    assert bool(((probs > 0) <= mask).all())  # mass only on legal slots
    for a in legal:
        idx = _combination_to_action_index(a)
        if idx is not None:
            assert probs[idx] > 0 and mask[idx]


class StubPolicy:
    def play_action_scores(self, pv):
        legal = list(legal_actions_for(pv))
        return [(a, 1.0 / len(legal)) for a in legal]

    def act(self, pv):
        return min(legal_actions_for(pv), key=repr)

    def should_call(self, pv, kind):
        return False


def _agent(**kw):
    return SelfPlaySearchAgent.from_policy(
        StubPolicy(), worlds=2, sims=10, leaf_fn=fast_rollout_leaf,
        root_alpha=1.0, root_eps=0.25, temperature=1.0, seed=0, **kw
    )


def test_act_runs_search_records_target_and_returns_legal_action():
    state = deal_initial_state(seed=5)
    root = state.public.current_player
    pv = state.private_view(root)

    agent = _agent()
    action = agent.act(pv)

    legal = set(legal_actions_for(pv))
    assert action in legal
    assert len(agent.records) == 1
    feats, pi = agent.records[0]
    assert feats.shape == (FEATURIZER_OUTPUT_DIM,)
    assert abs(sum(pi.values()) - 1.0) < 1e-6
    assert set(pi).issubset(legal)


def test_pending_decision_delegates_and_is_not_recorded():
    state = deal_initial_state(seed=5)
    root = state.public.current_player
    pub = replace(state.public, pending_decision=MahjongWishPending(player=root))
    pv = PrivateState(player=root, hand=state.hands[root], public=pub)

    agent = _agent()
    agent.act(pv)
    assert agent.records == []
