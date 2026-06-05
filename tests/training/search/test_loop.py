"""Tests for the self-play collection step of the search+learning loop (ADR-0031).

`collect_round` runs one full Round under four SelfPlaySearchAgents and turns each
seat's recorded `(features, π)` targets into training `Sample`s carrying the
team-relative `round_outcome` z. `samples_from_records` is the pure z-attribution +
vectorisation step, tested without the engine.
"""

import numpy as np
from tichu_engine.legality import legal_actions_for
from tichu_engine.state import deal_initial_state
from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.action_space import ACTION_SPACE_SIZE
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
from tichu_training.search.engine_world import fast_rollout_leaf
from tichu_training.bc.heads import BCModel
from tichu_training.search.loop import (
    Sample,
    collect_round,
    samples_from_records,
    train_play_head,
)
from tichu_training.search.selfplay import SelfPlaySearchAgent


class StubPolicy:
    def play_action_scores(self, pv):
        legal = list(legal_actions_for(pv))
        return [(a, 1.0 / len(legal)) for a in legal]

    def act(self, pv):
        return min(legal_actions_for(pv), key=repr)

    def should_call(self, pv, kind):
        return False


class _FakeRec:
    def __init__(self, records):
        self.records = records


def test_samples_from_records_attaches_team_relative_outcome():
    # team0 +200: seats 0,2 (team0) see z=+2.0, seats 1,3 (team1) see z=-2.0.
    state = deal_initial_state(seed=5)
    pv = state.private_view(state.public.current_player)
    action = next(iter(legal_actions_for(pv)))
    rec = [(np.zeros(FEATURIZER_OUTPUT_DIM, dtype=np.float32), {action: 1.0})]
    agents = [_FakeRec(list(rec)) for _ in range(4)]

    samples = samples_from_records(agents, total=(200, 0))

    assert [s.z for s in samples] == [2.0, -2.0, 2.0, -2.0]
    assert all(isinstance(s, Sample) for s in samples)


def test_collect_round_produces_wellformed_samples():
    position = generate_full_position_pool(seed=0, n=1)[0]
    agents = [
        SelfPlaySearchAgent.from_policy(
            StubPolicy(), worlds=2, sims=8, leaf_fn=fast_rollout_leaf,
            root_alpha=1.0, root_eps=0.25, temperature=1.0, seed=s,
        )
        for s in range(4)
    ]

    samples = collect_round(agents, position)

    assert samples, "a full round must record at least one play decision"
    assert len(samples) == sum(len(a.records) for a in agents)
    for s in samples:
        assert s.features.shape == (FEATURIZER_OUTPUT_DIM,)
        assert s.target_probs.shape == (ACTION_SPACE_SIZE,)
        assert s.legal_mask.shape == (ACTION_SPACE_SIZE,)
        assert np.isfinite(s.z)


def test_train_play_head_drives_kl_down_toward_the_visit_target():
    import torch

    torch.manual_seed(0)
    model = BCModel(feature_dim=FEATURIZER_OUTPUT_DIM, skill_buckets=10, skill_dim=8,
                    trunk_hidden=32, trunk_depth=1, trunk_out_dim=16)
    legal_idx = [1, 4, 7]
    probs = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    probs[legal_idx] = [0.6, 0.3, 0.1]
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    mask[legal_idx] = True
    rng = np.random.default_rng(0)
    samples = [
        Sample(rng.standard_normal(FEATURIZER_OUTPUT_DIM).astype(np.float32),
               probs.copy(), mask.copy(), 0.0)
        for _ in range(64)
    ]

    history = train_play_head(model, samples, epochs=40, lr=1e-2)

    assert history[-1] < history[0]      # the play head learned toward π
    assert history[-1] < 0.1             # and got close to the target distribution
