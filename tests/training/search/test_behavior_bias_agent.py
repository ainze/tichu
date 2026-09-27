"""BiasedMLAgent — the champion with a Behavior Bias on its play logits.

A dummy exported policy stands in for the champion; the bias is pushed to ±1000
so the outcome does not depend on the dummy's random logits."""

import pytest

pytest.importorskip("torch")

from tichu_engine.legality import Pass, legal_actions_for
from tichu_inference.ml_agent import MLAgent
from tichu_ml.registry import build_agent
from tichu_training.action_space import ACTION_SPACE_SIZE
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
from tichu_training.search.behavior_bias_agent import BiasedMLAgent

from tests.inference.test_ml_agent import _export_dummy
from tests.training.search.test_behavior_bias import FOLLOW_OPP, _c, _pv


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    return _export_dummy(tmp_path_factory.mktemp("policy"), feature_dim=FEATURIZER_OUTPUT_DIM,
                         action_space_size=ACTION_SPACE_SIZE)


def _follow():
    return _pv([_c(9), _c(12), _c(13)], **FOLLOW_OPP)


def test_a_large_positive_pass_bias_always_passes(artifact):
    agent = BiasedMLAgent(artifact, lever="pass_vs_opponent", delta=1000.0)
    assert isinstance(agent.act(_follow()), Pass)


def test_a_large_negative_pass_bias_never_passes(artifact):
    agent = BiasedMLAgent(artifact, lever="pass_vs_opponent", delta=-1000.0)
    action = agent.act(_follow())
    assert not isinstance(action, Pass) and action in legal_actions_for(_follow())


def test_zero_delta_plays_exactly_like_the_champion(artifact):
    from tichu_engine.state import deal_initial_state

    base = MLAgent(artifact)
    biased = BiasedMLAgent(artifact, lever="single_lead", delta=0.0)
    for seed in range(5):
        state = deal_initial_state(seed=seed)
        pv = state.private_view(state.public.current_player)
        assert biased.act(pv) == base.act(pv)
    assert biased.act(_follow()) == base.act(_follow())


def test_the_bias_leaves_the_forward_memo_unbiased(artifact):
    agent = BiasedMLAgent(artifact, lever="pass_vs_opponent", delta=1000.0)
    pv = _follow()
    agent.act(pv)
    assert isinstance(agent.act(pv), Pass)                       # memo hit, still biased
    raw = MLAgent._play_logits(agent, pv)
    assert raw.max() < 1000.0                                    # the cached buffer was not touched


def test_registered_as_ml_biased(artifact):
    agent = build_agent("ml_biased", checkpoint_path=artifact, lever="dog_lead", delta=-2.0)
    assert isinstance(agent, BiasedMLAgent)


def test_an_unknown_lever_fails_at_construction(artifact):
    with pytest.raises(KeyError):
        BiasedMLAgent(artifact, lever="nope", delta=1.0)


# --- gap recording (δ calibration) --------------------------------------------

def test_the_gap_recorder_plays_exactly_like_the_champion(artifact):
    from tichu_engine.state import deal_initial_state
    from tichu_training.search.behavior_bias_agent import GapRecordingMLAgent

    base, rec = MLAgent(artifact), GapRecordingMLAgent(artifact)
    for seed in range(5):
        state = deal_initial_state(seed=seed)
        pv = state.private_view(state.public.current_player)
        assert rec.act(pv) == base.act(pv)
    assert rec.act(_follow()) == base.act(_follow())
    assert len(rec.gaps["pass_vs_opponent"]) == 1


def test_collected_gaps_are_the_same_serial_and_parallel(artifact):
    import pickle
    from functools import partial

    from tichu_eval.full_position_pool import generate_full_position_pool
    from tichu_ml.rule_agent import RuleAgent
    from tichu_training.cli.eval_matrix import _build_agent
    from tichu_training.search.behavior_bias_agent import collect_lever_gaps

    positions = generate_full_position_pool(seed=0, n=4)
    rec = partial(_build_agent, "ml_gaps", checkpoint_path=str(artifact))
    # An ML agent breaks ties between concrete realisations of one Intent by
    # legal-enumeration order, which follows the hand frozenset's iteration
    # order — and pickling a Position to a worker rebuilds that order. So serial
    # equals parallel only on pickled Positions (every parallel run is alike).
    serial = collect_lever_gaps(rec, RuleAgent, pickle.loads(pickle.dumps(positions)))
    parallel = collect_lever_gaps(rec, RuleAgent, positions, workers=2)
    # Same Decisions in the same order; values equal up to float noise. The
    # noise is larger than thread count alone explains (~3e-3 seen once the
    # main process had run other suites first — TorchScript re-optimises a
    # graph after warm-up), so the stitching is checked, not the last bits.
    assert serial.keys() == parallel.keys()
    for name in serial:
        assert serial[name] == pytest.approx(parallel[name], abs=1e-2), name
    assert len(serial["single_lead"]) > 0
