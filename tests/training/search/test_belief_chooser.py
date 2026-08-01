"""Belief-Optimal Chooser decision rule (ADR-0041).

The chooser deviates from the frozen champion only on the blunder-miner's
validated tier-2 criterion (world-win-rate and mean paired delta), which carries
a *measured* 0.7% false-positive rate. A raw argmax over a K-world mean is the
optimizer's curse — the mechanism behind piKL's -58.66 (ADR-0037).
"""

from tichu_engine.legality import legal_actions_for
from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_eval.play_full import play_full_round
from tichu_ml.rule_agent import RuleAgent
from tichu_training.search.belief_chooser import (
    BeliefOptimalChooser,
    choose_with_deadband,
)

_CHOSEN = "chosen"


def _rule(candidates):
    return choose_with_deadband(
        _CHOSEN, candidates, win_threshold=0.70, delta_threshold=15.0,
    )


def test_champion_action_stands_when_nothing_clears_the_deadband():
    """Noise must not move the chooser: a candidate that wins often but by a
    trivial margin, and one that wins big but rarely, both fail."""
    assert _rule({
        "often_but_tiny": [1.0, 2.0, 1.0, 3.0],      # win-rate 1.0, mean 1.75
        "rarely_but_big": [-10.0, -10.0, -10.0, 400.0],  # mean +92.5, win-rate 0.25
    }) is _CHOSEN


def test_the_best_clearing_candidate_wins():
    """Among candidates that clear both bars, the highest mean paired delta is
    taken — and a near-miss on either bar is not rescued by the other."""
    assert _rule({
        "clears_modestly":  [16.0, 20.0, 18.0, 22.0],   # win 1.0, mean 19.0
        "clears_strongly":  [40.0, 60.0, -5.0, 55.0],   # win 0.75, mean 37.5
        "misses_win_rate":  [90.0, 90.0, -5.0, -5.0],   # win 0.50, mean 42.5
        "misses_delta_bar": [14.0, 14.0, 14.0, 14.0],   # win 1.0, mean 14.0
    }) == "clears_strongly"


def test_a_candidate_with_no_worlds_is_not_considered():
    """An illegal-in-every-world candidate yields no deltas and must not be
    picked over the champion's action by an empty-mean accident."""
    assert _rule({"never_legal": []}) is _CHOSEN


class _CountingRule(RuleAgent):
    """RuleAgent that records the actions it was asked to produce."""

    def __init__(self):
        super().__init__()
        self.asked = []

    def act(self, private_state):
        action = super().act(private_state)
        self.asked.append(action)
        return action


def _chooser(seat, policy, agents, **kwargs):
    return BeliefOptimalChooser(policy, agents, seed=1, worlds=3, top_k=4, **kwargs)


def _run(chooser, seat, position):
    """Play a Round with `chooser` at `seat`, frozen RuleAgents elsewhere."""
    agents = [RuleAgent() for _ in range(4)]
    agents[seat] = chooser
    seen = []
    play_full_round(
        agents, position.state, position.grand_prefixes,
        state_observer=lambda s, gs, action: seen.append((s, gs, action)),
    )
    return seen


def test_an_unreachable_bar_makes_the_chooser_the_champion():
    """The additive-fallback discipline: with a deadband nothing can clear, the
    Chooser reproduces the frozen policy's action at every Decision. `D_off` and
    `D_on` are then measured against a baseline that is provably the champion."""
    position = generate_full_position_pool(seed=5, n=1)[0]
    seat = 0

    champion_actions = [
        (s, repr(a)) for s, _gs, a in _run(RuleAgent(), seat, position)
    ]
    chooser = _chooser(seat, RuleAgent(), [RuleAgent() for _ in range(4)],
                       delta_threshold=1e9)
    chooser_actions = [(s, repr(a)) for s, _gs, a in _run(chooser, seat, position)]

    assert chooser_actions == champion_actions


def test_the_chooser_only_ever_plays_a_legal_candidate():
    """With a permissive bar the Chooser deviates, but only ever to an action the
    frozen policy itself ranked — never outside the legal set."""
    position = generate_full_position_pool(seed=5, n=1)[0]
    seat = 0
    chooser = _chooser(seat, RuleAgent(), [RuleAgent() for _ in range(4)],
                       win_threshold=0.0, delta_threshold=-1e9)

    for s, gs, action in _run(chooser, seat, position):
        if s != seat:
            continue
        view = gs.private_view(s)
        assert action in set(legal_actions_for(view))
