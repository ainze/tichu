"""Full-strength single-Round runner: Grand-Tichu -> Schupfen -> Tichu -> Play.

Drives the complete product stack through one Round and reports the team score
deltas plus the Call-bonus component. See ADR-0025.
"""

from tichu_engine.state import deal_for_schupfen
from tichu_eval.play_full import play_full_round
from tichu_ml.rule_agent import RuleAgent


def _grand_prefixes(start_state):
    """Each seat's first-8 Grand-Tichu Prefix, derived from its 14-card hand."""
    return tuple(frozenset(list(start_state.hands[p])[:8]) for p in range(4))


class ScriptedCaller(RuleAgent):
    """A RuleAgent (deterministic, identical play) that additionally answers the
    Call Decisions on command. Lets a test hold Play fixed while toggling calls."""

    def __init__(self, *, grand: bool = False, tichu: bool = False):
        super().__init__()
        self._grand = grand
        self._tichu = tichu

    def should_call(self, private_state, kind: str) -> bool:
        return self._grand if kind == "grand" else self._tichu


def test_baselines_play_full_round_and_decline_all_calls():
    start = deal_for_schupfen(seed=0)
    agents = tuple(RuleAgent() for _ in range(4))
    result = play_full_round(agents, start, _grand_prefixes(start))
    assert isinstance(result.total, tuple) and len(result.total) == 2
    assert isinstance(result.total[0], int) and isinstance(result.total[1], int)
    # RuleAgent exposes no `should_call`, so no Tichu / Grand-Tichu is ever made.
    assert result.call_bonus == (0, 0)


def test_grand_caller_bonus_equals_marginal_score_effect():
    start = deal_for_schupfen(seed=3)
    prefixes = _grand_prefixes(start)
    # Seat 0 (team 0) calls Grand-Tichu; everyone plays identically either way.
    calling = (ScriptedCaller(grand=True), RuleAgent(), RuleAgent(), RuleAgent())
    plain = tuple(RuleAgent() for _ in range(4))

    r_call = play_full_round(calling, start, prefixes)
    r_plain = play_full_round(plain, start, prefixes)

    # Calls only change scoring, never play -> the bonus is exactly the delta.
    assert r_plain.call_bonus == (0, 0)
    marginal = (r_call.total[0] - r_plain.total[0], r_call.total[1] - r_plain.total[1])
    assert r_call.call_bonus == marginal
    # One Grand-Tichu on team 0: +/-200 to team 0, nothing to team 1.
    assert abs(r_call.call_bonus[0]) == 200
    assert r_call.call_bonus[1] == 0


def test_tichu_caller_bonus_equals_marginal_score_effect():
    start = deal_for_schupfen(seed=5)
    prefixes = _grand_prefixes(start)
    # Seat 0 (team 0) calls Tichu before its first non-Pass Play; Play unchanged.
    calling = (ScriptedCaller(tichu=True), RuleAgent(), RuleAgent(), RuleAgent())
    plain = tuple(RuleAgent() for _ in range(4))

    r_call = play_full_round(calling, start, prefixes)
    r_plain = play_full_round(plain, start, prefixes)

    marginal = (r_call.total[0] - r_plain.total[0], r_call.total[1] - r_plain.total[1])
    assert r_call.call_bonus == marginal
    # One Tichu on team 0: +/-100 to team 0, nothing to team 1.
    assert abs(r_call.call_bonus[0]) == 100
    assert r_call.call_bonus[1] == 0


class SpyGrand(RuleAgent):
    """Records the hand size it is asked Grand-Tichu on; always declines."""

    def __init__(self):
        super().__init__()
        self.grand_hand_sizes: list[int] = []

    def should_call(self, private_state, kind: str) -> bool:
        if kind == "grand":
            self.grand_hand_sizes.append(len(private_state.hand))
        return False


def test_grand_is_decided_on_an_eight_card_prefix_state():
    start = deal_for_schupfen(seed=1)
    spy = SpyGrand()
    agents = (spy, RuleAgent(), RuleAgent(), RuleAgent())
    play_full_round(agents, start, _grand_prefixes(start))
    assert spy.grand_hand_sizes == [8]


def test_grand_caller_is_not_also_charged_for_tichu():
    start = deal_for_schupfen(seed=3)
    prefixes = _grand_prefixes(start)
    # Seat 0 would answer yes to BOTH; grand supersedes tichu, so only +/-200.
    both = (ScriptedCaller(grand=True, tichu=True), RuleAgent(), RuleAgent(), RuleAgent())
    r = play_full_round(both, start, prefixes)
    assert abs(r.call_bonus[0]) == 200  # not 300
    assert r.call_bonus[1] == 0


def test_state_observer_exposes_full_gamestate_per_play_decision():
    # ADR-0033: the Perfect-Info collector needs every seat's hand mid-round.
    start = deal_for_schupfen(seed=2)
    agents = tuple(RuleAgent() for _ in range(4))
    obs_views: list = []
    pi_states: list = []

    def observer(seat, private_state, action):
        obs_views.append((seat, private_state, action))

    def state_observer(seat, game_state, action):
        pi_states.append((seat, game_state, action))

    play_full_round(
        agents, start, _grand_prefixes(start),
        observer=observer, state_observer=state_observer,
    )

    assert len(pi_states) == len(obs_views) > 0
    for (seat, gs, act), (oseat, pv, oact) in zip(pi_states, obs_views):
        assert seat == oseat and act == oact
        # All four hands are visible in the GameState the hook receives.
        assert len(gs.hands) == 4
        # The observable view is reconstructible and matches the plain observer.
        assert gs.private_view(seat) == pv


def test_same_agents_and_position_produce_identical_result():
    start = deal_for_schupfen(seed=9)
    prefixes = _grand_prefixes(start)
    mk = lambda: (ScriptedCaller(grand=True), RuleAgent(), RuleAgent(), RuleAgent())
    assert play_full_round(mk(), start, prefixes) == play_full_round(mk(), start, prefixes)
