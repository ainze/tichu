"""The caller-pressure probes must fire only when following an OPPONENT's trick while an
opponent has called, and force the right correction (press / yield). Validated through the
real detection path with a stub policy — no torch. See `caller_pressure_probes` docstring.
"""

from tichu_engine.cards import Card, Suit
from tichu_engine.combinations import FourOfAKindBomb, Single
from tichu_engine.legality import Pass
from tichu_engine.state import (
    MahjongWishPending,
    Play,
    PrivateState,
    PublicState,
    Trick,
)
from tichu_training.search.caller_pressure_probes import (
    ForcedPressOppCallerAgent,
    ForcedYieldOppCallerAgent,
    ForcedYieldPartnerCallerAgent,
    _following_opp_caller,
    _following_partner_caller,
)


def _c(suit, rank):
    return Card(suit=suit, rank=rank)


A = _c(Suit.JADE, 14)
K = _c(Suit.SWORD, 13)
BOMB = FourOfAKindBomb(_c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7), _c(Suit.STAR, 7))


class _Stub:
    def __init__(self, action, ranking=None):
        self._a = action
        self._r = ranking
    def act(self, pv):
        return self._a
    def should_call(self, pv, kind):
        return False
    def rank_actions(self, pv):
        return self._r


def _pv(player, hand, *, leader=None, top=None, callers=frozenset(), pending=None):
    trick = Trick(plays=(Play(player=leader, combination=top),), leader=leader) if top else Trick.empty()
    hand = frozenset(hand)
    sizes = [5, 5, 5, 5]
    sizes[player] = len(hand)
    pub = PublicState(current_player=player, hand_sizes=tuple(sizes), scores=(0, 0),
                      trick=trick, tichu_callers=callers, pending_decision=pending)
    return PrivateState(player=player, hand=hand, public=pub)


# --- trigger predicate -------------------------------------------------------

def test_following_opp_caller_true_only_for_opponent_trick_and_opponent_caller():
    # following opponent (seat 1) who has called -> True
    assert _following_opp_caller(_pv(0, {A}, leader=1, top=Single(K), callers=frozenset({1})))
    # an opponent (seat 3) called while following opponent seat 1 -> still True (any opp caller)
    assert _following_opp_caller(_pv(0, {A}, leader=1, top=Single(K), callers=frozenset({3})))


def test_following_opp_caller_false_cases():
    # leading (no top)
    assert not _following_opp_caller(_pv(0, {A}, callers=frozenset({1})))
    # following partner (seat 2)
    assert not _following_opp_caller(_pv(0, {A}, leader=2, top=Single(K), callers=frozenset({1})))
    # opponent leads but no caller
    assert not _following_opp_caller(_pv(0, {A}, leader=1, top=Single(K)))
    # only caller is partner / self (not an opponent)
    assert not _following_opp_caller(_pv(0, {A}, leader=1, top=Single(K), callers=frozenset({2})))
    assert not _following_opp_caller(_pv(0, {A}, leader=1, top=Single(K), callers=frozenset({0})))
    # pending decision (e.g. wish) — not a normal following play
    assert not _following_opp_caller(
        _pv(0, {A}, leader=1, top=Single(K), callers=frozenset({1}), pending=MahjongWishPending(player=0)))


# --- forced_press_opp_caller -------------------------------------------------

def test_press_forces_best_non_bomb_beat_when_master_cedes():
    pv = _pv(0, {A}, leader=1, top=Single(K), callers=frozenset({1}))
    ace = Single(A)
    agent = ForcedPressOppCallerAgent.from_policy(_Stub(Pass(), ranking=[ace, Pass()]))
    assert agent.act(pv) is ace
    assert agent.interventions == 1


def test_press_leaves_master_pass_when_no_opp_caller():
    pv = _pv(0, {A}, leader=1, top=Single(K))  # no caller
    p = Pass()
    agent = ForcedPressOppCallerAgent.from_policy(_Stub(p, ranking=[Single(A), Pass()]))
    assert agent.act(pv) is p
    assert agent.interventions == 0


def test_press_leaves_a_non_pass_master_action_alone():
    pv = _pv(0, {A}, leader=1, top=Single(K), callers=frozenset({1}))
    ace = Single(A)
    agent = ForcedPressOppCallerAgent.from_policy(_Stub(ace, ranking=[ace]))
    assert agent.act(pv) is ace  # only converts cedes
    assert agent.interventions == 0


# --- forced_yield_opp_caller -------------------------------------------------

def test_yield_forces_pass_when_master_contests_a_caller():
    pv = _pv(0, {A}, leader=1, top=Single(K), callers=frozenset({1}))
    agent = ForcedYieldOppCallerAgent.from_policy(_Stub(Single(A)))
    assert isinstance(agent.act(pv), Pass)
    assert agent.interventions == 1


def test_yield_leaves_a_bomb_alone():
    # bombing is a separate lever; yield must not convert a bomb to a pass.
    pv = _pv(0, {_c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7), _c(Suit.STAR, 7)},
             leader=1, top=Single(K), callers=frozenset({1}))
    agent = ForcedYieldOppCallerAgent.from_policy(_Stub(BOMB))
    assert agent.act(pv) is BOMB
    assert agent.interventions == 0


# --- forced_yield_partner_caller (the blunder EV probe) ----------------------

def test_following_partner_caller_true_only_for_calling_partner_trick():
    # following partner (seat 2) who has called -> True (partner of seat 0 is seat 2)
    assert _following_partner_caller(_pv(0, {A}, leader=2, top=Single(K), callers=frozenset({2})))


def test_following_partner_caller_false_cases():
    assert not _following_partner_caller(_pv(0, {A}, callers=frozenset({2})))          # leading
    assert not _following_partner_caller(_pv(0, {A}, leader=1, top=Single(K), callers=frozenset({1})))  # opponent's trick
    assert not _following_partner_caller(_pv(0, {A}, leader=2, top=Single(K)))          # partner leads but no caller
    assert not _following_partner_caller(_pv(0, {A}, leader=2, top=Single(K), callers=frozenset({0})))  # caller is self, not partner
    assert not _following_partner_caller(
        _pv(0, {A}, leader=2, top=Single(K), callers=frozenset({2}), pending=MahjongWishPending(player=0)))


def test_partner_yield_forces_pass_over_a_calling_winning_partner():
    pv = _pv(0, {A}, leader=2, top=Single(K), callers=frozenset({2}))
    agent = ForcedYieldPartnerCallerAgent.from_policy(_Stub(Single(A)))
    assert isinstance(agent.act(pv), Pass)
    assert agent.interventions == 1


def test_partner_yield_leaves_a_bomb_alone():
    pv = _pv(0, {_c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7), _c(Suit.STAR, 7)},
             leader=2, top=Single(K), callers=frozenset({2}))
    agent = ForcedYieldPartnerCallerAgent.from_policy(_Stub(BOMB))
    assert agent.act(pv) is BOMB
    assert agent.interventions == 0


def test_partner_yield_leaves_action_alone_when_partner_did_not_call():
    pv = _pv(0, {A}, leader=2, top=Single(K))  # partner winning but no call
    ace = Single(A)
    agent = ForcedYieldPartnerCallerAgent.from_policy(_Stub(ace))
    assert agent.act(pv) is ace
    assert agent.interventions == 0
