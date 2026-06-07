"""ForcedClaimAgent forces the chain Guaranteed Out exactly at its triggers — a
leading Tichu/Grand caller, or a leading seat whose partner is the sole player out
(a live Slam) — and otherwise delegates to the wrapped policy. Validated through the
real Claim Solver on constructed states. See ADR-0032.
"""

from tichu_engine.cards import Card, DRAGON, Suit
from tichu_engine.combinations import Single
from tichu_engine.deck import fresh_deck
from tichu_engine.state import PrivateState, PublicState, Trick
from tichu_training.search.forced_claim import ForcedClaimAgent


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


_SENTINEL = object()


class _PolicyStub:
    def __init__(self) -> None:
        self.act_calls = 0

    def act(self, pv):
        self.act_calls += 1
        return _SENTINEL

    def should_call(self, pv, kind):
        return False

    def rank_actions(self, pv):
        return None


# A chain-out hand: {Dragon, 2} with no bomb unseen -> (Single(Dragon), Single(2)).
_OWN = {DRAGON, _c(Suit.JADE, 2)}
_UNSEEN = {_c(Suit.SWORD, 5), _c(Suit.PAGODA, 9)}


def _leading_state(*, callers=frozenset(), out_order=()) -> PrivateState:
    played = frozenset(fresh_deck()) - frozenset(_OWN) - frozenset(_UNSEEN)
    public = PublicState(
        current_player=0,
        hand_sizes=(len(_OWN), 0, 0, 0),
        scores=(0, 0),
        trick=Trick.empty(),
        tichu_callers=frozenset(callers),
        out_order=tuple(out_order),
        played_cards_this_round=played,
    )
    return PrivateState(player=0, hand=frozenset(_OWN), public=public)


def test_forces_chain_out_first_move_for_a_leading_caller():
    agent = ForcedClaimAgent.from_policy(_PolicyStub())
    pv = _leading_state(callers={0})
    assert agent.act(pv) == Single(DRAGON)
    assert agent.forced_count == 1


def test_forces_chain_out_for_a_live_slam_partner_sole_out():
    agent = ForcedClaimAgent.from_policy(_PolicyStub())
    pv = _leading_state(out_order=(2,))  # partner of seat 0 is seat 2
    assert agent.act(pv) == Single(DRAGON)


def test_delegates_to_policy_when_not_a_caller_or_slam():
    policy = _PolicyStub()
    agent = ForcedClaimAgent.from_policy(policy)
    assert agent.act(_leading_state()) is _SENTINEL
    assert policy.act_calls == 1
    assert agent.forced_count == 0


def test_reclaim_mode_forces_where_chain_mode_would_not():
    # {3, 5, A, A}: not a chain out, but a reclaim out (shed a single, reclaim with the pair).
    aj, as_ = _c(Suit.JADE, 14), _c(Suit.SWORD, 14)
    own = {_c(Suit.JADE, 3), _c(Suit.STAR, 5), aj, as_}
    unseen = {DRAGON, _c(Suit.PAGODA, 6), _c(Suit.SWORD, 7)}
    played = frozenset(fresh_deck()) - frozenset(own) - frozenset(unseen)
    public = PublicState(
        current_player=0, hand_sizes=(4, 0, 0, 0), scores=(0, 0),
        trick=Trick.empty(), tichu_callers=frozenset({0}), played_cards_this_round=played,
    )
    pv = PrivateState(player=0, hand=frozenset(own), public=public)

    assert ForcedClaimAgent.from_policy(_PolicyStub()).act(pv) is _SENTINEL  # chain: delegates
    reclaim_agent = ForcedClaimAgent.from_policy(_PolicyStub(), reclaim=True)
    assert reclaim_agent.act(pv) is not _SENTINEL                            # reclaim: forces
    assert reclaim_agent.forced_count == 1


def test_delegates_when_following_even_as_a_caller():
    # A caller, but a top combination is on the table -> not leading -> chain does not apply.
    played = frozenset(fresh_deck()) - frozenset(_OWN) - frozenset(_UNSEEN)
    trick = Trick.empty().add_play(player=3, combination=Single(_c(Suit.SWORD, 5)))
    public = PublicState(
        current_player=0,
        hand_sizes=(len(_OWN), 0, 0, 0),
        scores=(0, 0),
        trick=trick,
        tichu_callers=frozenset({0}),
        played_cards_this_round=played,
    )
    policy = _PolicyStub()
    agent = ForcedClaimAgent.from_policy(policy)
    assert agent.act(PrivateState(player=0, hand=frozenset(_OWN), public=public)) is _SENTINEL
    assert policy.act_calls == 1
