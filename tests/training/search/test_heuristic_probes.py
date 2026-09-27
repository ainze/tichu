"""Each heuristic probe must force its prescribed action in EXACTLY its trigger state and
pass the master through everywhere else. Validated through the real detection path
(legal_actions_for + the real PublicState fields the runner injects), with a stub policy
standing in for the master — no torch, no checkpoints. See `heuristic_probes` docstring.
"""

from tichu_engine.cards import Card, DRAGON, MAHJONG, Suit
from tichu_engine.combinations import FourOfAKindBomb, Pair, Single
from tichu_engine.legality import DragonGive, MahjongWish, Pass, legal_actions_for
from tichu_engine.state import (
    DragonGivePending,
    MahjongWishPending,
    Play,
    PrivateState,
    PublicState,
    Trick,
)
from tichu_training.search.heuristic_probes import (
    ForcedDragonLastoutAgent,
    ForcedFollowLowAgent,
    ForcedKeepPartnerTrickAgent,
    ForcedSplitAcesAgent,
    ForcedSupportTichuAgent,
    PartnerTrickGuardAgent,
    suppress_partner_trick_bomb,
)


def _card(suit, rank):
    return Card(suit=suit, rank=rank)


A_JADE = _card(Suit.JADE, 14)
A_SWORD = _card(Suit.SWORD, 14)
K_JADE = _card(Suit.JADE, 13)
K_SWORD = _card(Suit.SWORD, 13)
BOMB_5S = FourOfAKindBomb(
    _card(Suit.JADE, 5), _card(Suit.SWORD, 5), _card(Suit.PAGODA, 5), _card(Suit.STAR, 5)
)


class _Stub:
    """A master stand-in: returns a fixed action, declines calls, ranks trivially."""

    def __init__(self, action):
        self._action = action

    def act(self, pv):
        return self._action

    def should_call(self, pv, kind):
        return False

    def rank_actions(self, pv):
        return [self._action]


def _pv(player, hand, public):
    return PrivateState(player=player, hand=frozenset(hand), public=public)


def _leading_public(hand_sizes):
    return PublicState(
        current_player=0, hand_sizes=hand_sizes, scores=(0, 0), trick=Trick.empty()
    )


def _following_public(hand_sizes, *, leader, top, tichu_callers=frozenset()):
    trick = Trick(plays=(Play(player=leader, combination=top),), leader=leader)
    return PublicState(
        current_player=0,
        hand_sizes=hand_sizes,
        scores=(0, 0),
        trick=trick,
        tichu_callers=tichu_callers,
    )


# --- forced_split_aces -------------------------------------------------------

def test_split_aces_breaks_a_natural_ace_pair_lead():
    pub = _leading_public((2, 5, 5, 5))
    pv = _pv(0, {A_JADE, A_SWORD}, pub)
    agent = ForcedSplitAcesAgent.from_policy(_Stub(Pair(A_JADE, A_SWORD)))
    out = agent.act(pv)
    assert isinstance(out, Single) and out.card.rank == 14
    assert agent.interventions == 1


def test_split_aces_leaves_non_ace_pair_lead_alone():
    pub = _leading_public((2, 5, 5, 5))
    pv = _pv(0, {K_JADE, K_SWORD}, pub)
    kings = Pair(K_JADE, K_SWORD)
    agent = ForcedSplitAcesAgent.from_policy(_Stub(kings))
    assert agent.act(pv) is kings
    assert agent.interventions == 0


# --- forced_follow_low -------------------------------------------------------

def test_follow_low_replaces_dragon_with_cheapest_natural_beat():
    # Following a King single, holding {Ace, Dragon}: master plays the Dragon; the
    # probe must force the natural Ace (the cheapest legal beat).
    pub = _following_public((2, 1, 1, 1), leader=1, top=Single(K_SWORD))
    pv = _pv(0, {A_JADE, DRAGON}, pub)
    agent = ForcedFollowLowAgent.from_policy(_Stub(Single(DRAGON)))
    out = agent.act(pv)
    assert isinstance(out, Single) and out.card == A_JADE
    assert agent.interventions == 1


def test_follow_low_keeps_master_when_it_already_follows_low():
    pub = _following_public((2, 1, 1, 1), leader=1, top=Single(K_SWORD))
    pv = _pv(0, {A_JADE, DRAGON}, pub)
    ace = Single(A_JADE)
    agent = ForcedFollowLowAgent.from_policy(_Stub(ace))
    assert agent.act(pv) is ace
    assert agent.interventions == 0


def test_follow_low_passes_through_a_pending_mahjong_wish():
    # The Mahjong holder must declare a wish: legal actions are MahjongWish options,
    # not combinations. The probe must not try to ladder over them (regression).
    pub = PublicState(
        current_player=0,
        hand_sizes=(1, 1, 1, 1),
        scores=(0, 0),
        trick=Trick(plays=(Play(player=0, combination=Single(MAHJONG)),), leader=0),
        pending_decision=MahjongWishPending(player=0),
    )
    pv = _pv(0, {DRAGON}, pub)
    wish = MahjongWish(rank=5)
    agent = ForcedFollowLowAgent.from_policy(_Stub(wish))
    assert agent.act(pv) is wish
    assert agent.interventions == 0


def test_follow_low_keeps_premium_when_it_is_the_only_beat():
    # Only a Dragon beats the King — no cheaper natural beat exists, so leave it.
    pub = _following_public((1, 1, 1, 1), leader=1, top=Single(K_SWORD))
    pv = _pv(0, {DRAGON}, pub)
    dragon = Single(DRAGON)
    agent = ForcedFollowLowAgent.from_policy(_Stub(dragon))
    assert agent.act(pv) is dragon
    assert agent.interventions == 0


# --- forced_support_tichu ----------------------------------------------------

def test_support_tichu_passes_instead_of_overtaking_called_partner():
    pub = _following_public((1, 1, 1, 1), leader=2, top=Single(K_SWORD),
                            tichu_callers=frozenset({2}))
    pv = _pv(0, {A_JADE}, pub)
    agent = ForcedSupportTichuAgent.from_policy(_Stub(Single(A_JADE)))
    assert isinstance(agent.act(pv), Pass)
    assert agent.interventions == 1


def test_support_tichu_leaves_overtake_alone_when_partner_has_not_called():
    pub = _following_public((1, 1, 1, 1), leader=2, top=Single(K_SWORD))  # no callers
    pv = _pv(0, {A_JADE}, pub)
    ace = Single(A_JADE)
    agent = ForcedSupportTichuAgent.from_policy(_Stub(ace))
    assert agent.act(pv) is ace
    assert agent.interventions == 0


# --- forced_keep_partner_trick ----------------------------------------------

def test_keep_partner_trick_suppresses_a_bomb_on_partners_trick():
    pub = _following_public((4, 1, 1, 1), leader=2, top=Single(K_SWORD))  # partner=2 leads
    pv = _pv(0, {_card(Suit.JADE, 5), _card(Suit.SWORD, 5),
                 _card(Suit.PAGODA, 5), _card(Suit.STAR, 5)}, pub)
    agent = ForcedKeepPartnerTrickAgent.from_policy(_Stub(BOMB_5S))
    assert isinstance(agent.act(pv), Pass)
    assert agent.interventions == 1


def test_keep_partner_trick_allows_bombing_an_opponent_trick():
    pub = _following_public((4, 1, 1, 1), leader=1, top=Single(K_SWORD))  # opponent=1 leads
    pv = _pv(0, {_card(Suit.JADE, 5), _card(Suit.SWORD, 5),
                 _card(Suit.PAGODA, 5), _card(Suit.STAR, 5)}, pub)
    agent = ForcedKeepPartnerTrickAgent.from_policy(_Stub(BOMB_5S))
    assert agent.act(pv) is BOMB_5S
    assert agent.interventions == 0


# --- partner_trick_guard (the retired served guard) --------------------------

FIVES = {_card(Suit.JADE, 5), _card(Suit.SWORD, 5), _card(Suit.PAGODA, 5), _card(Suit.STAR, 5)}


def _bomb_pv(*, leader, extra=frozenset(), tichu_callers=frozenset()):
    hand = FIVES | set(extra)
    pub = _following_public((len(hand), 5, 5, 5), leader=leader, top=Single(K_SWORD),
                            tichu_callers=tichu_callers)
    return _pv(0, hand, pub)


def test_guard_suppresses_bomb_on_partners_trick():
    pv = _bomb_pv(leader=2)
    assert suppress_partner_trick_bomb(pv, BOMB_5S, legal_actions_for(pv)) is True


def test_guard_allows_bombing_an_opponent_trick():
    pv = _bomb_pv(leader=1)
    assert suppress_partner_trick_bomb(pv, BOMB_5S, legal_actions_for(pv)) is False


def test_guard_allows_non_bomb_actions():
    pv = _bomb_pv(leader=2, extra={A_JADE})
    assert suppress_partner_trick_bomb(pv, Single(A_JADE), legal_actions_for(pv)) is False


def test_guard_carveout_lets_a_caller_go_out_on_the_bomb():
    # Seat 0 called Tichu and the bomb is its whole hand: playing it goes out.
    pv = _bomb_pv(leader=2, tichu_callers=frozenset({0}))
    assert suppress_partner_trick_bomb(pv, BOMB_5S, legal_actions_for(pv)) is False


def test_guard_still_fires_for_a_caller_not_going_out():
    pv = _bomb_pv(leader=2, extra={_card(Suit.JADE, 3)}, tichu_callers=frozenset({0}))
    assert suppress_partner_trick_bomb(pv, BOMB_5S, legal_actions_for(pv)) is True


def test_partner_trick_guard_probe_turns_the_bomb_into_a_pass():
    pv = _bomb_pv(leader=2)
    agent = PartnerTrickGuardAgent.from_policy(_Stub(BOMB_5S))
    assert isinstance(agent.act(pv), Pass)
    assert agent.interventions == 1


# --- forced_dragon_lastout ---------------------------------------------------

def _dragon_pending_pv(hand_sizes):
    pub = PublicState(
        current_player=0,
        hand_sizes=hand_sizes,
        scores=(0, 0),
        trick=Trick.empty(),
        pending_decision=DragonGivePending(winner=0, points=25),
    )
    return _pv(0, {DRAGON}, pub)


def test_dragon_give_targets_the_opponent_with_more_cards():
    # Winner 0; opponents are 1 (6 cards) and 3 (3 cards) -> give to 1 (out last).
    pv = _dragon_pending_pv((1, 6, 5, 3))
    agent = ForcedDragonLastoutAgent.from_policy(_Stub(None))
    out = agent.act(pv)
    assert out == DragonGive(target=1)
    assert agent.interventions == 1


def test_dragon_give_breaks_ties_to_the_right():
    # Opponents 1 and 3 hold equal hands -> tie-break to the winner's right, seat 3.
    pv = _dragon_pending_pv((1, 5, 5, 5))
    agent = ForcedDragonLastoutAgent.from_policy(_Stub(None))
    out = agent.act(pv)
    assert out == DragonGive(target=3)
    assert agent.interventions == 1
