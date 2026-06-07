"""Claim-solver primitives: card-counting the Unseen Cards and certifying an
Unbeatable Lead.

An Unbeatable Lead is a Combination that, led into an empty Trick, no opponent
can legally beat from the Unseen Cards. The check pools all Unseen Cards into one
worst-case responder and asks the engine's own `legal_actions` — so Phoenix-
following and bomb rules come from the single source of truth, never a
reimplementation. See ADR-0032 and CONTEXT.md "Endgame / claim terms".
"""

from tichu_engine.cards import Card, DRAGON, PHOENIX, Suit
from tichu_engine.claim import (
    guaranteed_out_chain,
    is_unbeatable_lead,
    reclaim_out,
    unseen_cards,
)
from tichu_engine.combinations import Pair, Single
from tichu_engine.state import PrivateState, PublicState, Trick


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


def test_low_single_is_beatable_when_a_higher_single_is_unseen():
    lead = Single(_c(Suit.JADE, 5))
    unseen = frozenset({_c(Suit.SWORD, 6)})
    assert is_unbeatable_lead(lead, unseen) is False


def test_dragon_single_is_unbeatable_when_no_bomb_is_unseen():
    lead = Single(DRAGON)
    unseen = frozenset({_c(Suit.JADE, 3), _c(Suit.SWORD, 9)})
    assert is_unbeatable_lead(lead, unseen) is True


def test_ace_single_is_beatable_when_the_phoenix_is_unseen():
    # The Phoenix follows any single (bar the Dragon) by +0.5 — a hand-rolled
    # "higher same-rank" check would wrongly certify the Ace as unbeatable.
    lead = Single(_c(Suit.JADE, 14))
    unseen = frozenset({PHOENIX})
    assert is_unbeatable_lead(lead, unseen) is False


def test_dragon_single_is_beatable_when_a_bomb_is_unseen():
    # Even the highest single falls to a bomb. Pairs with the no-bomb case above.
    lead = Single(DRAGON)
    unseen = frozenset(
        {_c(s, 7) for s in (Suit.JADE, Suit.SWORD, Suit.PAGODA, Suit.STAR)}
    )
    assert is_unbeatable_lead(lead, unseen) is False


def test_pair_is_unbeatable_when_only_a_lower_pair_and_no_bomb_are_unseen():
    lead = Pair(_c(Suit.JADE, 13), _c(Suit.SWORD, 13))
    unseen = frozenset({_c(Suit.JADE, 4), _c(Suit.SWORD, 4)})  # a lower pair only
    assert is_unbeatable_lead(lead, unseen) is True


def test_unseen_cards_excludes_own_hand_and_played_cards():
    own = {_c(Suit.JADE, 2), _c(Suit.SWORD, 3)}
    played = {_c(Suit.PAGODA, 4)}
    public = PublicState(
        current_player=0,
        hand_sizes=(2, 0, 0, 0),
        scores=(0, 0),
        trick=Trick.empty(),
        played_cards_this_round=frozenset(played),
    )
    private = PrivateState(player=0, hand=frozenset(own), public=public)

    unseen = unseen_cards(private)

    assert own.isdisjoint(unseen)           # own hand is not unseen
    assert played.isdisjoint(unseen)        # already-played cards are not unseen
    assert _c(Suit.STAR, 14) in unseen      # an untouched card is unseen
    assert len(unseen) == 56 - len(own) - len(played)


# ---- Guaranteed Out: the uninterrupted chain (depth-0) ----

def test_one_combo_hand_is_a_chain_out_even_if_the_last_card_is_beatable():
    hand = frozenset({_c(Suit.JADE, 2)})
    unseen = frozenset({_c(Suit.SWORD, 14)})  # a higher card exists — irrelevant, it's our last
    assert guaranteed_out_chain(hand, unseen) == (Single(_c(Suit.JADE, 2)),)


def test_chain_orders_the_beatable_card_last():
    # {Dragon, 2}: only viable as Dragon (unbeatable, keep lead) then 2 (last, out).
    # Played 2-first, an opponent beats it and strands the Dragon.
    hand = frozenset({DRAGON, _c(Suit.JADE, 2)})
    unseen = frozenset({_c(Suit.SWORD, 5), _c(Suit.PAGODA, 9)})  # no bomb; both > 2
    assert guaranteed_out_chain(hand, unseen) == (Single(DRAGON), Single(_c(Suit.JADE, 2)))


def test_no_chain_when_no_ordering_retains_the_lead():
    # Two beatable singles: whichever leads is beaten and strands the other.
    hand = frozenset({_c(Suit.JADE, 2), _c(Suit.SWORD, 3)})
    unseen = frozenset({_c(Suit.PAGODA, 6)})  # beats both 2 and 3
    assert guaranteed_out_chain(hand, unseen) is None


def test_chain_uses_a_pair_when_the_singles_would_be_beaten():
    # A lone unseen Ace beats either King *single*, but there is no Ace *pair* to
    # beat the King pair — so the only chain leads the pair, then dumps the 2.
    kj, ks = _c(Suit.JADE, 13), _c(Suit.SWORD, 13)
    hand = frozenset({kj, ks, _c(Suit.JADE, 2)})
    unseen = frozenset({_c(Suit.PAGODA, 14)})
    assert guaranteed_out_chain(hand, unseen) == (Pair(kj, ks), Single(_c(Suit.JADE, 2)))


# ---- Reclaim Out: shed a beatable card while holding a re-entry (probabilistic) ----

def test_reclaim_out_sheds_a_single_then_reclaims_with_an_unbeatable_pair():
    # Hand {3, 5, A, A}: two stray beatable singles plus an unbeatable Ace pair.
    # Not a chain out (after the pair, {3,5} can't both go last); but a reclaim out —
    # shed the 3, keep the Ace pair in reserve to win the lead back, then dump the 5.
    aj, as_ = _c(Suit.JADE, 14), _c(Suit.SWORD, 14)
    hand = frozenset({_c(Suit.JADE, 3), _c(Suit.STAR, 5), aj, as_})
    unseen = frozenset({DRAGON, _c(Suit.PAGODA, 6), _c(Suit.SWORD, 7)})  # no bomb, no higher pair
    assert guaranteed_out_chain(hand, unseen) is None
    assert reclaim_out(hand, unseen) is not None


def test_reclaim_out_uses_a_bomb_as_the_re_entry():
    # {3, 5} stray singles + a four-of-a-kind bomb: shed a single, reclaim with the bomb.
    bomb_cards = {_c(s, 9) for s in (Suit.JADE, Suit.SWORD, Suit.PAGODA, Suit.STAR)}
    hand = frozenset({_c(Suit.JADE, 3), _c(Suit.STAR, 5)} | bomb_cards)
    unseen = frozenset({DRAGON, _c(Suit.PAGODA, 6), _c(Suit.SWORD, 8)})  # no higher bomb
    assert guaranteed_out_chain(hand, unseen) is None
    assert reclaim_out(hand, unseen) is not None


def test_no_reclaim_out_without_a_re_entry():
    # Two beatable singles, nothing unbeatable to reclaim with -> no reclaim out either.
    hand = frozenset({_c(Suit.JADE, 3), _c(Suit.STAR, 5)})
    unseen = frozenset({_c(Suit.PAGODA, 9)})  # beats both; no re-entry exists
    assert reclaim_out(hand, unseen) is None
