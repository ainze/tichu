"""Concrete Resolver: choosing WHICH cards realise a chosen Intent.

The v1 Action Space is intent-level — `PlayPair(rank=5, with_phoenix=False)` is
one slot, however many concrete pairs of 5s the hand can build. Every concrete
variant therefore shares one logit, and the pick among them is a tie-break the
network cannot express. These tests pin the tie-break: prefer the variant that
keeps the hand's bombs, then the one that fragments same-suit runs least.
"""

from tichu_engine.cards import Card, Suit
from tichu_engine.combinations import Pair

from tichu_training.concrete_resolver import resolution_cost_fn


def _hand(*cards):
    return frozenset(cards)


J, S, T = Suit.JADE, Suit.SWORD, Suit.STAR


def test_cost_prefers_the_pair_that_keeps_a_straight_flush_bomb():
    # jade 3-4-5-6-7 is a straight-flush bomb. Three concrete pairs of 5s exist
    # and all three collapse to one Intent slot; only the off-suit pair leaves
    # the bomb standing.
    hand = _hand(
        Card(suit=J, rank=3), Card(suit=J, rank=4), Card(suit=J, rank=5),
        Card(suit=J, rank=6), Card(suit=J, rank=7),
        Card(suit=S, rank=5), Card(suit=T, rank=5),
    )
    cost = resolution_cost_fn(hand)

    keeps_bomb = Pair(Card(suit=S, rank=5), Card(suit=T, rank=5))
    breaks_bomb = Pair(Card(suit=J, rank=5), Card(suit=S, rank=5))

    assert cost(keeps_bomb)[0] == 0
    assert cost(breaks_bomb)[0] == 1
    assert cost(keeps_bomb) < cost(breaks_bomb)


def test_cheapest_straight_variant_is_the_one_that_spares_a_bomb():
    # The motivating case: jade 3-7 and sword 3-7 are both straight-flush bombs,
    # and 32 concrete 3-4-5-6-7 straights collapse to a single Intent slot. 30 of
    # them shred both bombs; the two single-suit ones spend only their own.
    from tichu_engine.legality import _enumerate_all
    from tichu_training.action_space import play_intent_index

    hand = _hand(*(Card(suit=s, rank=r) for s in (J, S) for r in range(3, 8)))
    cost = resolution_cost_fn(hand)

    target = play_intent_index(
        next(
            c for c in _enumerate_all(hand)
            if type(c).__name__ == "Straight" and len(c.cards) == 5
        )
    )
    variants = [
        c for c in _enumerate_all(hand)
        if type(c).__name__ == "Straight" and play_intent_index(c) == target
    ]
    assert len(variants) == 32  # the tie group the network sees as one logit

    best = min(variants, key=cost)
    assert len({c.suit for c in best.cards}) == 1  # single-suit: spares the other bomb
    assert cost(best)[0] == 1
    assert max(cost(v)[0] for v in variants) == 2


def test_cost_prefers_the_variant_that_fragments_a_suit_run_least():
    # No bombs in this hand at all, so only the fragment term can discriminate.
    # jade 4-5-6 is a live straight-flush fragment; spending the jade 5 splits it
    # into two isolated cards, spending the off-suit 5s leaves it whole.
    hand = _hand(
        Card(suit=J, rank=4), Card(suit=J, rank=5), Card(suit=J, rank=6),
        Card(suit=S, rank=5), Card(suit=T, rank=5),
    )
    cost = resolution_cost_fn(hand)

    keeps_run = Pair(Card(suit=S, rank=5), Card(suit=T, rank=5))
    splits_run = Pair(Card(suit=J, rank=5), Card(suit=S, rank=5))

    assert cost(keeps_run) == (0, 0)
    assert cost(splits_run) == (0, 2)  # loses both 4-5 and 5-6 adjacencies
    assert cost(keeps_run) < cost(splits_run)


def test_cost_is_free_for_actions_that_spend_no_cards():
    # Pass and the pending-decision actions are in the same legal set as
    # combinations, so the cost function has to accept them. They consume
    # nothing, so they cost nothing.
    from tichu_engine.legality import PASS, DragonGive, MahjongWish

    hand = _hand(*(Card(suit=J, rank=r) for r in range(3, 8)))
    cost = resolution_cost_fn(hand)

    assert cost(PASS) == (0, 0)
    assert cost(MahjongWish(rank=7)) == (0, 0)
    assert cost(DragonGive(target=1)) == (0, 0)
