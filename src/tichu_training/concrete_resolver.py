"""Concrete Resolver: pick WHICH cards realise a chosen Intent.

The v1 Action Space is intent-level (see `action_space`): `PlayPair(rank=5,
with_phoenix=False)` is one slot no matter how many concrete pairs of 5s the
hand can build. Every concrete variant of an Intent therefore shares one
logit, so the network cannot express a preference among them — the pick is a
tie-break, and until now it was whatever order the legal-action frozenset
happened to iterate in.

This module supplies the ordering the network can't: among variants of one
Intent, prefer the one that costs the hand the least future structure. Both
the served agent (`tichu_inference.ml_agent`) and the co-train rollout
(`ppo.policy`) resolve through it, so the training world plays the learner's
hand the same way the exported agent does.
"""

from tichu_engine.cards import Card
from tichu_engine.combinations import (
    FourOfAKindBomb,
    FullHouse,
    Pair,
    PairStep,
    Single,
    Straight,
    StraightFlushBomb,
    Triple,
)
from tichu_engine.enumeration import (
    enumerate_four_of_a_kind_bombs,
    enumerate_straight_flush_bombs,
)
from tichu_engine.legality import _cards_in

# Pass and the pending-decision actions share the legal set with combinations
# but spend no cards, so they never cost the hand anything.
_COMBINATIONS = (
    Single, Pair, Triple, FullHouse, Straight, PairStep,
    FourOfAKindBomb, StraightFlushBomb,
)


def _bombs(hand) -> list:
    return list(enumerate_four_of_a_kind_bombs(hand)) + list(
        enumerate_straight_flush_bombs(hand)
    )


def _flush_edges(hand) -> tuple:
    """Same-suit adjacent-rank card pairs in `hand` — a decomposition of how much
    live straight-flush structure it holds. A 5-card run contributes 4 edges, so
    taking a card off an end costs 1 and taking one from the middle costs 2: the
    metric prefers trimming ends over splitting runs. Special cards hold no suit
    and contribute nothing."""
    by_suit: dict[object, dict[int, Card]] = {}
    for card in hand:
        if isinstance(card, Card):
            by_suit.setdefault(card.suit, {})[card.rank] = card
    return tuple(
        (card, ranks[card.rank + 1])
        for ranks in by_suit.values()
        for card in ranks.values()
        if card.rank + 1 in ranks
    )


def resolution_cost_fn(hand):
    """Return `cost(action) -> (bombs_lost, flush_edges_lost)` for actions played
    from `hand`.

    Lower is better; the tuple is compared lexicographically, so keeping a bomb
    outranks keeping suit-run structure. Costs are before-minus-after on `hand`
    vs `hand - cards(action)`, which makes them meaningful only *between
    variants of the same Intent* — those spend the same ranks and differ only in
    which suits they take.

    Note the bomb term discriminates only for straight-flush bombs: a
    four-of-a-kind occupies all four suits of a rank, so every variant of an
    Intent that touches that rank breaks it equally.

    Callers only invoke `cost` for Intents with more than one realisation —
    everything else has no tie to break. Hand-level state is precomputed here so
    the returned closure stays cheap when they do:

    * Removing cards can never create an adjacency, so the edges an action
      destroys are exactly the hand's edges with a spent endpoint. Giving each
      card a bitmask of the edges it touches turns the fragment term into a few
      dict lookups and a popcount, instead of re-deriving the suit runs of the
      remaining hand once per action.
    * A bomb can only break if the action spends one of its cards, so the
      enumerators only run for actions that touch `bomb_cards` (and never at all
      for a bomb-less hand, the common case).
    """
    hand = frozenset(hand)
    bombs = _bombs(hand)
    n_bombs = len(bombs)
    bomb_cards = frozenset(c for b in bombs for c in _cards_in(b))

    edge_mask: dict[object, int] = {}
    for bit, (low, high) in enumerate(_flush_edges(hand)):
        edge_mask[low] = edge_mask.get(low, 0) | (1 << bit)
        edge_mask[high] = edge_mask.get(high, 0) | (1 << bit)

    def cost(action) -> tuple[int, int]:
        if not isinstance(action, _COMBINATIONS):
            return (0, 0)
        spent = _cards_in(action)

        bombs_lost = 0
        if n_bombs and any(c in bomb_cards for c in spent):
            bombs_lost = n_bombs - len(_bombs(hand - frozenset(spent)))

        broken = 0
        for card in spent:
            broken |= edge_mask.get(card, 0)
        return (bombs_lost, broken.bit_count())

    return cost
