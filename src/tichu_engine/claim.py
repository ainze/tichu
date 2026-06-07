"""Claim-solver primitives.

Card-counting the **Unseen Cards** and certifying an **Unbeatable Lead** — the
deterministic, exact building blocks of the endgame Claim Solver (ADR-0032;
CONTEXT.md "Endgame / claim terms"). These are pure rules queries, not strategy.

A lead into an empty Trick can only be beaten by a combination of its **own type**
(same length, higher rank) or by a **Bomb** — never by another type. So unbeatability
is decided by enumerating just those candidate beaters from the Unseen Cards and
deferring the comparison to the engine's `_beats`. (Enumerating *every* combination of
the ~full-deck Unseen set instead would explode on straights/full-houses/pair-steps.)
The one cross-type subtlety — the Phoenix *following* a single by +0.5 — is handled
explicitly, matching the engine's `legal_actions`.
"""

from tichu_engine.cards import DRAGON, PHOENIX
from tichu_engine.combinations import (
    CardOrSpecial,
    FourOfAKindBomb,
    FullHouse,
    Pair,
    PairStep,
    Single,
    Straight,
    StraightFlushBomb,
    Triple,
)
from tichu_engine.deck import fresh_deck
from tichu_engine.enumeration import (
    enumerate_four_of_a_kind_bombs,
    enumerate_full_houses,
    enumerate_pair_steps,
    enumerate_pairs,
    enumerate_singles,
    enumerate_straight_flush_bombs,
    enumerate_straights,
    enumerate_triples,
)
from tichu_engine.legality import Combination, _beats, _cards_in, _enumerate_all
from tichu_engine.state import PrivateState

_SAME_TYPE_ENUM = {
    Single: enumerate_singles,
    Pair: enumerate_pairs,
    Triple: enumerate_triples,
    FullHouse: enumerate_full_houses,
    Straight: enumerate_straights,
    PairStep: enumerate_pair_steps,
    FourOfAKindBomb: enumerate_four_of_a_kind_bombs,
    StraightFlushBomb: enumerate_straight_flush_bombs,
}


def unseen_cards(private_state: PrivateState) -> frozenset[CardOrSpecial]:
    """The Unseen Cards from `private_state`'s perspective — every card not in
    the acting Player's Hand and not yet played this Round.

    Exact common knowledge: `fresh_deck − hand − played_cards_this_round`, with no
    inference. These are the cards that could be in an opponent's Hand (the joint
    partition is the Belief Model's job; this is just the certain set).
    """
    return (
        frozenset(fresh_deck())
        - private_state.hand
        - private_state.public.played_cards_this_round
    )


def is_unbeatable_lead(lead: Combination, unseen: frozenset[CardOrSpecial]) -> bool:
    """True iff `lead`, led into an empty Trick, cannot be beaten by any
    combination formable from `unseen` (worst-case: all Unseen Cards pooled into
    one responder — exact for a single Trick).

    Only same-type-higher combinations and Bombs can beat a lead, so only those are
    enumerated; the comparison itself is the engine's `_beats`.
    """
    # Bombs beat any non-bomb (and higher bombs beat lower) — always check them.
    for bomb in enumerate_four_of_a_kind_bombs(unseen):
        if _beats(bomb, lead):
            return False
    for bomb in enumerate_straight_flush_bombs(unseen):
        if _beats(bomb, lead):
            return False
    # Same-type beaters (higher rank, matching length where it applies).
    enum = _SAME_TYPE_ENUM.get(type(lead))
    if enum is not None:
        for candidate in enum(unseen):
            if _beats(candidate, lead):
                return False
    # The Phoenix follows a single by +0.5 (it cannot beat the Dragon) — a cross-type
    # case enumerate_singles does not surface as a beater. Mirrors `legal_actions`.
    if isinstance(lead, Single) and PHOENIX in unseen and lead.card is not DRAGON:
        return False
    return True


def guaranteed_out_chain(
    hand: frozenset[CardOrSpecial], unseen: frozenset[CardOrSpecial]
) -> tuple[Combination, ...] | None:
    """The uninterrupted-chain Guaranteed Out (depth-0): a sequence of plays,
    each combo except the last an Unbeatable Lead, that empties `hand` from the
    lead. Opponents never get the lead, so `unseen` is fixed across the chain.
    Returns the winning sequence, or None if no such chain exists.
    """
    for combo in _enumerate_all(hand):
        remaining = hand - frozenset(_cards_in(combo))
        if not remaining:
            # Last play: empties the Hand, so its beatability is irrelevant.
            return (combo,)
        # A non-final combo must keep the lead, i.e. be unbeatable, and the
        # rest of the Hand must itself chain out against the same Unseen Cards.
        if is_unbeatable_lead(combo, unseen):
            rest = guaranteed_out_chain(remaining, unseen)
            if rest is not None:
                return (combo,) + rest
    return None


def _has_unbeatable_lead(hand: frozenset[CardOrSpecial], unseen: frozenset[CardOrSpecial]) -> bool:
    """True iff some combination playable from `hand` is an Unbeatable Lead — i.e.
    a Re-entry is held in reserve to reclaim the lead after a shed."""
    return any(is_unbeatable_lead(c, unseen) for c in _enumerate_all(hand))


def reclaim_out(
    hand: frozenset[CardOrSpecial], unseen: frozenset[CardOrSpecial]
) -> tuple[Combination, ...] | None:
    """The **Reclaim Out** — a *probabilistic* regain-the-lead out (ADR-0032). Like the
    chain, but a non-final combo may be **beatable** (a shed) provided the remaining Hand
    still holds an Unbeatable Lead (a **Re-entry** — a bomb, a card-counted-unbeatable high
    single/pair, etc.) to win the lead back. Strictly generalises `guaranteed_out_chain`.

    *Not* a worst-case guarantee: a non-bomb reclaim assumes the opponent eventually leads
    into the held Re-entry (a worst-case opponent can refuse). Returns a plausible winning
    sequence, or None.
    """
    for combo in _enumerate_all(hand):
        remaining = hand - frozenset(_cards_in(combo))
        if not remaining:
            return (combo,)
        # Keep the lead (unbeatable), OR shed a beatable combo while a Re-entry remains.
        if is_unbeatable_lead(combo, unseen) or _has_unbeatable_lead(remaining, unseen):
            rest = reclaim_out(remaining, unseen)
            if rest is not None:
                return (combo,) + rest
    return None
