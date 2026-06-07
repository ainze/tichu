"""Legal-action enumeration for Tichu.

`legal_actions(state)` returns the set of actions the current player may take
in the given state — every playable combination from their hand, plus PASS when
they are following another player's combination.

Type-and-rank legality is encoded by the existing combination __gt__ operators.
Bombs interrupt that ordering: any four-of-a-kind bomb beats any non-bomb;
straight-flush bombs beat any four-of-a-kind bomb; straight-flushes compare to
each other by length then rank.
"""

import functools
from dataclasses import dataclass
from typing import Union

from tichu_engine.cards import Card, DRAGON, PHOENIX
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
    enumerate_full_houses,
    enumerate_pair_steps,
    enumerate_pairs,
    enumerate_singles,
    enumerate_straight_flush_bombs,
    enumerate_straights,
    enumerate_triples,
)
from tichu_engine.state import (
    NUM_PLAYERS,
    DragonGivePending,
    GameState,
    MahjongWishPending,
    PrivateState,
    SchupfenPending,
)


@dataclass(frozen=True)
class Pass:
    """The pass action. All Pass() instances compare equal."""


PASS = Pass()


@dataclass(frozen=True)
class DragonGive:
    """The winner of a Dragon-led trick chooses which opponent receives it."""

    target: int


@dataclass(frozen=True)
class MahjongWish:
    """The Mahjong-playing player declares a wish rank, or None to decline."""

    rank: int | None


@dataclass(frozen=True)
class BombInterrupt:
    """A non-current player plays a bomb to interrupt the current trick."""

    player: int
    bomb: object  # FourOfAKindBomb | StraightFlushBomb — kept opaque to avoid import cycles


@dataclass(frozen=True)
class SchupfenPass:
    """One player's schupfen submission: one card to each of the three other players.

    Slots are relative to the actor's seat:
      to_next:     (self + 1) % 4
      to_partner:  (self + 2) % 4
      to_previous: (self + 3) % 4
    """

    to_next: object
    to_partner: object
    to_previous: object


Combination = Union[
    Single, Pair, Triple, FullHouse, Straight, PairStep, FourOfAKindBomb, StraightFlushBomb
]
ConcreteAction = Union[Combination, Pass, DragonGive, MahjongWish, BombInterrupt, SchupfenPass]


@functools.lru_cache(maxsize=4096)
def _enumerate_all(hand) -> frozenset[Combination]:
    """Every combination playable from `hand`, across all combination types.

    Memoised on `hand` (a hashable `frozenset`). Enumeration is a pure function
    of the hand, so the cache never needs invalidation. This collapses the
    dominant redundancy on the hot path: every Play Decision enumerates the same
    hand twice — once to build the policy/BC legal mask (`legal_actions_for`) and
    again inside `step`'s `action not in legal_actions(state)` validation — plus
    re-enumerating an unchanged hand on every Pass. On RuleAgent self-play ~50% of
    calls are the immediate policy→step repeat and ~69% are cache hits overall
    (only ~31% of hands are distinct), yielding ~1.8x on the engine-bound rollout.
    The returned frozenset is immutable and every caller copies it, so sharing the
    cached value across callers is safe. The cache is per-process (the cotrain
    rollout fans across spawn workers); maxsize bounds it to the live-game working
    set with headroom. See docs/notes/2026-06-07-legal-actions-memo.md."""
    return frozenset[Combination](
        set(enumerate_singles(hand))
        | set(enumerate_pairs(hand))
        | set(enumerate_triples(hand))
        | set(enumerate_full_houses(hand))
        | set(enumerate_straights(hand))
        | set(enumerate_pair_steps(hand))
        | set(enumerate_four_of_a_kind_bombs(hand))
        | set(enumerate_straight_flush_bombs(hand))
    )


def _cards_in(combo: Combination) -> tuple:
    """All constituent cards in a combination, flattened. Phoenix substitutions
    appear as the PHOENIX SpecialCard, not as the rank they substitute for."""
    if isinstance(combo, Single):
        return (combo.card,)
    if isinstance(combo, Pair):
        return (combo.a, combo.b)
    if isinstance(combo, Triple):
        return (combo.a, combo.b, combo.c)
    if isinstance(combo, FullHouse):
        return _cards_in(combo.triple) + _cards_in(combo.pair)
    if isinstance(combo, Straight):
        return combo.cards
    if isinstance(combo, PairStep):
        out: tuple = ()
        for p in combo.pairs:
            out = out + _cards_in(p)
        return out
    if isinstance(combo, FourOfAKindBomb):
        return (combo.a, combo.b, combo.c, combo.d)
    if isinstance(combo, StraightFlushBomb):
        return combo.cards
    raise AssertionError(f"unknown combination type: {type(combo).__name__}")


def _fulfills_wish(combo: Combination, wish_rank: int) -> bool:
    """True iff the combination contains a natural card of the wished rank.
    Phoenix substituting for the wished rank does not count."""
    return any(isinstance(c, Card) and c.rank == wish_rank for c in _cards_in(combo))


def _beats(candidate: Combination, top: Combination) -> bool:
    """Does `candidate` legally beat `top` in a trick?

    Bomb rules:
      - StraightFlushBomb beats any non-StraightFlushBomb (incl. four-of-a-kind bombs).
      - FourOfAKindBomb beats any non-bomb; loses to all StraightFlushBombs.
      - Same-type bombs compare by their natural ordering.
    Otherwise, same-type same-length comparison via the combination's __gt__.
    """
    # Same type: use the type's own ordering.
    if type(candidate) is type(top):
        try:
            return candidate > top  # type: ignore[operator]
        except TypeError:
            return False
    # Bombs interrupt the normal type-equality rule.
    if isinstance(candidate, StraightFlushBomb):
        # Beats any other type.
        return True
    if isinstance(candidate, FourOfAKindBomb):
        # Beats any non-bomb; loses to straight-flush bombs (handled above by type check).
        return not isinstance(top, StraightFlushBomb)
    return False


Bomb = Union[FourOfAKindBomb, StraightFlushBomb]


def legal_bomb_interrupts(state: GameState, player: int) -> frozenset[Bomb]:
    """Bombs that `player` may play out-of-turn to interrupt the current trick.

    Returns empty when `player` is the current player (their bombs are part
    of legal_actions instead) or when no bomb in their hand beats the current
    top. When the trick is empty (previous trick just resolved, current
    player hasn't led yet), any bomb in the player's hand is a legal preempt
    — BSW logs treat preempt-bombs by non-current players as bomb interrupts
    that seize the lead.
    """
    if player == state.public.current_player:
        return frozenset()
    hand = state.hands[player]
    bombs: set[Bomb] = set(enumerate_four_of_a_kind_bombs(hand)) | set(enumerate_straight_flush_bombs(hand))
    top = state.public.trick.top_combination
    if top is None:
        return frozenset(bombs)
    return frozenset(b for b in bombs if _beats(b, top))  # type: ignore[arg-type]


def legal_actions(state: GameState) -> frozenset[ConcreteAction]:
    """Legal actions for the current player in `state`.

    When leading (no top combination), every enumerated combination is legal and
    PASS is not. When following, only combinations that beat the top combination
    are legal, plus PASS.

    If a pending decision is set (Dragon-give or Mahjong-wish), the only legal
    actions are the decision options for that pending state.
    """
    pending = state.public.pending_decision
    if isinstance(pending, DragonGivePending):
        # Winner gives the trick to one of the two opponents.
        opponents = {p for p in range(4) if p % 2 != pending.winner % 2}
        return frozenset[ConcreteAction]({DragonGive(target=t) for t in opponents})
    if isinstance(pending, MahjongWishPending):
        return frozenset[ConcreteAction](
            {MahjongWish(rank=r) for r in range(2, 15)} | {MahjongWish(rank=None)}
        )
    if isinstance(pending, SchupfenPending):
        # Every permutation of 3 distinct cards from hand is a valid schupfen pass.
        hand = list(state.hands[state.public.current_player])
        passes: set[ConcreteAction] = set()
        for a in hand:
            for b in hand:
                if b == a:
                    continue
                for c in hand:
                    if c == a or c == b:
                        continue
                    passes.add(SchupfenPass(to_next=a, to_partner=b, to_previous=c))
        return frozenset(passes)

    hand = state.hands[state.public.current_player]
    all_combos = _enumerate_all(hand)
    top = state.public.trick.top_combination
    wish = state.public.mahjong_wish

    if top is None:
        leading: set[ConcreteAction] = set(all_combos)  # type: ignore[arg-type]
        return _apply_wish(leading, wish, can_pass=False)

    actions: set[ConcreteAction] = {c for c in all_combos if _beats(c, top)}  # type: ignore[arg-type]
    if isinstance(top, Single):
        # The leading-rank Phoenix single (rank 1.5) is only valid when leading.
        # When following, Phoenix must declare its following rank explicitly.
        actions = {
            a for a in actions
            if not (isinstance(a, Single) and a.card is PHOENIX and a.as_rank is None)
        }
        if PHOENIX in hand and top.card is not DRAGON:
            actions.add(Single.phoenix_following(top_rank=top.rank))
    return _apply_wish(actions, wish, can_pass=True)


def legal_actions_for(private_state: PrivateState) -> frozenset[ConcreteAction]:
    """Legal actions for the player owning `private_state`.

    `legal_actions` requires the full `GameState`, but an `Agent` only sees its
    own `PrivateState`. For the player's own turn, the legal-action set depends
    only on their hand and the public state, so this helper synthesises a
    placeholder `GameState` to reuse the same logic.
    """
    hands: list = [frozenset() for _ in range(NUM_PLAYERS)]
    hands[private_state.player] = private_state.hand
    synthetic = GameState(hands=tuple(hands), public=private_state.public)
    return legal_actions(synthetic)


def _apply_wish(actions: set[ConcreteAction], wish: int | None, *, can_pass: bool) -> frozenset[ConcreteAction]:
    """If a Mahjong wish is active and any action fulfills it, restrict to those
    actions (Pass is also forbidden). Otherwise, return actions plus Pass when allowed."""
    if wish is not None:
        wish_actions = {
            a for a in actions
            if not isinstance(a, Pass) and _fulfills_wish(a, wish)  # type: ignore[arg-type]
        }
        if wish_actions:
            return frozenset(wish_actions)
    if can_pass:
        actions = set(actions) | {PASS}
    return frozenset(actions)
