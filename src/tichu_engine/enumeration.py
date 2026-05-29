from collections import defaultdict
from itertools import combinations, product

from tichu_engine.cards import Card, MAHJONG, PHOENIX, SpecialCard
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


def _group_by_rank(hand: frozenset[CardOrSpecial]) -> dict[int, list[Card]]:
    """Group normal cards by rank. Special cards are excluded — callers handle them separately."""
    groups: dict[int, list[Card]] = defaultdict(list)
    for card in hand:
        if isinstance(card, Card):
            groups[card.rank].append(card)
    return groups


def enumerate_singles(hand: frozenset[CardOrSpecial]) -> frozenset[Single]:
    """All Single combinations playable from `hand`. Special cards become Singles too."""
    return frozenset(Single(card) for card in hand)


def enumerate_pairs(hand: frozenset[CardOrSpecial]) -> frozenset[Pair]:
    """All Pair combinations playable from `hand`, including Phoenix substitutions."""
    pairs: set[Pair] = set()
    by_rank = _group_by_rank(hand)
    for cards_of_rank in by_rank.values():
        if len(cards_of_rank) >= 2:
            for a, b in combinations(cards_of_rank, 2):
                pairs.add(Pair(a, b))
    if PHOENIX in hand:
        for cards_of_rank in by_rank.values():
            for c in cards_of_rank:
                pairs.add(Pair(PHOENIX, c))
    return frozenset(pairs)


def enumerate_triples(hand: frozenset[CardOrSpecial]) -> frozenset[Triple]:
    """All Triple combinations playable from `hand`, including Phoenix substitutions."""
    triples: set[Triple] = set()
    by_rank = _group_by_rank(hand)
    for cards_of_rank in by_rank.values():
        if len(cards_of_rank) >= 3:
            for a, b, c in combinations(cards_of_rank, 3):
                triples.add(Triple(a, b, c))
    if PHOENIX in hand:
        for cards_of_rank in by_rank.values():
            if len(cards_of_rank) >= 2:
                for a, b in combinations(cards_of_rank, 2):
                    triples.add(Triple(PHOENIX, a, b))
    return frozenset(triples)


def enumerate_full_houses(hand: frozenset[CardOrSpecial]) -> frozenset[FullHouse]:
    """All FullHouse combinations playable from `hand`. Phoenix appears in at most one part."""
    triples = enumerate_triples(hand)
    pairs = enumerate_pairs(hand)
    full_houses: set[FullHouse] = set()
    for triple in triples:
        triple_has_phoenix = PHOENIX in (triple.a, triple.b, triple.c)
        for pair in pairs:
            if pair.rank == triple.rank:
                continue
            pair_has_phoenix = PHOENIX in (pair.a, pair.b)
            if triple_has_phoenix and pair_has_phoenix:
                continue
            full_houses.add(FullHouse(triple=triple, pair=pair))
    return frozenset(full_houses)


def enumerate_four_of_a_kind_bombs(hand: frozenset[CardOrSpecial]) -> frozenset[FourOfAKindBomb]:
    """All four-of-a-kind bombs playable from `hand`. Phoenix cannot appear in a bomb."""
    bombs: set[FourOfAKindBomb] = set()
    for cards_of_rank in _group_by_rank(hand).values():
        if len(cards_of_rank) == 4:
            bombs.add(FourOfAKindBomb(*cards_of_rank))
    return frozenset(bombs)


def enumerate_straight_flush_bombs(hand: frozenset[CardOrSpecial]) -> frozenset[StraightFlushBomb]:
    """All straight-flush bombs (>= 5 consecutive same-suit cards). Phoenix cannot appear."""
    by_suit: dict[object, list[Card]] = defaultdict(list)
    for card in hand:
        if isinstance(card, Card):
            by_suit[card.suit].append(card)
    bombs: set[StraightFlushBomb] = set()
    for cards_in_suit in by_suit.values():
        ranks = sorted({c.rank for c in cards_in_suit})
        if len(ranks) < 5:
            continue
        cards_by_rank = {c.rank: c for c in cards_in_suit}
        i = 0
        while i < len(ranks):
            j = i
            while j + 1 < len(ranks) and ranks[j + 1] == ranks[j] + 1:
                j += 1
            run = ranks[i : j + 1]
            if len(run) >= 5:
                for start_idx in range(len(run)):
                    for length in range(5, len(run) - start_idx + 1):
                        chosen = tuple(cards_by_rank[r] for r in run[start_idx : start_idx + length])
                        bombs.add(StraightFlushBomb(chosen))
            i = j + 1
    return frozenset(bombs)


_MIN_PAIR_STEP_PAIRS = 2


def enumerate_pair_steps(hand: frozenset[CardOrSpecial]) -> frozenset[PairStep]:
    """All PairStep combinations (>= 2 consecutive-rank pairs), including Phoenix-bridged steps.

    Phoenix may substitute for at most one pair in the step (it is one card).
    """
    # Group pairs by rank. Distinguish natural pairs (no Phoenix) from Phoenix-pairs.
    pairs_by_rank: dict[int, list[Pair]] = defaultdict(list)
    phoenix_pairs_by_rank: dict[int, list[Pair]] = defaultdict(list)
    for pair in enumerate_pairs(hand):
        if PHOENIX in (pair.a, pair.b):
            phoenix_pairs_by_rank[pair.rank].append(pair)
        else:
            pairs_by_rank[pair.rank].append(pair)

    # Ranks that have *any* pair option (natural or phoenix-bridged).
    all_pair_ranks = set(pairs_by_rank) | set(phoenix_pairs_by_rank)
    if not all_pair_ranks:
        return frozenset()

    pair_steps: set[PairStep] = set()
    ranks_sorted = sorted(all_pair_ranks)
    for start in ranks_sorted:
        end = start
        while end + 1 in all_pair_ranks:
            end += 1
        run = end - start + 1
        if run < _MIN_PAIR_STEP_PAIRS:
            continue
        for s in range(start, end + 1):
            for length in range(_MIN_PAIR_STEP_PAIRS, end - s + 2):
                run_ranks = list(range(s, s + length))
                # Build the choice set for each rank: natural pairs + Phoenix pairs.
                # But Phoenix can be used in at most ONE rank across the whole step.
                # Enumerate by choosing the phoenix-rank (or none), then natural pairs elsewhere.
                # Case 1: no Phoenix used.
                if all(pairs_by_rank.get(r) for r in run_ranks):
                    rank_choices = [pairs_by_rank[r] for r in run_ranks]
                    for picked in product(*rank_choices):
                        pair_steps.add(PairStep(picked))
                # Case 2: Phoenix used at exactly one rank in the run.
                for phoenix_rank in run_ranks:
                    if not phoenix_pairs_by_rank.get(phoenix_rank):
                        continue
                    other_ranks = [r for r in run_ranks if r != phoenix_rank]
                    if not all(pairs_by_rank.get(r) for r in other_ranks):
                        continue
                    rank_choices = []
                    for r in run_ranks:
                        if r == phoenix_rank:
                            rank_choices.append(phoenix_pairs_by_rank[r])
                        else:
                            rank_choices.append(pairs_by_rank[r])
                    for picked in product(*rank_choices):
                        pair_steps.add(PairStep(picked))
    return frozenset(pair_steps)


_MIN_STRAIGHT_LENGTH = 5


_MAHJONG_TUPLE: tuple = (MAHJONG,)
_EMPTY_TUPLE: tuple = ()
_PHOENIX_TUPLE: tuple = (PHOENIX,)

# Length-5 windows are the shortest legal straight; a hand that can't
# cover any one of them can't form a straight of any length. Start at
# rank 1 (Mahjong slot) through rank 10 (the 10..14 window).
_STRAIGHT_WINDOW_STARTS = range(1, 15 - _MIN_STRAIGHT_LENGTH + 1)


def _no_straight_possible(
    ranks_present: frozenset[int], has_phoenix: bool, has_mahjong: bool,
) -> bool:
    """True iff no length-5 rank window can be covered by `hand`.

    A window [start, start+5) is coverable when every rank in it is
    either backed by a real card, the Mahjong slot (rank 1), or fillable
    by the single Phoenix. The Phoenix covers at most one missing slot.

    This is an O(1)-window precheck (10 windows of 5) that lets the
    ~65% of decision hands with no straight bail before the nested
    window/product loop below — that loop costs ~30 µs/call building and
    discarding per-window candidates even when nothing is emittable. The
    precheck is exact: it returns False whenever the loop would emit at
    least one Straight, so it never skips real output. See the
    2026-05-29 enumerate-straights spike notes.
    """
    coverable = ranks_present | {1} if has_mahjong else ranks_present
    phoenix_budget = 1 if has_phoenix else 0
    for start in _STRAIGHT_WINDOW_STARTS:
        missing = 0
        for rank in range(start, start + _MIN_STRAIGHT_LENGTH):
            if rank not in coverable:
                missing += 1
                if missing > phoenix_budget:
                    break
        else:
            return False  # this window is coverable -> a straight exists
    return True


def enumerate_straights(hand: frozenset[CardOrSpecial]) -> frozenset[Straight]:
    """All Straight combinations (length >= 5), including Phoenix-bridged and
    Mahjong-led straights.

    Phoenix may substitute for at most one rank slot. Mahjong, when present,
    occupies rank 1 — the lowest slot of any straight that includes it.

    Constructs each Straight in canonical order at this site and uses
    `Straight._from_canonical_cards()` to bypass `__post_init__`
    validation. Validation reconstructed the canonical order on every
    call — that work is wasted here because we already produce it
    correctly. See ADR-0017 (pending) for the validation contract.
    """
    by_rank = _group_by_rank(hand)
    has_phoenix = PHOENIX in hand
    has_mahjong = MAHJONG in hand

    # Bail before the window/product loop when no length-5 window is
    # coverable — the common case for small mid-round hands. Exact: this
    # only short-circuits when the loop would emit nothing.
    if _no_straight_possible(frozenset(by_rank), has_phoenix, has_mahjong):
        return frozenset()

    straights: set[Straight] = set()

    # Windows starting at rank 1 are only valid when Mahjong is in hand
    # (Mahjong fills the rank-1 slot). Otherwise start at 2.
    min_start = 1 if has_mahjong else 2
    for start in range(min_start, 15):
        mahjong_in_window = start == 1
        for length in range(_MIN_STRAIGHT_LENGTH, 15 - start + 1):
            window_end = start + length  # exclusive
            # Ranks needing a real card. Excludes rank 1 (Mahjong slot).
            if mahjong_in_window:
                ranks_to_fill = list(range(2, window_end))
            else:
                ranks_to_fill = list(range(start, window_end))

            # Case 1: no Phoenix.
            # Canonical order: `picked` is already rank-sorted (since
            # ranks_to_fill is sorted ascending); prepend Mahjong if the
            # window starts at rank 1.
            if all(by_rank.get(r) for r in ranks_to_fill):
                rank_choices = [by_rank[r] for r in ranks_to_fill]
                if mahjong_in_window:
                    for picked in product(*rank_choices):
                        straights.add(
                            Straight._from_canonical_cards(
                                _MAHJONG_TUPLE + picked, None,
                            ),
                        )
                else:
                    for picked in product(*rank_choices):
                        straights.add(
                            Straight._from_canonical_cards(picked, None),
                        )
            # Case 2: Phoenix substitutes for one non-Mahjong rank.
            # Canonical order: real cards (sorted), with Phoenix spliced
            # at the slot for `phoenix_rank`, then Mahjong prepended if
            # the window starts at rank 1.
            if has_phoenix:
                # The first rank in `ranks_to_fill` is `start` (or 2 if
                # Mahjong is present) — so the index of `phoenix_rank`
                # in the rank-sorted result is (phoenix_rank - ranks_to_fill[0]).
                base_rank = 2 if mahjong_in_window else start
                for phoenix_rank in ranks_to_fill:
                    other_ranks = [r for r in ranks_to_fill if r != phoenix_rank]
                    if not all(by_rank.get(r) for r in other_ranks):
                        continue
                    rank_choices = [by_rank[r] for r in other_ranks]
                    phoenix_idx = phoenix_rank - base_rank  # slot in 'picked' insertion
                    if mahjong_in_window:
                        for picked in product(*rank_choices):
                            cards = (
                                _MAHJONG_TUPLE
                                + picked[:phoenix_idx]
                                + _PHOENIX_TUPLE
                                + picked[phoenix_idx:]
                            )
                            straights.add(
                                Straight._from_canonical_cards(
                                    cards, phoenix_rank,
                                ),
                            )
                    else:
                        for picked in product(*rank_choices):
                            cards = (
                                picked[:phoenix_idx]
                                + _PHOENIX_TUPLE
                                + picked[phoenix_idx:]
                            )
                            straights.add(
                                Straight._from_canonical_cards(
                                    cards, phoenix_rank,
                                ),
                            )
    return frozenset(straights)
