from collections import defaultdict
from itertools import combinations, product

from tichu_engine.cards import Card, PHOENIX, SpecialCard
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


def enumerate_straights(hand: frozenset[CardOrSpecial]) -> frozenset[Straight]:
    """All Straight combinations (length >= 5), including Phoenix-bridged straights.

    Phoenix may substitute for at most one rank slot.
    """
    by_rank = _group_by_rank(hand)
    has_phoenix = PHOENIX in hand
    straights: set[Straight] = set()

    # Enumerate every (start_rank, length) window of length >= 5 across ranks 2..14.
    for start in range(2, 15):
        for length in range(_MIN_STRAIGHT_LENGTH, 15 - start + 1):
            window = list(range(start, start + length))
            # Case 1: no Phoenix — every rank in window must have at least one card.
            if all(by_rank.get(r) for r in window):
                rank_choices = [by_rank[r] for r in window]
                for picked in product(*rank_choices):
                    straights.add(Straight(picked))
            # Case 2: Phoenix substitutes for exactly one rank in the window.
            if has_phoenix:
                for phoenix_rank in window:
                    # Phoenix takes this slot; cannot also be a natural card of that rank in the straight.
                    other_ranks = [r for r in window if r != phoenix_rank]
                    if not all(by_rank.get(r) for r in other_ranks):
                        continue
                    rank_choices = [by_rank[r] for r in other_ranks]
                    for picked in product(*rank_choices):
                        cards = tuple(list(picked) + [PHOENIX])
                        straights.add(Straight(cards, phoenix_as_rank=phoenix_rank))
    return frozenset(straights)
