r"""Microbench for tichu_engine.enumeration hot paths.

Background: py-spy on a parse_bsw worker (2026-05-29) showed
`enumerate_straights` consuming ~10% of worker CPU across 7 distinct
lines; `_enumerate_all` (the combined enumeration entry point in
legality.py) was ~21% combined. Smaller frame contributors: hash /
equality on the constructed combination objects (~14%), pair-step and
full-house enumeration (~3% combined).

This script characterises the per-call cost of each enumerator on a
small set of representative hands, so we can prioritise optimisations
by measured ms rather than guessed top-of-stack samples.

Usage (PowerShell):

  $env:PYTHONPATH = (Resolve-Path src).Path
  py -3.14 scripts\bench_enumeration.py --iters 5000
"""

from __future__ import annotations

import argparse
import time

from tichu_engine.cards import Card, MAHJONG, PHOENIX, Suit
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
from tichu_engine.legality import _enumerate_all


_SUITS = (Suit.JADE, Suit.SWORD, Suit.PAGODA, Suit.STAR)


def _hand_full_ladder_no_specials() -> frozenset:
    """All naturals 2..14 in JADE plus an extra rank-2 in SWORD.

    14 cards, one of every rank 2..14, plus one duplicate so pairs +
    full_houses have something to chew. Excludes Mahjong/Phoenix so
    enumerate_straights is exercised in the "no phoenix" branch only.
    """
    cards: set = set()
    for r in range(2, 15):
        cards.add(Card(Suit.JADE, r))
    cards.add(Card(Suit.SWORD, 2))
    return frozenset(cards)


def _hand_full_ladder_with_phoenix() -> frozenset:
    """Same as above but swap the duplicate for PHOENIX.

    Triggers the "phoenix substitution" branches in every enumerator.
    Realistic worst-case for `enumerate_straights` (every window has
    both a no-phoenix and a phoenix-substituted variant).
    """
    cards: set = {Card(Suit.JADE, r) for r in range(2, 15)}
    cards.add(PHOENIX)
    return frozenset(cards)


def _hand_full_ladder_with_mahjong_and_phoenix() -> frozenset:
    """Ranks 2..14 in JADE + MAHJONG + PHOENIX.

    Triggers the Mahjong-led straight branch on top of phoenix
    substitution. The most expensive hand for `enumerate_straights`.
    """
    cards: set = {Card(Suit.JADE, r) for r in range(2, 15)}
    cards.add(MAHJONG)
    cards.add(PHOENIX)
    return frozenset(cards)


def _hand_dense_suits() -> frozenset:
    """Ranks 3..12 in all 4 suits (40 cards). Massive product enumeration.

    Not realistic for a single Tichu hand (max 14 cards) but exercises
    the worst case for the suit-product loop inside enumerate_straights
    — `product(*rank_choices)` blows up when many ranks have 4 cards.
    Useful as an "upper bound" stress test.
    """
    cards: set = set()
    for r in range(3, 13):
        for s in _SUITS:
            cards.add(Card(s, r))
    return frozenset(cards)


def _hand_realistic_mid_round() -> frozenset:
    """A plausible mid-round hand: 7 mixed cards across a few suits.

    Closer to what most calls actually look like during replay. Workers
    spend most of their time on hands like this, not the 14-card deals.
    """
    cards: set = {
        Card(Suit.JADE, 3), Card(Suit.SWORD, 3),
        Card(Suit.JADE, 5),
        Card(Suit.PAGODA, 7), Card(Suit.STAR, 7),
        Card(Suit.JADE, 10),
        Card(Suit.SWORD, 13),
    }
    return frozenset(cards)


_HANDS: dict[str, frozenset] = {
    "ladder_naturals_dup2": _hand_full_ladder_no_specials(),
    "ladder_with_phoenix": _hand_full_ladder_with_phoenix(),
    "ladder_phoenix_mahjong": _hand_full_ladder_with_mahjong_and_phoenix(),
    "dense_suits_3_to_12": _hand_dense_suits(),
    "mid_round_7_cards": _hand_realistic_mid_round(),
}


_ENUMERATORS = [
    ("enumerate_singles", enumerate_singles),
    ("enumerate_pairs", enumerate_pairs),
    ("enumerate_triples", enumerate_triples),
    ("enumerate_full_houses", enumerate_full_houses),
    ("enumerate_straights", enumerate_straights),
    ("enumerate_pair_steps", enumerate_pair_steps),
    ("enumerate_four_of_a_kind_bombs", enumerate_four_of_a_kind_bombs),
    ("enumerate_straight_flush_bombs", enumerate_straight_flush_bombs),
    ("_enumerate_all", _enumerate_all),
]


def _bench(name: str, fn, hand: frozenset, iters: int) -> tuple[float, int]:
    # Warm up + JIT-friendly: one call to amortise the dispatch cost.
    fn(hand)
    t0 = time.perf_counter()
    for _ in range(iters):
        result = fn(hand)
    elapsed = time.perf_counter() - t0
    return elapsed, len(result)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=2000,
                   help="Iterations per (hand, enumerator) cell. "
                        "Default 2000.")
    p.add_argument("--hand", choices=sorted(_HANDS), default=None,
                   help="Bench only one named hand. Default: all.")
    p.add_argument("--enumerator", choices=[name for name, _ in _ENUMERATORS],
                   default=None,
                   help="Bench only one enumerator. Default: all.")
    args = p.parse_args()

    hands = {args.hand: _HANDS[args.hand]} if args.hand else _HANDS
    enumerators = (
        [(n, f) for n, f in _ENUMERATORS if n == args.enumerator]
        if args.enumerator else _ENUMERATORS
    )

    print(f"iters per cell: {args.iters}\n")
    header = f"{'enumerator':<32}" + "".join(f"{h:>22}" for h in hands)
    print(header)
    print("-" * len(header))
    for name, fn in enumerators:
        cells: list[str] = []
        for hand in hands.values():
            elapsed, n_results = _bench(name, fn, hand, args.iters)
            us_per_call = elapsed * 1e6 / args.iters
            cells.append(f"{us_per_call:>8.1f}us /{n_results:>5}")
        print(f"{name:<32}" + "".join(f"{c:>22}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
