"""Full-strength Starting-Position Pool.

A Full-strength Starting Position is the pre-Schupfen, deal-time `GameState`
(`SchupfenPending`, 14-card hands) plus each seat's Grand-Tichu Prefix — the
first 8 cards in deal order. The prefix is a genuine subset of the 14 cards that
seat then Schupfens and Plays in the same Round; only the 8/6 split point is
fixed by convention (deck-deal order). See ADR-0025.

The Pool's identity is exactly `(seed, n)`: same inputs produce identical
positions and identical serialized files.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from tichu_engine.deck import fresh_deck
from tichu_engine.state import (
    INITIAL_HAND_SIZE,
    NUM_PLAYERS,
    GameState,
    PublicState,
    SchupfenPending,
    Trick,
    deal_for_schupfen,
)


_GRAND_PREFIX_SIZE = 8
_DECK = fresh_deck()
_CARD_TO_ID: dict = {card: i for i, card in enumerate(_DECK)}
_ID_TO_CARD: tuple = tuple(_DECK)


@dataclass(frozen=True)
class FullStartingPosition:
    state: GameState
    grand_prefixes: tuple[frozenset, ...]


def _ordered_hands(seed: int) -> list[list]:
    """Replicate `deal_for_schupfen`'s shuffle, preserving deal order so the
    Grand-Tichu Prefix is well-defined (a frozenset hand loses it)."""
    rng = random.Random(seed)
    deck = list(fresh_deck())
    rng.shuffle(deck)
    return [
        deck[p * INITIAL_HAND_SIZE : (p + 1) * INITIAL_HAND_SIZE]
        for p in range(NUM_PLAYERS)
    ]


def generate_full_position_pool(seed: int, n: int) -> list[FullStartingPosition]:
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    pool: list[FullStartingPosition] = []
    for i in range(n):
        s = seed + i
        ordered = _ordered_hands(s)
        prefixes = tuple(
            frozenset(hand[:_GRAND_PREFIX_SIZE]) for hand in ordered
        )
        pool.append(
            FullStartingPosition(state=deal_for_schupfen(seed=s), grand_prefixes=prefixes)
        )
    return pool


def save_full_position_pool(
    pool: Sequence[FullStartingPosition], path: Path
) -> None:
    """Persist each position's 14-card hands and 8-card Grand-Tichu Prefixes as
    card-id lists. Order within a list is irrelevant — the prefix is stored
    explicitly, so the 8/6 split survives without preserving deal order on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: dict = {"deal_index": pa.array(list(range(len(pool))), type=pa.int32())}
    for p in range(NUM_PLAYERS):
        columns[f"hand_p{p}"] = pa.array(
            [sorted(_CARD_TO_ID[c] for c in pos.state.hands[p]) for pos in pool],
            type=pa.list_(pa.int8()),
        )
        columns[f"prefix_p{p}"] = pa.array(
            [sorted(_CARD_TO_ID[c] for c in pos.grand_prefixes[p]) for pos in pool],
            type=pa.list_(pa.int8()),
        )
    pq.write_table(pa.table(columns), path)


def load_full_position_pool(path: Path) -> list[FullStartingPosition]:
    table = pq.read_table(Path(path))
    hands_by_player = [table.column(f"hand_p{p}").to_pylist() for p in range(NUM_PLAYERS)]
    prefixes_by_player = [
        table.column(f"prefix_p{p}").to_pylist() for p in range(NUM_PLAYERS)
    ]
    pool: list[FullStartingPosition] = []
    for row in range(table.num_rows):
        hands = tuple(
            frozenset(_ID_TO_CARD[i] for i in hands_by_player[p][row])
            for p in range(NUM_PLAYERS)
        )
        prefixes = tuple(
            frozenset(_ID_TO_CARD[i] for i in prefixes_by_player[p][row])
            for p in range(NUM_PLAYERS)
        )
        public = PublicState(
            current_player=0,
            hand_sizes=tuple(len(h) for h in hands),  # type: ignore[arg-type]
            scores=(0, 0),
            trick=Trick.empty(),
            pending_decision=SchupfenPending(submitted=(None, None, None, None)),
        )
        pool.append(
            FullStartingPosition(
                state=GameState(hands=hands, public=public), grand_prefixes=prefixes
            )
        )
    return pool
