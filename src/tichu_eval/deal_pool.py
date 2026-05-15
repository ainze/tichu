"""Fixed seeded deal pool.

A "deal" is the post-schupfen starting `GameState` for one Tichu round:
all 56 cards dealt 14-per-player, current player set to the Mahjong holder,
no points or trick state yet. Schupfen is intentionally skipped — the
harness measures play strength, not schupfen heuristics.

The pool's identity is exactly `(seed, n)`: same inputs across two
processes produce identical hands, identical starting players, and
identical serialized files.
"""

from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from tichu_engine.deck import fresh_deck
from tichu_engine.state import GameState, PublicState, Trick, deal_initial_state


_DECK = fresh_deck()
_CARD_TO_ID: dict = {card: i for i, card in enumerate(_DECK)}
_ID_TO_CARD: tuple = tuple(_DECK)


def generate_deal_pool(seed: int, n: int) -> list[GameState]:
    """Deterministic post-schupfen starting deals.

    Each deal `i` is `deal_initial_state(seed + i)`. Same `(seed, n)` ⇒
    identical pool across runs.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return [deal_initial_state(seed + i) for i in range(n)]


def save_deal_pool(deals: Sequence[GameState], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deal_indices = []
    starters = []
    hands_columns: list[list[list[int]]] = [[], [], [], []]
    for idx, deal in enumerate(deals):
        deal_indices.append(idx)
        starters.append(deal.public.current_player)
        for p in range(4):
            ids = sorted(_CARD_TO_ID[c] for c in deal.hands[p])
            hands_columns[p].append(ids)
    table = pa.table({
        "deal_index": pa.array(deal_indices, type=pa.int32()),
        "starting_player": pa.array(starters, type=pa.int32()),
        "hand_p0": pa.array(hands_columns[0], type=pa.list_(pa.int8())),
        "hand_p1": pa.array(hands_columns[1], type=pa.list_(pa.int8())),
        "hand_p2": pa.array(hands_columns[2], type=pa.list_(pa.int8())),
        "hand_p3": pa.array(hands_columns[3], type=pa.list_(pa.int8())),
    })
    pq.write_table(table, path)


def load_deal_pool(path: Path) -> list[GameState]:
    table = pq.read_table(Path(path))
    starters = table.column("starting_player").to_pylist()
    hands_by_player = [
        table.column(f"hand_p{p}").to_pylist() for p in range(4)
    ]
    deals: list[GameState] = []
    for row in range(table.num_rows):
        hands = tuple(
            frozenset(_ID_TO_CARD[i] for i in hands_by_player[p][row])
            for p in range(4)
        )
        public = PublicState(
            current_player=starters[row],
            hand_sizes=tuple(len(h) for h in hands),  # type: ignore[arg-type]
            scores=(0, 0),
            trick=Trick.empty(),
        )
        deals.append(GameState(hands=hands, public=public))
    return deals
