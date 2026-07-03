"""H3 pre-check probe: remaining-cards-by-rank (card-counting summary).

Multi-hot over ranks still UNSEEN among opponents = full deck - own hand - cards
played this Round (by conservation, exactly the union of the three opponents'
current hands). Fully DERIVABLE from the v6 featurizer (own_hand + played_by),
so it fails the "reduce label noise the net can't see" filter on paper; this
block makes it explicit only so the residual pre-check can test whether an
explicit representation nonetheless helps a small classifier (ADR-0039 dropped
H3 pending exactly this cheap check).

Layout (17 dims): indices 0..12 = natural ranks 2..14; 13=Dragon, 14=Phoenix,
15=Mahjong, 16=Dog.
"""

import numpy as np

from tichu_engine.cards import DOG, DRAGON, MAHJONG, PHOENIX, SpecialCard
from tichu_engine.state import PrivateState
from tichu_training.card_slots import CARD_SLOTS, slot_to_card

REMAINING_BLOCK_DIM: int = 17

_SPECIAL_INDEX: dict[SpecialCard, int] = {
    DRAGON: 13, PHOENIX: 14, MAHJONG: 15, DOG: 16,
}
_FULL_DECK: tuple = tuple(slot_to_card(i) for i in range(CARD_SLOTS))


def _rank_block_index(card) -> int:
    if isinstance(card, SpecialCard):
        return _SPECIAL_INDEX[card]
    return card.rank - 2


def remaining_by_rank_block(private_state: PrivateState) -> np.ndarray:
    """Multi-hot (17,) over ranks/specials with ≥1 card unseen among opponents."""
    out = np.zeros(REMAINING_BLOCK_DIM, dtype=np.float32)
    seen = private_state.hand | private_state.public.played_cards_this_round
    for card in _FULL_DECK:
        if card not in seen:
            out[_rank_block_index(card)] = 1.0
    return out
