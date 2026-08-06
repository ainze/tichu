"""Public card↔slot mapping — torch-free, 56-slot.

Layout: 52 naturals first (suit-major: JADE/SWORD/PAGODA/STAR × ranks 2..14),
then 4 specials (MAHJONG=52, DOG=53, PHOENIX=54, DRAGON=55).

The featurizer's `own_hand` and `seen_cards` sections already use this exact
mapping via a private `_card_slot`; this module is the public, importable
single source of truth so downstream code (the Schupfen Network's decoder,
in particular) can both encode AND decode without duplicating the table.

Lives outside `featurizer.py` so torch-free importers (parse_bsw, the
schupfen training adapter, the Schupfen Network at inference time) can
pull it without dragging the featurizer module into their import graph.
Mirrors the `decision_types.py` split rationale.
"""

from tichu_engine.cards import Card, DOG, DRAGON, MAHJONG, PHOENIX, SpecialCard, Suit


CARD_SLOTS: int = 56

_SUIT_ORDER = (Suit.JADE, Suit.SWORD, Suit.PAGODA, Suit.STAR)

_NATURAL_SLOT_BY_CARD: dict[Card, int] = {}
for _suit_idx, _suit in enumerate(_SUIT_ORDER):
    for _rank in range(2, 15):
        _NATURAL_SLOT_BY_CARD[Card(_suit, _rank)] = _suit_idx * 13 + (_rank - 2)

_SPECIAL_SLOT: dict[SpecialCard, int] = {
    MAHJONG: 52,
    DOG: 53,
    PHOENIX: 54,
    DRAGON: 55,
}

_CARD_BY_SLOT: list[object] = [None] * CARD_SLOTS
for _c, _s in _NATURAL_SLOT_BY_CARD.items():
    _CARD_BY_SLOT[_s] = _c
for _c, _s in _SPECIAL_SLOT.items():
    _CARD_BY_SLOT[_s] = _c


def card_slot(card: object) -> int:
    if isinstance(card, Card):
        return _NATURAL_SLOT_BY_CARD[card]
    if isinstance(card, SpecialCard):
        return _SPECIAL_SLOT[card]
    raise TypeError(f"unsupported card type: {type(card).__name__}")


def slot_to_card(slot: int) -> object:
    if not 0 <= slot < CARD_SLOTS:
        raise ValueError(f"slot must be in [0, {CARD_SLOTS}), got {slot}")
    return _CARD_BY_SLOT[slot]
