"""Public card↔slot mapping (Q2 of the schupfen design pass).

The Schupfen Network outputs 56-way logits per direction; it needs the
inverse `slot_to_card` to decode predicted slots back to a `SchupfenPass`.
The featurizer already encodes hands into the same 56-slot layout via a
private `_card_slot` — these tests pin that the public module is exactly
the same mapping (no second hand-rolled table to drift).
"""

import pytest

from tichu_engine.cards import Card, DOG, DRAGON, MAHJONG, PHOENIX, Suit
from tichu_training import card_slots
from tichu_training.featurizer import _card_slot as _featurizer_card_slot


_ALL_NATURALS = [Card(s, r) for s in Suit for r in range(2, 15)]
_ALL_SPECIALS = [MAHJONG, DOG, PHOENIX, DRAGON]
_ALL_CARDS = _ALL_NATURALS + _ALL_SPECIALS


def test_card_slots_constant_is_56():
    assert card_slots.CARD_SLOTS == 56


def test_card_slot_round_trips_for_every_card():
    for c in _ALL_CARDS:
        slot = card_slots.card_slot(c)
        assert 0 <= slot < 56
        assert card_slots.slot_to_card(slot) == c


def test_card_slot_matches_featurizer_ordering():
    """The public mapping MUST be identical to the featurizer's private
    one. Drift here would silently mis-train every checkpoint."""
    for c in _ALL_CARDS:
        assert card_slots.card_slot(c) == _featurizer_card_slot(c)


def test_slots_cover_full_range_without_collision():
    seen = {card_slots.card_slot(c) for c in _ALL_CARDS}
    assert len(seen) == 56
    assert seen == set(range(56))
