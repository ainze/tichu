"""Backward-compatible re-export of the card↔slot mapping.

The table moved to `tichu_engine.card_slots` at v7 (ADR-0044): the engine's own
Rich History Block accumulator is slot-indexed, and engine → training would have
inverted the layering (`card_slots` itself imports `tichu_engine.cards`).
Re-exported here so the many existing `tichu_training.card_slots` importers keep
working against the SAME table — one source of truth, not two.
"""

from tichu_engine.card_slots import (  # noqa: F401
    CARD_SLOTS,
    card_slot,
    slot_to_card,
)

__all__ = ["CARD_SLOTS", "card_slot", "slot_to_card"]
