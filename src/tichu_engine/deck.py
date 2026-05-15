from tichu_engine.cards import Card, DOG, DRAGON, MAHJONG, PHOENIX, Suit


def fresh_deck() -> tuple:
    """Return the canonical 56-card Tichu deck in a fixed deterministic order."""
    suited = [Card(suit=suit, rank=rank) for suit in Suit for rank in range(2, 15)]
    specials = [MAHJONG, DOG, PHOENIX, DRAGON]
    return tuple(suited + specials)
