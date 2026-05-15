"""Tokens used in BSW .tch logs.

A BSW card is a 2-or-3-character string: a one-letter suit (B/G/R/S) followed
by a rank (2..10, B, D, K, A) — or one of the special-card 2-letter codes
(Ma, Hu, Dr, Ph).
"""

from tichu_engine.cards import Card, DOG, DRAGON, MAHJONG, PHOENIX, Suit


# The suit mapping is arbitrary but stable: every BSW corpus must be parsed
# with the same mapping for hand equality to hold.
_SUIT_BY_LETTER: dict[str, Suit] = {
    "B": Suit.STAR,    # Blau (blue)
    "G": Suit.JADE,    # Grün (green)
    "R": Suit.PAGODA,  # Rot (red)
    "S": Suit.SWORD,   # Schwarz (black)
}

_RANK_BY_CODE: dict[str, int] = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "10": 10,
    "B": 11,  # Bube
    "D": 12,  # Dame
    "K": 13,  # König
    "A": 14,  # Ass
}

_SPECIALS = {
    "Ma": MAHJONG,
    "Hu": DOG,
    "Dr": DRAGON,
    "Ph": PHOENIX,
}


def parse_card_token(token: str):
    """Parse one BSW card token into a `Card` or special-card singleton."""
    if not token:
        raise ValueError("empty card token")
    if token in _SPECIALS:
        return _SPECIALS[token]
    suit_letter, rank_code = token[0], token[1:]
    if suit_letter not in _SUIT_BY_LETTER:
        raise ValueError(f"unknown card token: {token!r}")
    if rank_code not in _RANK_BY_CODE:
        raise ValueError(f"unknown card token: {token!r}")
    return Card(suit=_SUIT_BY_LETTER[suit_letter], rank=_RANK_BY_CODE[rank_code])
