"""Structured records produced by parsing one .tch file.

`ParsedAction` represents one decision recorded in the log — a play, pass,
schupfen submission, Mahjong-wish declaration, Dragon-give, or Tichu/Grand-
Tichu call. `ParsedRound` collects all decisions of one round plus the dealt
hands and the recorded `Ergebnis`. `ParsedGame` is the full multi-round game.

These records are deliberately closer to the BSW log than to the engine: a
later replay step converts them into engine actions and verifies legality and
score consistency.
"""

from dataclasses import dataclass, field

from tichu_engine.combinations import CardOrSpecial


@dataclass(frozen=True)
class ParsedAction:
    player: int
    kind: str  # 'play' | 'pass' | 'wish' | 'dragon_give' | 'tichu' | 'grand_tichu' | 'schupfen'
    cards: tuple[CardOrSpecial, ...] = ()
    wish_rank: int | None = None
    dragon_target: int | None = None
    schupfen_to_next: CardOrSpecial | None = None
    schupfen_to_partner: CardOrSpecial | None = None
    schupfen_to_previous: CardOrSpecial | None = None


@dataclass(frozen=True)
class ParsedRound:
    round_index: int
    pre_deal_hands: tuple[frozenset[CardOrSpecial], ...]  # 8 cards each
    start_hands: tuple[frozenset[CardOrSpecial], ...]     # 14 cards each
    grand_tichu_callers: frozenset[int]
    tichu_callers: frozenset[int]
    schupfen: tuple[ParsedAction, ...]
    plays: tuple[ParsedAction, ...]
    ergebnis: tuple[int, int]  # team 0 score, team 1 score for this round


@dataclass(frozen=True)
class ParsedGame:
    game_id: str | None
    handles: tuple[str, str, str, str]  # seat -> player handle
    rounds: tuple[ParsedRound, ...]
