"""BSW .tch parser.

Converts the raw text of a BSW Tichu log into a `ParsedGame` containing one
`ParsedRound` per round. Each round records the dealt hands, schupfen
submissions, Tichu calls, the ordered sequence of play/pass/wish/dragon-give
actions, and the recorded `Ergebnis`.

The parser is permissive on whitespace and tolerant of the `BOMBE:` cosmetic
lines BSW emits between schupfen and play. It does NOT validate against engine
legality — that's done by the replay step downstream.

Format reference (lines from sample/2417500.tch):
  ---------------Gr.Tichukarten------------------
  (N)handle <8 cards>
  ---------------Startkarten------------------
  (N)handle <14 cards>
  Grosses Tichu: (N)handle        -- optional, repeatable
  Schupfen:
  (N)handle gibt: handle: card - handle: card - handle: card -
  BOMBE: (N)handle ...             -- optional, cosmetic
  ---------------Rundenverlauf------------------
  (N)handle: card card ...         -- play
  (N)handle passt.                 -- pass
  Wunsch:X                         -- mahjong-wish following a Mahjong play
  Tichu: (N)handle                 -- mid-round tichu call
  Drache an: (N)handle             -- dragon-give recipient
  Ergebnis: <team0> - <team1>
"""

import re
from dataclasses import replace

from tichu_engine.cards import Card, DRAGON, MAHJONG, PHOENIX
from tichu_engine.combinations import CardOrSpecial
from tichu_training.bsw.records import ParsedAction, ParsedGame, ParsedRound
from tichu_training.bsw.tokens import _RANK_BY_CODE, parse_card_token


# Handles can contain U+00A0 (NO-BREAK SPACE) as a literal in-handle space
# (BSW logs "Lucky Luke" as "Lucky\xa0Luke") or as cosmetic padding before a
# delimiter ("schubsi\xa0:"). Python's `\S` excludes \xa0 (Unicode whitespace),
# so we match handles as "anything that isn't ASCII space/tab/colon" and
# normalise the capture with `_clean_handle` before use.
_HANDLE = r"[^ \t:]*"

_PLAYER_LINE_RE = re.compile(rf"^\((\d)\)({_HANDLE})\s*(.*)$")
_GROSSES_TICHU_RE = re.compile(rf"^Grosses Tichu:\s*\((\d)\)({_HANDLE})\s*$")
_TICHU_RE = re.compile(rf"^Tichu:\s*\((\d)\)({_HANDLE})\s*$")
_DRAGON_RE = re.compile(rf"^Drache an:\s*\((\d)\)({_HANDLE})\s*$")
_WUNSCH_RE = re.compile(r"^Wunsch:\s*(\S+)\s*$")
_ERGEBNIS_RE = re.compile(r"^Ergebnis:\s*(-?\d+)\s*-\s*(-?\d+)\s*$")
_PASST_RE = re.compile(rf"^\((\d)\)({_HANDLE})\s+passt\.\s*$")
_SCHUPFEN_GIBT_RE = re.compile(rf"^\((\d)\)({_HANDLE})\s+gibt:\s*(.*)$")


def _clean_handle(raw: str) -> str:
    """Normalise a handle captured from BSW log text.

    BSW encodes a literal space inside a handle as `\\xa0` (NBSP) so the
    space doesn't break the line's whitespace-delimited grammar. Map it back
    to a real space for downstream skill_decile lookup, then strip cosmetic
    trailing NBSPs that BSW sometimes pads with.
    """
    return raw.replace("\xa0", " ").strip()


class _RoundBuilder:
    """Mutable accumulator for one round; finalises to an immutable ParsedRound."""

    def __init__(self, round_index: int) -> None:
        self.round_index = round_index
        self.handles: list[str | None] = [None, None, None, None]
        self.pre_deal: list[frozenset[CardOrSpecial]] = [frozenset()] * 4
        self.start: list[frozenset[CardOrSpecial]] = [frozenset()] * 4
        self.grand_tichu_callers: set[int] = set()
        self.tichu_callers: set[int] = set()
        self.schupfen: list[ParsedAction | None] = [None, None, None, None]
        self.plays: list[ParsedAction] = []
        self.ergebnis: tuple[int, int] | None = None
        # Tracking for context-dependent actions:
        self._last_player: int | None = None  # player of most recent play/pass

    def finalise(self) -> ParsedRound:
        if self.ergebnis is None:
            raise ValueError(f"round {self.round_index}: no Ergebnis line found")
        for seat in range(4):
            if self.schupfen[seat] is None:
                raise ValueError(f"round {self.round_index}: missing schupfen for seat {seat}")
        return ParsedRound(
            round_index=self.round_index,
            pre_deal_hands=tuple(self.pre_deal),
            start_hands=tuple(self.start),
            grand_tichu_callers=frozenset(self.grand_tichu_callers),
            tichu_callers=frozenset(self.tichu_callers),
            schupfen=tuple(self.schupfen),  # type: ignore[arg-type]
            plays=tuple(self.plays),
            ergebnis=self.ergebnis,
            handles=tuple(h or "" for h in self.handles),  # type: ignore[arg-type]
        )


def parse_tch(text: str, *, game_id: str | None = None) -> ParsedGame:
    lines = text.splitlines()
    rounds: list[ParsedRound] = []
    i = 0
    while i < len(lines):
        if "Gr.Tichukarten" in lines[i]:
            built, i = _parse_round(lines, i, round_index=len(rounds))
            rounds.append(built.finalise())
        else:
            i += 1
    if not rounds:
        raise ValueError("no rounds found in log")
    return ParsedGame(game_id=game_id, rounds=tuple(rounds))


def _parse_round(lines: list[str], start: int, *, round_index: int) -> tuple[_RoundBuilder, int]:
    """Parse one round starting at `lines[start]` (a Gr.Tichukarten header).

    Returns the populated builder and the index of the first line after the
    round's `Ergebnis:` line.

    A small fraction of BSW logs (~0.6 %) are truncated mid-round — the dump
    cuts off before the final `Ergebnis:` line. Any IndexError raised while
    indexing past the end of `lines` is normalised to a single
    `"round N: truncated log"` ValueError so the caller doesn't have to
    distinguish bounds-overrun from the explicit missing-Ergebnis check.
    """
    try:
        return _parse_round_body(lines, start, round_index=round_index)
    except IndexError:
        raise ValueError(f"round {round_index}: truncated log") from None


def _parse_round_body(lines: list[str], start: int, *, round_index: int) -> tuple[_RoundBuilder, int]:
    builder = _RoundBuilder(round_index)
    i = start + 1  # skip Gr.Tichukarten header

    # 8-card pre-deal hands
    for seat in range(4):
        i = _skip_blank(lines, i)
        handle, cards, i = _parse_player_card_line(lines, i)
        builder.handles[seat] = handle
        builder.pre_deal[seat] = frozenset(cards)

    # Startkarten section
    while i < len(lines) and "Startkarten" not in lines[i]:
        i += 1
    if i >= len(lines):
        raise ValueError(f"round {round_index}: missing Startkarten header")
    i += 1

    # 14-card start hands
    for seat in range(4):
        i = _skip_blank(lines, i)
        handle, cards, i = _parse_player_card_line(lines, i)
        if builder.handles[seat] != handle:
            # Sanity: handle must match across pre-deal and start hands for the same seat.
            raise ValueError(
                f"round {round_index}: seat {seat} handle mismatch "
                f"{builder.handles[seat]!r} vs {handle!r}"
            )
        builder.start[seat] = frozenset(cards)

    # Optional Grosses Tichu / Tichu calls, then the Schupfen header.
    # A regular `Tichu:` call is legal here — a player who's just seen their
    # 14-card hand may declare Tichu before passing. We surface both call
    # kinds as ParsedActions in `builder.plays` (in addition to recording
    # the seat in the corresponding `*_callers` set for end-of-round
    # scoring) so the replay's decision stream — and downstream consumers
    # like the parquet emitter and the call-network dataset — see the call
    # decision. Without this, Grand Tichu (which is ALWAYS declared in
    # this block) never reaches the action stream, and pre-schupfen Tichu
    # calls are dropped too.
    while i < len(lines):
        line = lines[i].rstrip()
        m = _GROSSES_TICHU_RE.match(line)
        if m:
            seat = int(m.group(1))
            builder.grand_tichu_callers.add(seat)
            builder.plays.append(ParsedAction(player=seat, kind="grand_tichu"))
            i += 1
            continue
        m = _TICHU_RE.match(line)
        if m:
            seat = int(m.group(1))
            builder.tichu_callers.add(seat)
            builder.plays.append(ParsedAction(player=seat, kind="tichu"))
            i += 1
            continue
        if line.startswith("Schupfen:"):
            i += 1
            break
        if not line:
            i += 1
            continue
        raise ValueError(
            f"round {round_index}: unexpected line before Schupfen: {line!r}"
        )

    # Four schupfen lines
    for seat in range(4):
        i = _skip_blank(lines, i)
        sub, i = _parse_schupfen_line(lines, i, seat=seat)
        builder.schupfen[seat] = sub

    # Optional BOMBE: lines and Rundenverlauf header.
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("BOMBE:") or not line:
            i += 1
            continue
        if "Rundenverlauf" in line:
            i += 1
            break
        raise ValueError(
            f"round {round_index}: unexpected line before Rundenverlauf: {line!r}"
        )

    # Play actions until Ergebnis.
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        m = _ERGEBNIS_RE.match(line)
        if m:
            builder.ergebnis = (int(m.group(1)), int(m.group(2)))
            i += 1
            break
        _consume_action_line(builder, line, round_index)
        i += 1

    if builder.ergebnis is None:
        raise ValueError(f"round {round_index}: truncated log")
    return builder, i


def _skip_blank(lines: list[str], i: int) -> int:
    while i < len(lines) and not lines[i].strip():
        i += 1
    return i


def _parse_player_card_line(
    lines: list[str], i: int
) -> tuple[str, list[CardOrSpecial], int]:
    line = lines[i].rstrip()
    m = _PLAYER_LINE_RE.match(line)
    if not m:
        raise ValueError(f"expected '(N)handle <cards>' line, got {line!r}")
    handle = _clean_handle(m.group(2))
    tokens = m.group(3).split()
    cards = [parse_card_token(t) for t in tokens]
    return handle, cards, i + 1


def _parse_schupfen_line(
    lines: list[str], i: int, *, seat: int
) -> tuple[ParsedAction, int]:
    line = lines[i].rstrip()
    m = _SCHUPFEN_GIBT_RE.match(line)
    if not m:
        raise ValueError(f"expected schupfen 'gibt:' line, got {line!r}")
    seat_in_line = int(m.group(1))
    if seat_in_line != seat:
        raise ValueError(f"schupfen seat {seat_in_line} out of order, expected {seat}")
    body = m.group(3)
    # Body looks like "handleA: cardX - handleB: cardY - handleC: cardZ - ".
    # The real recipient separator is " - " (space-dash-space) — splitting on
    # bare "-" over-splits handles that contain a hyphen (e.g. "raf-4").
    # Peel the trailing " -" sentinel first so the split yields exactly the
    # three recipient strings.
    body = body.rstrip()
    if body.endswith("-"):
        body = body[:-1].rstrip()
    # Recipients appear in seat order: (seat+1) % 4, (seat+2) % 4, (seat+3) % 4.
    parts = [p.strip() for p in body.split(" - ") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"schupfen line should have 3 recipients, got {parts}")
    cards: list[CardOrSpecial] = []
    for part in parts:
        _, card_token = part.split(":", 1)
        cards.append(parse_card_token(card_token.strip()))
    return (
        ParsedAction(
            player=seat,
            kind="schupfen",
            schupfen_to_next=cards[0],
            schupfen_to_partner=cards[1],
            schupfen_to_previous=cards[2],
        ),
        i + 1,
    )


def _consume_action_line(builder: _RoundBuilder, line: str, round_index: int) -> None:
    """Append one play/pass/wish/dragon_give/tichu action to the builder."""
    m = _PASST_RE.match(line)
    if m:
        seat = int(m.group(1))
        builder.plays.append(ParsedAction(player=seat, kind="pass"))
        builder._last_player = seat
        return

    m = _WUNSCH_RE.match(line)
    if m:
        rank_code = m.group(1)
        if rank_code not in _RANK_BY_CODE:
            raise ValueError(f"round {round_index}: unknown Wunsch rank {rank_code!r}")
        wish_rank = _RANK_BY_CODE[rank_code]
        # The wish belongs to the player who just played the Mahjong.
        attributed_player = builder._last_player if builder._last_player is not None else -1
        builder.plays.append(
            ParsedAction(player=attributed_player, kind="wish", wish_rank=wish_rank)
        )
        return

    m = _TICHU_RE.match(line)
    if m:
        seat = int(m.group(1))
        builder.tichu_callers.add(seat)
        builder.plays.append(ParsedAction(player=seat, kind="tichu"))
        return

    m = _GROSSES_TICHU_RE.match(line)
    if m:
        # Rare but possible inside Rundenverlauf? Defensive support.
        seat = int(m.group(1))
        builder.grand_tichu_callers.add(seat)
        builder.plays.append(ParsedAction(player=seat, kind="grand_tichu"))
        return

    m = _DRAGON_RE.match(line)
    if m:
        target = int(m.group(1))
        # Giver is the trick winner — `_last_player` is the most recent player
        # to have played, which is the trick winner in BSW's serialisation.
        attributed_player = builder._last_player if builder._last_player is not None else -1
        builder.plays.append(
            ParsedAction(
                player=attributed_player,
                kind="dragon_give",
                dragon_target=target,
            )
        )
        return

    # Plays are formatted "(N)handle: card card ..." — handle ends with ':'.
    # Handle may be empty (BSW anonymous seat); the colon is what makes this
    # a play line.
    m = re.match(rf"^\((\d)\)({_HANDLE}?):\s*(.*)$", line)
    if m:
        seat = int(m.group(1))
        body = m.group(3).strip()
        if not body:
            # An empty play body shouldn't happen; treat as a parse error.
            raise ValueError(f"round {round_index}: empty play body in {line!r}")
        tokens = body.split()
        cards = tuple(parse_card_token(t) for t in tokens)
        builder.plays.append(ParsedAction(player=seat, kind="play", cards=cards))
        builder._last_player = seat
        return

    raise ValueError(f"round {round_index}: unrecognised line: {line!r}")
