"""Behavioral Drift Benchmark — the per-Decision recorder.

Plays one Full-strength Round and logs **one row per Decision** (all 6 kinds)
plus **one row per Round** (the outcome). Metrics are never counted here: the
recorder writes raw situation facts and the chosen Action, and every metric is a
pure function over the log computed afterwards (`drift_metrics`). That keeps the
play loop free of metric code and lets a new metric run over an existing log
without replaying a single Round. See CONTEXT.md §"Behavioral Drift Benchmark".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tichu_engine.cards import DOG, DRAGON, MAHJONG, PHOENIX, SpecialCard
from tichu_engine.combinations import (
    FourOfAKindBomb,
    FullHouse,
    Pair,
    PairStep,
    Single,
    Straight,
    StraightFlushBomb,
    Triple,
)
from tichu_engine.engine import trick_point_value
from tichu_engine.enumeration import (
    enumerate_four_of_a_kind_bombs,
    enumerate_straight_flush_bombs,
)
from tichu_engine.legality import Pass, _cards_in, legal_actions
from tichu_eval.play_full import play_full_round

_BOMB_TYPES = (FourOfAKindBomb, StraightFlushBomb)
_RANK_LABELS = {11: "J", 12: "Q", 13: "K", 14: "A"}
_COMBO_LABELS = (
    (Pair, "pair"), (Triple, "triple"), (FullHouse, "full_house"),
    (Straight, "straight"), (PairStep, "pair_step"),
)


@dataclass
class RoundLog:
    """One Round's recording: a row per Decision and one outcome row."""

    decisions: list[dict] = field(default_factory=list)
    round: dict = field(default_factory=dict)


def record_round(agents, position, *, deal: int, half: int, subject_seats) -> RoundLog:
    """Play `position` with `agents` (one per seat) and record every Decision.

    `deal` / `half` identify the Round within the benchmark (the deal index in the
    Pool and the Seat-Swap half); `subject_seats` marks which seats belong to the
    Subject team, the only seats a metric reads in the Fixed-Opponent design."""
    subject = frozenset(subject_seats)
    log = RoundLog()
    tracker = RoundTracker()

    def on_decision(seat, kind, game_state, action):
        tracker.observe(kind, action)
        log.decisions.append(decision_row(seat, kind, game_state, action,
                                          deal=deal, half=half, subject=subject))

    result = play_full_round(agents, position.state, position.grand_prefixes,
                             decision_observer=on_decision, collect_telemetry=True)
    tel = result.telemetry
    log.round = round_row(
        deal=deal, half=half, subject_team=min(subject) % 2,
        totals=result.total, call_bonus=result.call_bonus,
        grand_callers=tuple(s for s in range(4) if tel.grand_called[s]),
        tichu_callers=tuple(s for s in range(4) if tel.tichu_called[s]),
        out_order=tel.out_order, tricks_won=tuple(tel.tricks_won),
        tricks_total=tel.tricks_total, tracker=tracker,
    )
    return log


class RoundTracker:
    """Round-long facts gathered Decision by Decision: which specials were played
    and whether a wished rank was later played (the engine clears the wish on the
    first natural card of that rank)."""

    def __init__(self) -> None:
        self.played: set = set()
        self.wish_rank: int | None = None
        self.wish_fulfilled: bool | None = None   # None: no rank was wished

    def observe(self, kind: str, action) -> None:
        if kind == "play" and not isinstance(action, Pass):
            cards = _cards_in(action)
            self.played.update(cards)
            if self.wish_fulfilled is False and any(
                getattr(c, "rank", None) == self.wish_rank for c in cards
            ):
                self.wish_fulfilled = True
        if kind == "wish" and action.rank is not None:
            self.wish_rank, self.wish_fulfilled = action.rank, False


def decision_row(seat: int, kind: str, game_state, action, *, deal, half, subject) -> dict:
    """One Decision's log row: identity, the situation it was taken in (from the
    full GameState) and the chosen Action — shared by every source of Decisions
    (benchmark self-play and replayed human games alike)."""
    row = {
        "deal": deal,
        "half": half,
        "seat": seat,
        "team": seat % 2,
        "is_subject": seat in subject,
        "kind": kind,
    }
    if kind == "play":
        row.update(_play_facts(seat, game_state, action))
    elif kind in ("grand", "tichu"):
        row.update(_call_facts(seat, game_state, action))
    elif kind == "schupfen":
        row.update(_schupfen_facts(seat, game_state, action))
    elif kind == "wish":
        row["wish"] = "none" if action.rank is None else _RANK_LABELS.get(
            action.rank, str(action.rank))
    elif kind == "dragon":
        row.update(_dragon_facts(seat, game_state, action))
    return row


def round_row(*, deal, half, subject_team, totals, call_bonus, grand_callers, tichu_callers,
              out_order, tricks_won, tricks_total, tracker: RoundTracker) -> dict:
    """One Round's outcome row. `tricks_won` / `tricks_total` may be None when the
    source cannot attribute Tricks (replayed human games)."""
    out = tuple(out_order)
    slam = len(out) >= 2 and out[0] % 2 == out[1] % 2
    return {
        "deal": deal,
        "half": half,
        "subject_team": subject_team,
        "total_0": totals[0],
        "total_1": totals[1],
        "call_bonus_0": call_bonus[0],
        "call_bonus_1": call_bonus[1],
        "grand_callers": tuple(grand_callers),
        "tichu_callers": tuple(tichu_callers),
        "out_order": out,
        "slam_team": out[0] % 2 if slam else None,
        "tricks_won": tricks_won,
        "tricks_total": tricks_total,
        # None when no rank was wished; else whether a natural card of it was
        # later played (the engine clears the wish exactly then).
        "wish_fulfilled": tracker.wish_fulfilled,
        "mahjong_played": MAHJONG in tracker.played,
        "dragon_played": DRAGON in tracker.played,
    }


def _play_facts(seat: int, game_state, action) -> dict:
    """The situation a Play Decision was taken in, plus the Action's type.

    `holder` is who currently holds the Trick — `Trick.leader` IS the current
    winner (reassigned on every play), not the opener: "none" when leading an
    empty Trick, else "partner" / "opponent" relative to the acting seat."""
    public = game_state.public
    legal = legal_actions(game_state)
    leader = public.trick.leader
    if leader is None:
        holder = "none"
    elif leader == seat:
        holder = "self"
    else:
        holder = "partner" if leader % 2 == seat % 2 else "opponent"
    callers = _callers(public)
    hand = game_state.hands[seat]
    cards = () if isinstance(action, Pass) else _cards_in(action)
    top = public.trick.top_combination
    return {
        "forced": len(legal) == 1,
        "holder": holder,
        "has_beat": any(not isinstance(a, Pass) for a in legal),
        "bomb_legal": any(isinstance(a, _BOMB_TYPES) for a in legal),
        "partner_called": (seat + 2) % 4 in callers,
        "opponent_called": bool({(seat + 1) % 4, (seat + 3) % 4} & callers),
        "trick_points": trick_point_value(public.trick),
        "top_rank": None if top is None else float(top.rank),
        "self_called": seat in callers,
        "holds_dog": DOG in hand,
        "holds_bomb": bool(_bombs(hand)),
        "plays_dragon": DRAGON in cards,
        "plays_phoenix": PHOENIX in cards,
        "action": _action_label(action),
    }


def _action_label(action) -> str:
    """The Play's Combination type: pass / single / dog / pair / ... / bomb."""
    if isinstance(action, Pass):
        return "pass"
    if isinstance(action, _BOMB_TYPES):
        return "bomb"
    if isinstance(action, Single):
        return "dog" if action.card is DOG else "single"
    for cls, label in _COMBO_LABELS:
        if isinstance(action, cls):
            return label
    raise TypeError(f"unknown Play action {action!r}")


def card_label(card) -> str:
    """A card's rank as a player says it: 2..10, J, Q, K, A, or the special's
    name (dog / mahjong / phoenix / dragon). Suits are irrelevant to every metric."""
    if isinstance(card, SpecialCard):
        return card.name
    return _RANK_LABELS.get(card.rank, str(card.rank))


def _callers(public) -> frozenset:
    return public.tichu_callers | public.grand_tichu_callers


def _bombs(hand) -> list:
    return [*enumerate_four_of_a_kind_bombs(hand), *enumerate_straight_flush_bombs(hand)]


def _call_facts(seat: int, game_state, called: bool) -> dict:
    """A Grand-Tichu / Tichu Call: the answer, who had already called, and the
    Hand's **power** (Dragon + Phoenix + Aces) and Bomb count — the strata that
    separate "calls more often" from "calls on weaker hands"."""
    hand = game_state.hands[seat]
    callers = _callers(game_state.public)
    return {
        "called": bool(called),
        "power": sum(1 for c in hand if c in (DRAGON, PHOENIX) or getattr(c, "rank", 0) == 14),
        "n_bombs": len(_bombs(hand)),
        "partner_called": (seat + 2) % 4 in callers,
        "opponent_called": bool({(seat + 1) % 4, (seat + 3) % 4} & callers),
    }


def _schupfen_facts(seat: int, game_state, action) -> dict:
    """The card given in each direction, the specials held, whether the partner
    had called Grand (the only Call visible before Schupfen), and whether a gift
    broke a Bomb the seat held."""
    hand = game_state.hands[seat]
    gifts = (action.to_next, action.to_partner, action.to_previous)
    bomb_cards = {c for b in _bombs(hand) for c in _cards_in(b)}
    return {
        "give_next": card_label(action.to_next),
        "give_partner": card_label(action.to_partner),
        "give_previous": card_label(action.to_previous),
        "holds_dog": DOG in hand,
        "holds_dragon": DRAGON in hand,
        "holds_phoenix": PHOENIX in hand,
        "holds_mahjong": MAHJONG in hand,
        "partner_grand": (seat + 2) % 4 in game_state.public.grand_tichu_callers,
        "had_bomb": bool(bomb_cards),
        "bomb_broken": any(g in bomb_cards for g in gifts),
    }


def _dragon_facts(seat: int, game_state, action) -> dict:
    """Dragon Assignment: which opponent received the Trick (next / previous in
    turn order), whether they hold more / fewer / equal cards than the other
    opponent, and whether the receiver has called."""
    nxt, prv = (seat + 1) % 4, (seat + 3) % 4
    target, other = (nxt, prv) if action.target == nxt else (prv, nxt)
    sizes = game_state.public.hand_sizes
    return {
        "target": "next" if target == nxt else "previous",
        "target_cards": ("more" if sizes[target] > sizes[other]
                         else "fewer" if sizes[target] < sizes[other] else "equal"),
        "target_called": target in _callers(game_state.public),
    }
