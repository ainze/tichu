"""Decision tape — a human-reviewable record of an Agent's Play Decisions.

For each Play Decision in a Round, records the acting seat's Hand, the current
Trick top, what the Agent chose, and its top-k ranked legal alternatives with
policy probabilities. Lets a strong human reviewer localise *concrete* tactical
weaknesses (is it playing a soft line? wasting a card? mis-ranking?) — the
diagnostic that decides whether the play-strength gap is tactical sharpness
(→ PPO) vs the Resolver / Schupfen.

Agent-agnostic for the choice; the ranked alternatives need an Agent exposing
`play_action_scores(private_state)` (MLAgent does). Built on the `observer` hook
of `play_full_round`, so it drives the exact Full-strength eval path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tichu_engine.cards import Card
from tichu_engine.combinations import FourOfAKindBomb, StraightFlushBomb
from tichu_engine.legality import DragonGive, MahjongWish, Pass, SchupfenPass, _cards_in
from tichu_eval.play_full import play_full_round

_BOMB_TYPES = (FourOfAKindBomb, StraightFlushBomb)


def _card(c) -> str:
    if isinstance(c, Card):
        return f"{c.suit.value}-{c.rank}"
    return getattr(c, "name", str(c))   # Dragon / Phoenix / Dog / Mahjong


def _action_str(action) -> str:
    if isinstance(action, Pass):
        return "PASS"
    cards = sorted(_cards_in(action), key=_sort_key)
    return f"{type(action).__name__}[{','.join(_card(c) for c in cards)}]"


def _sort_key(c):
    r = getattr(c, "rank", 0)
    try:
        r = float(r)
    except (TypeError, ValueError):
        r = 0.0
    return (r, _card(c))


@dataclass
class DecisionRecord:
    round_idx: int
    seat: int
    hand: list[str]
    leading: bool
    top: str
    chosen: str
    chosen_prob: float
    topk: list[tuple[str, float]]
    bomb_legal: bool


@dataclass
class TapeSink:
    records: list[DecisionRecord] = field(default_factory=list)


def build_record(agent, private_state, action, *, seat: int = 0,
                 round_idx: int = 0, top_k: int = 5) -> DecisionRecord | None:
    """One reviewable DecisionRecord from an agent's choice, or None if the move
    was forced (0/1 legal) or the agent exposes no `play_action_scores`."""
    scores = getattr(agent, "play_action_scores", None)
    ranked = scores(private_state) if scores is not None else []
    if len(ranked) < 2:
        return None
    chosen = _action_str(action)
    chosen_prob = next((p for a, p in ranked if _action_str(a) == chosen), float("nan"))
    top = private_state.public.trick.top_combination
    return DecisionRecord(
        round_idx=round_idx,
        seat=seat,
        hand=[_card(c) for c in sorted(private_state.hand, key=_sort_key)],
        leading=top is None,
        top="—(lead)" if top is None else _action_str(top),
        chosen=chosen,
        chosen_prob=chosen_prob,
        topk=[(_action_str(a), p) for a, p in ranked[:top_k]],
        bomb_legal=any(isinstance(a, _BOMB_TYPES) for a, _ in ranked),
    )


def render_record(r: DecisionRecord, *, header: str | None = None) -> str:
    """Render one DecisionRecord as a reviewable text block."""
    flag = " [BOMB legal]" if r.bomb_legal else ""
    unsure = " [unsure]" if r.chosen_prob == r.chosen_prob and r.chosen_prob < 0.5 else ""
    head = f"{header}\n" if header else ""
    return (
        f"{head}seat {r.seat}  top={r.top}{flag}{unsure}\n"
        f"    hand: {' '.join(r.hand)}\n"
        f"    chose: {r.chosen}  (p={r.chosen_prob:.2f})\n"
        f"    top{len(r.topk)}: " + "   ".join(f"{a} {p:.2f}" for a, p in r.topk)
    )


def _hand_line(private_state) -> str:
    return "    hand: " + " ".join(
        _card(c) for c in sorted(private_state.hand, key=_sort_key)
    )


def render_wish(agent, private_state, action, *, header: str | None = None) -> str | None:
    """Mahjong-Wish block: chosen rank, ranked alternatives, and the call context
    (grand/tichu callers) — the signal a Wish should react to."""
    scores = getattr(agent, "wish_action_scores", None)
    ranked = scores(private_state) if scores is not None else []
    if not ranked:
        return None
    chosen = action.rank if isinstance(action, MahjongWish) else None
    gt = sorted(private_state.public.grand_tichu_callers) or "-"
    ti = sorted(private_state.public.tichu_callers) or "-"
    top = "  ".join(f"{a.rank}:{p:.2f}" for a, p in ranked[:8])
    head = f"{header}\n" if header else ""
    return (f"{head}WISH  chose={chosen}  grand_callers={gt}  tichu_callers={ti}\n"
            f"{_hand_line(private_state)}\n    ranks: {top}")


def render_dragon(agent, private_state, action, *, header: str | None = None) -> str | None:
    scores = getattr(agent, "dragon_action_scores", None)
    ranked = scores(private_state) if scores is not None else []
    if not ranked:
        return None
    chosen = action.target if isinstance(action, DragonGive) else None
    top = "  ".join(f"seat{a.target}:{p:.2f}" for a, p in ranked)
    head = f"{header}\n" if header else ""
    return f"{head}DRAGON  chose=seat{chosen}\n    targets: {top}"


def render_schupfen(agent, private_state, action, *, header: str | None = None) -> str | None:
    """Schupfen block: the card given in each direction (next/partner/previous)
    with the head's top candidate cards — to judge passes like 'gave the Ace'."""
    scores = getattr(agent, "schupfen_action_scores", None)
    perdir = scores(private_state) if scores is not None else {}
    if not perdir:
        return None
    chosen = {}
    if isinstance(action, SchupfenPass):
        chosen = {"next": action.to_next, "partner": action.to_partner,
                  "previous": action.to_previous}
    lines = [header] if header else []
    lines.append("SCHUPFEN")
    lines.append(_hand_line(private_state))
    for d in ("next", "partner", "previous"):
        ch = chosen.get(d)
        cand = "  ".join(f"{_card(c)}:{p:.2f}" for c, p in perdir.get(d, [])[:4])
        lines.append(f"    {d:>8} -> {_card(ch) if ch is not None else '?'}   cand: {cand}")
    return "\n".join(lines)


def _make_observer(agents, round_idx: int, sink: TapeSink, top_k: int):
    def observer(seat: int, private_state, action) -> None:
        rec = build_record(agents[seat], private_state, action,
                           seat=seat, round_idx=round_idx, top_k=top_k)
        if rec is not None:
            sink.records.append(rec)
    return observer


def record_round(agents, position, round_idx: int, sink: TapeSink, *, top_k: int = 5) -> None:
    play_full_round(
        agents, position.state, position.grand_prefixes,
        observer=_make_observer(agents, round_idx, sink, top_k),
    )


def render_text(sink: TapeSink) -> str:
    lines: list[str] = []
    cur_round = -1
    for r in sink.records:
        if r.round_idx != cur_round:
            cur_round = r.round_idx
            lines.append(f"\n=== Round {r.round_idx} ===")
        lines.append(render_record(r))
    return "\n".join(lines)
