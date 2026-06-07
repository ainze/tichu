"""Full-strength single-Round runner.

Drives one Round through the complete product stack — Grand-Tichu Call ->
Schupfen -> Tichu Call -> Play -> Wish -> Dragon — from a pre-Schupfen
deal-time `GameState`, and reports the team score deltas plus the Call-bonus
component. See ADR-0025.

The engine does not drive the two Call Decisions (it only reads
`tichu_callers` / `grand_tichu_callers` at `_finalise_round`), so this runner
orchestrates them out-of-band and injects the callers into the state. Agents
without a `should_call` method (the Baseline Agents) decline every Call, the
same convention the inference service uses.
"""

from dataclasses import replace
from typing import NamedTuple, Sequence

from tichu_engine.combinations import FourOfAKindBomb, StraightFlushBomb
from tichu_engine.engine import step
from tichu_engine.legality import Pass, legal_actions
from tichu_engine.state import GameState, PublicState, Trick
from tichu_ml.agent import Agent


_MAX_STEPS = 10_000  # safety net against an infinite-loop bug.

_BOMB_TYPES = (FourOfAKindBomb, StraightFlushBomb)


class RoundTelemetry(NamedTuple):
    """Per-seat behavioral record for one Full-strength Round (see ADR-0025).

    All four-tuples are indexed by absolute seat 0..3. Rates / success flags are
    not pre-computed here — the aggregator derives them from these raw counts so
    the round record stays a pure observation. Captured only when
    `play_full_round(..., collect_telemetry=True)`; the Tournament hot path leaves
    it off and pays nothing.
    """

    grand_called: tuple[bool, bool, bool, bool]
    tichu_called: tuple[bool, bool, bool, bool]
    first_out: int | None
    out_order: tuple[int, ...]            # full order, captured before finalise resets it
    bombs_played: tuple[int, int, int, int]
    bomb_legal_decisions: tuple[int, int, int, int]   # Play decisions where a bomb was legal
    tricks_won: tuple[int, int, int, int]
    tricks_total: int                     # total tricks resolved this Round (denominator)
    # Caller passivity: a Tichu/Grand-Tichu caller, following an OPPONENT's top,
    # with a legal beating play available. `_opps` counts those situations;
    # `_passes` counts how many it ceded by Passing. High ratio = a caller that
    # fails to press its lead (jeopardising the call). See decision-tape flag.
    caller_pass_opportunities: tuple[int, int, int, int]
    caller_pass_events: tuple[int, int, int, int]
    # Sharpest, least-ambiguous subset: caller passivity where a BOMB was legal
    # (almost never a reason to cede). Defaulted so older constructions still build.
    caller_pass_bomb_opportunities: tuple[int, int, int, int] = (0, 0, 0, 0)
    caller_pass_bomb_events: tuple[int, int, int, int] = (0, 0, 0, 0)


class FullRoundResult(NamedTuple):
    total: tuple[int, int]
    call_bonus: tuple[int, int]
    telemetry: RoundTelemetry | None = None


def play_full_round(
    agents: Sequence[Agent],
    start_state: GameState,
    grand_prefixes: Sequence[frozenset],
    *,
    collect_telemetry: bool = False,
    observer=None,
    state_observer=None,
) -> FullRoundResult:
    """`observer`, if given, is called `observer(seat, private_state, action)` at
    every Play Decision (pending_decision is None) right after the agent chooses,
    before the engine steps — the hook decision-tape tooling uses to record the
    agent's ranked alternatives. Pure side-channel; does not affect play.

    `state_observer`, if given, is called `state_observer(seat, game_state, action)`
    at the same point with the full **GameState** (all four hands) — the
    perfect-information hook the Perfect-Info Critic collector needs (ADR-0033).
    `game_state.private_view(seat)` reproduces what `observer` sees."""
    if len(agents) != 4:
        raise ValueError(f"expected 4 agents (one per seat), got {len(agents)}")
    initial_scores = start_state.public.scores

    grand_callers = _ask_grand(agents, grand_prefixes)
    tichu_callers: set[int] = set()
    state = _with_callers(start_state, grand=grand_callers)

    # Telemetry accumulators — touched only when `collect_telemetry`. Kept as
    # plain lists so the no-telemetry path adds nothing but four unused refs.
    bombs_played = [0, 0, 0, 0]
    bomb_legal_decisions = [0, 0, 0, 0]
    tricks_won = [0, 0, 0, 0]
    tricks_total = 0
    caller_pass_opportunities = [0, 0, 0, 0]
    caller_pass_events = [0, 0, 0, 0]
    caller_pass_bomb_opportunities = [0, 0, 0, 0]
    caller_pass_bomb_events = [0, 0, 0, 0]
    last_out_order: tuple[int, ...] = ()

    asked_tichu: set[int] = set()
    first_out: int | None = None
    done = False
    for _ in range(_MAX_STEPS):
        if done:
            break
        current = state.public.current_player
        private = state.private_view(current)
        # Telemetry probes computed before the agent acts (one legal_actions scan
        # shared by both). Only on a normal Play decision (no pending).
        passivity_opp = False
        passivity_bomb = False
        if collect_telemetry and state.public.pending_decision is None:
            legal = legal_actions(state)
            has_bomb = any(isinstance(a, _BOMB_TYPES) for a in legal)
            if has_bomb:
                bomb_legal_decisions[current] += 1
            # Caller passivity: this seat called Tichu/Grand, is FOLLOWING an
            # opponent's top (different team), and has a legal beating play.
            leader = state.public.trick.leader
            if (
                (current in grand_callers or current in tichu_callers)
                and leader is not None
                and leader != current
                and (leader % 2) != (current % 2)
                and any(not isinstance(a, Pass) for a in legal)
            ):
                caller_pass_opportunities[current] += 1
                passivity_opp = True
                if has_bomb:
                    caller_pass_bomb_opportunities[current] += 1
                    passivity_bomb = True
        action = agents[current].act(private)
        if collect_telemetry:
            if isinstance(action, _BOMB_TYPES):
                bombs_played[current] += 1
            if isinstance(action, Pass):
                if passivity_opp:
                    caller_pass_events[current] += 1
                if passivity_bomb:
                    caller_pass_bomb_events[current] += 1
        if observer is not None and state.public.pending_decision is None:
            observer(current, private, action)
        if state_observer is not None and state.public.pending_decision is None:
            state_observer(current, state, action)
        # Tichu Call: asked once per seat, at its first non-Pass Play state
        # (ADR-0018). Skipped for Grand-Tichu callers (grand supersedes tichu).
        if (
            state.public.pending_decision is None
            and not isinstance(action, Pass)
            and current not in asked_tichu
            and current not in grand_callers
        ):
            asked_tichu.add(current)
            if _calls(agents[current], private, "tichu"):
                tichu_callers.add(current)
                state = _with_callers(state, tichu=frozenset(tichu_callers))
        state, _, done, info = step(state, action)
        if first_out is None and state.public.out_order:
            first_out = state.public.out_order[0]
        if collect_telemetry:
            if "trick_winner" in info:
                tricks_won[info["trick_winner"]] += 1
                tricks_total += 1
            # out_order is reset to () by _finalise_round; keep the last
            # non-empty value so the slam check survives round end.
            if state.public.out_order:
                last_out_order = state.public.out_order
    else:
        raise RuntimeError(
            f"play_full_round exceeded {_MAX_STEPS} steps without resolving — "
            "likely an infinite loop in agent or engine."
        )

    final = state.public.scores
    total = (final[0] - initial_scores[0], final[1] - initial_scores[1])
    call_bonus = _call_bonus(grand_callers, frozenset(tichu_callers), first_out)
    telemetry = None
    if collect_telemetry:
        telemetry = RoundTelemetry(
            grand_called=tuple(s in grand_callers for s in range(4)),  # type: ignore[arg-type]
            tichu_called=tuple(s in tichu_callers for s in range(4)),  # type: ignore[arg-type]
            first_out=first_out,
            out_order=last_out_order,
            bombs_played=tuple(bombs_played),  # type: ignore[arg-type]
            bomb_legal_decisions=tuple(bomb_legal_decisions),  # type: ignore[arg-type]
            tricks_won=tuple(tricks_won),  # type: ignore[arg-type]
            tricks_total=tricks_total,
            caller_pass_opportunities=tuple(caller_pass_opportunities),  # type: ignore[arg-type]
            caller_pass_events=tuple(caller_pass_events),  # type: ignore[arg-type]
            caller_pass_bomb_opportunities=tuple(caller_pass_bomb_opportunities),  # type: ignore[arg-type]
            caller_pass_bomb_events=tuple(caller_pass_bomb_events),  # type: ignore[arg-type]
        )
    return FullRoundResult(total=total, call_bonus=call_bonus, telemetry=telemetry)


def _ask_grand(
    agents: Sequence[Agent], grand_prefixes: Sequence[frozenset]
) -> frozenset[int]:
    """Ask each seat for a Grand-Tichu Call on its synthetic (8,8,8,8) deal-time
    state, independently (no prior callers visible — matching training)."""
    hands = tuple(frozenset(p) for p in grand_prefixes)
    grand_state = GameState(
        hands=hands,
        public=PublicState(
            current_player=0,
            hand_sizes=tuple(len(h) for h in hands),
            scores=(0, 0),
            trick=Trick.empty(),
        ),
    )
    callers: set[int] = set()
    for seat in range(4):
        if _calls(agents[seat], grand_state.private_view(seat), "grand"):
            callers.add(seat)
    return frozenset(callers)


def _calls(agent: Agent, private_state, kind: str) -> bool:
    should_call = getattr(agent, "should_call", None)
    return bool(should_call(private_state, kind)) if should_call else False


def _with_callers(state: GameState, *, grand=None, tichu=None) -> GameState:
    kw: dict = {}
    if grand is not None:
        kw["grand_tichu_callers"] = frozenset(grand)
    if tichu is not None:
        kw["tichu_callers"] = frozenset(tichu)
    return GameState(hands=state.hands, public=replace(state.public, **kw))


def _call_bonus(
    grand_callers: frozenset[int], tichu_callers: frozenset[int], first_out: int | None
) -> tuple[int, int]:
    """The +/-100 (Tichu) and +/-200 (Grand-Tichu) effect on each team — exactly
    `_finalise_round`'s caller loop, computed from the first-out player."""
    bonus = [0, 0]
    for caller in tichu_callers:
        bonus[caller % 2] += 100 if caller == first_out else -100
    for caller in grand_callers:
        bonus[caller % 2] += 200 if caller == first_out else -200
    return (bonus[0], bonus[1])
