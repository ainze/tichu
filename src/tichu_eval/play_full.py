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

from tichu_engine.engine import step
from tichu_engine.legality import Pass
from tichu_engine.state import GameState, PublicState, Trick
from tichu_ml.agent import Agent


_MAX_STEPS = 10_000  # safety net against an infinite-loop bug.


class FullRoundResult(NamedTuple):
    total: tuple[int, int]
    call_bonus: tuple[int, int]


def play_full_round(
    agents: Sequence[Agent],
    start_state: GameState,
    grand_prefixes: Sequence[frozenset],
) -> FullRoundResult:
    if len(agents) != 4:
        raise ValueError(f"expected 4 agents (one per seat), got {len(agents)}")
    initial_scores = start_state.public.scores

    grand_callers = _ask_grand(agents, grand_prefixes)
    tichu_callers: set[int] = set()
    state = _with_callers(start_state, grand=grand_callers)

    asked_tichu: set[int] = set()
    first_out: int | None = None
    done = False
    for _ in range(_MAX_STEPS):
        if done:
            break
        current = state.public.current_player
        private = state.private_view(current)
        action = agents[current].act(private)
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
        state, _, done, _ = step(state, action)
        if first_out is None and state.public.out_order:
            first_out = state.public.out_order[0]
    else:
        raise RuntimeError(
            f"play_full_round exceeded {_MAX_STEPS} steps without resolving — "
            "likely an infinite loop in agent or engine."
        )

    final = state.public.scores
    total = (final[0] - initial_scores[0], final[1] - initial_scores[1])
    call_bonus = _call_bonus(grand_callers, frozenset(tichu_callers), first_out)
    return FullRoundResult(total=total, call_bonus=call_bonus)


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
