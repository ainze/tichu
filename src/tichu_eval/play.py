"""Single-Round runner.

Drives one round from a starting `GameState` to round-end by repeatedly
asking the seated agent for an action and applying `engine.step`. Returns
the team-0 and team-1 score deltas relative to the deal's initial scores.

Scope notes:
  * Bomb Windows (ADR-0045) are always on: after every Play, each seat
    `bomb_window_queue` lists is asked "bomb now, or Pass?" before the seat on
    turn. A declined window changes nothing.
  * Tichu / Grand Tichu calling is out of scope at this slice: each deal
    is played with empty caller sets. Composing call networks with a
    play agent is a follow-up.
"""

from typing import Sequence

from tichu_engine.engine import step
from tichu_engine.legality import Pass, bomb_window_queue
from tichu_engine.state import GameState
from tichu_ml.agent import Agent


_MAX_STEPS = 10_000  # safety net against an infinite-loop bug.


def play_round(agents: Sequence[Agent], initial_state: GameState) -> tuple[int, int]:
    if len(agents) != 4:
        raise ValueError(f"expected 4 agents (one per seat), got {len(agents)}")
    initial_scores = initial_state.public.scores
    state = initial_state
    done = False
    window_queue: list[int] = []
    for _ in range(_MAX_STEPS):
        if done:
            break
        in_window = bool(window_queue)
        actor = window_queue.pop(0) if in_window else state.public.current_player
        action = agents[actor].act(state.private_view(actor))
        if in_window and isinstance(action, Pass):
            continue
        state, _, done, _ = step(state, action)
        if not done:
            window_queue = list(bomb_window_queue(state, action, actor))
    else:
        raise RuntimeError(
            f"play_round exceeded {_MAX_STEPS} steps without resolving — "
            "likely an infinite loop in agent or engine."
        )
    final = state.public.scores
    return (final[0] - initial_scores[0], final[1] - initial_scores[1])
