"""Single-deal runner.

Drives one round from a starting `GameState` to round-end by repeatedly
asking the seated agent for an action and applying `engine.step`. Returns
the team-0 and team-1 score deltas relative to the deal's initial scores.

Scope notes:
  * Out-of-turn `BombInterrupt`s are not exercised — `legal_actions_for`
    only returns in-turn actions for the current player.
  * Tichu / Grand Tichu calling is out of scope at this slice: each deal
    is played with empty caller sets. Composing call networks with a
    play agent is a follow-up.
"""

from typing import Sequence

from tichu_engine.engine import step
from tichu_engine.state import GameState
from tichu_ml.agent import Agent


_MAX_STEPS = 10_000  # safety net against an infinite-loop bug.


def play_deal(agents: Sequence[Agent], initial_state: GameState) -> tuple[int, int]:
    if len(agents) != 4:
        raise ValueError(f"expected 4 agents (one per seat), got {len(agents)}")
    initial_scores = initial_state.public.scores
    state = initial_state
    done = False
    for _ in range(_MAX_STEPS):
        if done:
            break
        current = state.public.current_player
        private = state.private_view(current)
        action = agents[current].act(private)
        state, _, done, _ = step(state, action)
    else:
        raise RuntimeError(
            f"play_deal exceeded {_MAX_STEPS} steps without resolving — "
            "likely an infinite loop in agent or engine."
        )
    final = state.public.scores
    return (final[0] - initial_scores[0], final[1] - initial_scores[1])
