"""End-to-end smoke test: play a full round with random legal actions.

This exercises the entire engine — deal, legal-action enumeration, step,
trick resolution, scoring, pending decisions, round-end detection — without
asserting anything about strategy. It catches crashes, illegal moves, and
infinite loops.
"""

import random

from tichu_engine.engine import step
from tichu_engine.legality import legal_actions
from tichu_engine.state import deal_for_schupfen, deal_initial_state


def _play_one_round(seed: int, max_steps: int = 2000, *, with_schupfen: bool = False) -> tuple[int, int]:
    """Play one round with uniformly random legal actions. Returns step count
    and final score difference."""
    rng = random.Random(seed)
    state = deal_for_schupfen(seed=seed) if with_schupfen else deal_initial_state(seed=seed)
    steps_taken = 0
    while steps_taken < max_steps:
        options = list(legal_actions(state))
        assert options, f"no legal actions at step {steps_taken} for player {state.public.current_player}"
        action = rng.choice(options)
        state, _, done, _ = step(state, action)
        steps_taken += 1
        if done:
            return steps_taken, state.public.scores[0] - state.public.scores[1]
    raise AssertionError(f"round did not terminate in {max_steps} steps")


def test_random_rollout_completes_for_several_seeds():
    for seed in range(20):
        steps_taken, _ = _play_one_round(seed=seed)
        assert steps_taken < 500, f"seed={seed} took {steps_taken} steps"


def test_random_rollout_produces_finite_scores():
    for seed in range(5):
        _, diff = _play_one_round(seed=seed)
        # Most card-point movements in a single round stay well under 200 in magnitude,
        # but final-round transfer rules aren't implemented yet so we only check finiteness.
        assert -300 < diff < 300


def test_random_rollout_with_schupfen_completes():
    # Full flow: deal -> 4 schupfen submissions -> play -> done.
    for seed in range(5):
        steps_taken, _ = _play_one_round(seed=seed, with_schupfen=True)
        assert steps_taken < 500, f"seed={seed} took {steps_taken} steps"
