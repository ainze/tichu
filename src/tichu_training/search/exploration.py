"""ExplorationAgent — exploration wrapper for the exploration-critic pilot (ADR-0030).

The deep-budget result falsified search-only on frozen nets: a critic regressed
from master self-play encodes the *passive* policy's values, so deeper PIMC
faithfully optimises a passive target and gets MORE passive, not less. The pilot
tests the cheapest escape — refit the critic on a self-play distribution that
actually *contains* aggressive caller-states with their true `round_outcome`,
then plug it into the existing `SearchAgent` (no new search code).

This wrapper is the data-generation half: it perturbs ONLY Play Decisions with
uniform ε-greedy over the full legal set. Uniform (not policy-weighted) is the
point — it visits beating plays and bombs at their structural frequency rather
than the master's suppressed one, so coverage broadens across passivity *as a
whole*, not just the bomb subset. Everything else (pending wish/dragon/schupfen
and the Tichu/Grand calls) delegates to the wrapped master, keeping the call and
leadership structure realistic so the explored play-states are labelled with
near-on-policy continuations.
"""

import random

from tichu_engine.legality import legal_actions_for


class ExplorationAgent:
    """ε-greedy exploration over Play Decisions; master everywhere else."""

    def __init__(self, master, *, epsilon: float = 0.2, seed: int = 0) -> None:
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self._master = master
        self._epsilon = float(epsilon)
        self._rng = random.Random(seed)

    def act(self, private_state):
        # Non-Play Decisions stay master — only in-trick play coverage broadens.
        if private_state.public.pending_decision is not None:
            return self._master.act(private_state)
        if self._epsilon > 0.0 and self._rng.random() < self._epsilon:
            legal = list(legal_actions_for(private_state))
            if legal:
                return self._rng.choice(legal)
        return self._master.act(private_state)

    def should_call(self, private_state, kind: str) -> bool:
        return self._master.should_call(private_state, kind)
