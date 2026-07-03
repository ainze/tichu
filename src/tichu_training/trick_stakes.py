"""Trick point-stakes features (ADR-0039), torch-free and pure.

H1 (`trick_point_value`) lives in the engine as the single source of the
scoring mapping; re-exported here so the pre-check and the v7 featurizer pull
both stakes features from one module.

H2 (`current_trick_winner_relseat`) is the relative-seat index of the player
holding the top Combination — the seat the acting player overtakes if they beat
the pile. Distinct from the trick *leader* (opener): the two differ after any
raise, and the winner's identity is not otherwise featurised.
"""

import numpy as np

from tichu_engine.engine import trick_point_value  # noqa: F401  (re-exported)
from tichu_engine.state import PrivateState


# The v7 stakes block: [trick_point_value / 25, winner one-hot × 4].
STAKES_BLOCK_DIM: int = 5
# H1 normaliser — Dragon (+25) = 1.0. Only bounds the ratio (cf. LEAD_TRICKS_CAP);
# not clipped, so rare high-stakes piles still rank-order (ADR-0039).
TRICK_POINT_DIVISOR: float = 25.0


def current_trick_winner_relseat(private_state: PrivateState) -> int | None:
    """Relative-seat index (0=self, 1=next, 2=partner, 3=previous) of the seat
    currently winning the Trick, or None on an empty Trick."""
    plays = private_state.public.trick.plays
    if not plays:
        return None
    return (plays[-1].player - private_state.player) % 4


def is_contested_trick_decision(private_state: PrivateState, legal_actions) -> bool:
    """Whether this play Decision is on the contested-trick slice (ADR-0039):
    a non-empty Trick with non-zero point stakes where the acting seat has a
    legal non-Pass action (it can beat the standing top). `legal_actions` is the
    engine's frozenset of legal ConcreteActions at the decision boundary."""
    from tichu_engine.legality import Pass

    trick = private_state.public.trick
    if not trick.plays:
        return False
    if trick_point_value(trick) == 0:
        return False
    return any(not isinstance(a, Pass) for a in legal_actions)


def stakes_block(private_state: PrivateState) -> np.ndarray:
    """The 5-dim v7 stakes block for a PrivateState:
    [trick_point_value / 25, winner_self, winner_next, winner_partner,
    winner_prev]. All-zero on an empty Trick. Mirrors what the v7 featurizer
    will emit (H1 continuous + H2 one-hot) so the pre-check measures the real
    feature, not a proxy."""
    out = np.zeros(STAKES_BLOCK_DIM, dtype=np.float32)
    if not private_state.public.trick.plays:
        return out
    out[0] = trick_point_value(private_state.public.trick) / TRICK_POINT_DIVISOR
    rel = current_trick_winner_relseat(private_state)
    out[1 + rel] = 1.0
    return out
