"""Perfect-Info feature + value-ceiling gate helpers (ADR-0033).

The **Perfect-Info Critic** (PTIE) sees all four hands at *training time*; the
**Policy Network** sees only the info-set and is the only thing at inference.
This module builds that train-time input — the 224-dim observable Feature Vector
+ the three opponents' *true* hands on the Belief Model's `(3 × 56)` relative-seat
grid (next/partner/previous; the same convention as `belief.emit`) — and the
helpers the `Perfect-Info Value-Ceiling Test` gate is scored with.

GameState-consuming (train-time only), so deliberately kept out of `featurizer.py`
to leave the glossary's *Featurizer = `featurize(PrivateState)`* untouched.
"""

from __future__ import annotations

import numpy as np

from tichu_training.card_slots import card_slot
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM, featurize

_NUM_OPPONENTS = 3
_NUM_CARDS = 56

PERFECT_INFO_DIM: int = FEATURIZER_OUTPUT_DIM + _NUM_OPPONENTS * _NUM_CARDS


def featurize_perfect_info(game_state, seat: int) -> np.ndarray:
    """`(PERFECT_INFO_DIM,)` float32: the observable Feature Vector for `seat`
    followed by the three opponents' true hands on the `(3, 56)` relative-seat
    grid (next / partner / previous), flattened. Reuses `card_slot` so the grid
    indexing matches `own_hand` / `seen_cards` and `belief.emit`.
    """
    observable = featurize(game_state.private_view(seat))
    grid = np.zeros((_NUM_OPPONENTS, _NUM_CARDS), dtype=np.float32)
    rel_seats = ((seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4)
    for opp_idx, opp_seat in enumerate(rel_seats):
        for card in game_state.hands[opp_seat]:
            grid[opp_idx, card_slot(card)] = 1.0
    return np.concatenate([observable, grid.reshape(-1)]).astype(np.float32)


def is_call_live(public_state, seat: int) -> bool:
    """True iff `seat` has an outstanding Tichu or Grand-Tichu call — the
    call-state breakout filter for the `Perfect-Info Value-Ceiling Test`
    (the ADR-0032 call-fulfillment trigger semantics)."""
    return (
        seat in public_state.tichu_callers
        or seat in public_state.grand_tichu_callers
    )


def r2_score(pred: np.ndarray, y: np.ndarray) -> float:
    """Coefficient of determination `1 - SS_res/SS_tot` (NaN if `y` is constant)."""
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def residual_variance_ratio(r2_perfect: float, r2_observable: float) -> float:
    """ρ = `(1 - R²_perfect) / (1 - R²_observable)` — the share of policy-gradient
    advantage variance that survives once the critic sees all hands. The gate
    metric (ADR-0033): ρ ≤ 0.5 → build; ρ ≥ 0.8 → kill ESCAPE; else PPO smoke."""
    return (1.0 - r2_perfect) / (1.0 - r2_observable)


def split_rho(
    y: np.ndarray,
    pred_observable: np.ndarray,
    pred_perfect: np.ndarray,
    call_live: np.ndarray,
) -> dict[str, dict[str, float]]:
    """The three-way gate readout: ρ (+ both R²s and the row count) over
    **overall**, **call-live** states, and **the rest** — call-live being the
    highest-variance, partnership-relevant sub-game and the build's #1 risk."""
    out: dict[str, dict[str, float]] = {}
    splits = {
        "overall": np.ones(len(y), dtype=bool),
        "call_live": np.asarray(call_live, dtype=bool),
        "rest": ~np.asarray(call_live, dtype=bool),
    }
    for key, mask in splits.items():
        r2_obs = r2_score(pred_observable[mask], y[mask])
        r2_pf = r2_score(pred_perfect[mask], y[mask])
        out[key] = {
            "n": int(mask.sum()),
            "r2_observable": r2_obs,
            "r2_perfect": r2_pf,
            "rho": residual_variance_ratio(r2_pf, r2_obs),
        }
    return out
