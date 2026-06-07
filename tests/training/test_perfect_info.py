"""Perfect-Info feature + gate helpers (ADR-0033).

The Perfect-Info Critic's train-time input: the 224-dim observable Feature
Vector + the three opponents' *true* hands on the Belief grid (next/partner/
previous), and the value-ceiling gate helpers built on it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from tichu_engine.state import deal_initial_state
from tichu_training.card_slots import card_slot
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM, featurize
from tichu_training.perfect_info import (
    PERFECT_INFO_DIM,
    featurize_perfect_info,
    is_call_live,
    r2_score,
    residual_variance_ratio,
    split_rho,
)


def _grid(pi):
    return pi[FEATURIZER_OUTPUT_DIM:].reshape(3, 56)


def test_perfect_info_shape_and_observable_prefix():
    state = deal_initial_state(seed=0)
    seat = state.public.current_player
    pi = featurize_perfect_info(state, seat)

    assert PERFECT_INFO_DIM == FEATURIZER_OUTPUT_DIM + 3 * 56
    assert pi.shape == (PERFECT_INFO_DIM,)
    assert pi.dtype == np.float32
    # The first 224 dims are exactly the observable Feature Vector.
    np.testing.assert_array_equal(
        pi[:FEATURIZER_OUTPUT_DIM], featurize(state.private_view(seat))
    )


def test_opponent_grid_matches_true_hands_in_relative_order():
    state = deal_initial_state(seed=3)
    seat = 1
    grid = _grid(featurize_perfect_info(state, seat))
    rel_seats = ((seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4)
    for opp_idx, opp_seat in enumerate(rel_seats):
        got = {s for s in range(56) if grid[opp_idx, s]}
        want = {card_slot(c) for c in state.hands[opp_seat]}
        assert got == want


def test_relative_ordering_is_seat_invariant():
    # From any seat, the partner (seat+2) lands in the partner row (index 1);
    # the same physical hand moves to its correct relative row as the seat moves.
    state = deal_initial_state(seed=7)
    for seat in range(4):
        grid = _grid(featurize_perfect_info(state, seat))
        partner = (seat + 2) % 4
        got = {s for s in range(56) if grid[1, s]}
        want = {card_slot(c) for c in state.hands[partner]}
        assert got == want


def test_own_cards_absent_from_opponent_grid():
    state = deal_initial_state(seed=11)
    seat = 2
    grid = _grid(featurize_perfect_info(state, seat))
    own = {card_slot(c) for c in state.hands[seat]}
    present = {s for s in range(56) if grid.any(axis=0)[s]}
    assert not (own & present)


def test_is_call_live_tracks_outstanding_callers():
    pub = deal_initial_state(seed=0).public
    assert not is_call_live(pub, 0)
    assert is_call_live(replace(pub, tichu_callers=frozenset({0})), 0)
    assert is_call_live(replace(pub, grand_tichu_callers=frozenset({2})), 2)
    # A caller elsewhere does not make THIS seat call-live.
    assert not is_call_live(replace(pub, tichu_callers=frozenset({1})), 0)


def test_r2_score_and_residual_variance_ratio():
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert r2_score(y, y) == 1.0                       # perfect prediction
    assert abs(r2_score(np.full_like(y, y.mean()), y)) < 1e-6  # mean -> 0
    # rho = (1 - r2_perfect) / (1 - r2_observable): perfect info halves residual.
    assert residual_variance_ratio(0.5, 0.0) == 0.5
    assert residual_variance_ratio(0.75, 0.5) == 0.5


def test_split_rho_partitions_by_call_live_and_computes_ratio():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.standard_normal(n).astype(np.float32)
    pred_obs = y + rng.standard_normal(n).astype(np.float32)            # noisy
    pred_perfect = y + 0.1 * rng.standard_normal(n).astype(np.float32)  # much better
    call_live = np.zeros(n, dtype=bool)
    call_live[:50] = True

    out = split_rho(y, pred_obs, pred_perfect, call_live)

    assert set(out) == {"overall", "call_live", "rest"}
    for key, mask in (("overall", np.ones(n, bool)),
                      ("call_live", call_live),
                      ("rest", ~call_live)):
        assert out[key]["n"] == int(mask.sum())
        exp_obs = r2_score(pred_obs[mask], y[mask])
        exp_pf = r2_score(pred_perfect[mask], y[mask])
        assert out[key]["r2_observable"] == exp_obs
        assert out[key]["r2_perfect"] == exp_pf
        assert out[key]["rho"] == residual_variance_ratio(exp_pf, exp_obs)
    # Perfect info slashes residual variance -> rho well below 1.
    assert out["overall"]["rho"] < 1.0
