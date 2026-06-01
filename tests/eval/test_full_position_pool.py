"""Full-strength Starting-Position Pool: pre-Schupfen deal-time positions with
order-preserving hands so the Grand-Tichu Prefix is reconstructible. See ADR-0025.
"""

from tichu_engine.state import SchupfenPending, deal_for_schupfen
from tichu_eval.full_position_pool import generate_full_position_pool


def test_generate_yields_pre_schupfen_positions():
    pool = generate_full_position_pool(seed=0, n=3)
    assert len(pool) == 3
    for pos in pool:
        assert isinstance(pos.state.public.pending_decision, SchupfenPending)
        assert pos.state.public.hand_sizes == (14, 14, 14, 14)


def test_state_hands_match_deal_for_schupfen():
    # The Pool's GameState must be the same deal `deal_for_schupfen` produces,
    # so the ordered-hands reconstruction can't silently diverge.
    pool = generate_full_position_pool(seed=100, n=2)
    for i, pos in enumerate(pool):
        assert pos.state.hands == deal_for_schupfen(seed=100 + i).hands


def test_grand_prefix_is_eight_cards_subset_of_hand():
    pool = generate_full_position_pool(seed=7, n=5)
    for pos in pool:
        for seat in range(4):
            prefix = pos.grand_prefixes[seat]
            assert len(prefix) == 8
            assert prefix <= pos.state.hands[seat]


def test_generate_is_deterministic_for_seed_n():
    a = generate_full_position_pool(seed=0, n=4)
    b = generate_full_position_pool(seed=0, n=4)
    for x, y in zip(a, b):
        assert x.state.hands == y.state.hands
        assert x.grand_prefixes == y.grand_prefixes


def test_save_load_round_trips_states_and_prefixes(tmp_path):
    from tichu_eval.full_position_pool import (
        load_full_position_pool,
        save_full_position_pool,
    )

    pool = generate_full_position_pool(seed=0, n=3)
    path = tmp_path / "full_pool.parquet"
    save_full_position_pool(pool, path)
    loaded = load_full_position_pool(path)

    assert len(loaded) == len(pool)
    for a, b in zip(pool, loaded):
        assert a.state.hands == b.state.hands
        assert b.state.public.hand_sizes == (14, 14, 14, 14)
        assert isinstance(b.state.public.pending_decision, SchupfenPending)
        assert a.grand_prefixes == b.grand_prefixes
