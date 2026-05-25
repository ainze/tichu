"""Fixed seeded deal pool — generate, persist, load."""

from tichu_engine.cards import MAHJONG
from tichu_engine.state import GameState
from tichu_eval.starting_position_pool import (
    generate_starting_position_pool,
    load_starting_position_pool,
    save_starting_position_pool,
)


def test_generate_is_deterministic():
    a = generate_starting_position_pool(seed=0, n=5)
    b = generate_starting_position_pool(seed=0, n=5)
    assert len(a) == 5
    assert len(b) == 5
    for da, db in zip(a, b):
        assert isinstance(da, GameState)
        assert da.hands == db.hands
        assert da.public.current_player == db.public.current_player


def test_generate_distinct_seeds_produce_distinct_deals():
    a = generate_starting_position_pool(seed=0, n=3)
    b = generate_starting_position_pool(seed=999, n=3)
    # Vanishingly unlikely to collide.
    assert any(da.hands != db.hands for da, db in zip(a, b))


def test_each_deal_has_14_cards_per_player_and_full_56_cards():
    deals = generate_starting_position_pool(seed=1, n=5)
    for deal in deals:
        sizes = tuple(len(h) for h in deal.hands)
        assert sizes == (14, 14, 14, 14)
        union = set().union(*deal.hands)
        assert len(union) == 56


def test_starting_player_holds_mahjong():
    for deal in generate_starting_position_pool(seed=2, n=10):
        starter = deal.public.current_player
        assert MAHJONG in deal.hands[starter]


def test_save_load_roundtrips(tmp_path):
    deals = generate_starting_position_pool(seed=3, n=4)
    path = tmp_path / "pool.parquet"
    save_starting_position_pool(deals, path)
    assert path.exists()
    loaded = load_starting_position_pool(path)
    assert len(loaded) == len(deals)
    for original, restored in zip(deals, loaded):
        assert original.hands == restored.hands
        assert original.public.current_player == restored.public.current_player
        assert original.public.hand_sizes == restored.public.hand_sizes
