"""Canonical action space: round-trip + structural invariants."""

import pytest

from tichu_training.action_space import (
    ACTION_SPACE_SIZE,
    ACTION_SPACE_VERSION,
    CANONICAL_ACTIONS,
    CallGrandTichu,
    CallTichu,
    DragonGive,
    Pass,
    PlayFourBomb,
    PlayFullHouse,
    PlayPair,
    PlayPairStep,
    PlaySingle,
    PlayStraight,
    PlayStraightFlushBomb,
    PlayTriple,
    SchupfenDirection,
    WishRank,
    decode,
    encode,
)


def test_version_is_pinned_to_v1():
    assert ACTION_SPACE_VERSION == "v1"


def test_size_matches_list_length():
    assert ACTION_SPACE_SIZE == len(CANONICAL_ACTIONS) > 0


def test_indices_are_unique():
    assert len(set(CANONICAL_ACTIONS)) == ACTION_SPACE_SIZE


def test_round_trip_encode_decode_every_action():
    for i, a in enumerate(CANONICAL_ACTIONS):
        assert encode(a) == i
        assert decode(i) == a


def test_round_trip_decode_encode_every_index():
    for i in range(ACTION_SPACE_SIZE):
        assert encode(decode(i)) == i


def test_non_play_actions_all_present():
    actions = set(CANONICAL_ACTIONS)
    assert Pass() in actions
    assert CallTichu() in actions
    assert CallGrandTichu() in actions
    assert SchupfenDirection("next") in actions
    assert SchupfenDirection("partner") in actions
    assert SchupfenDirection("previous") in actions
    assert WishRank(None) in actions
    for r in range(2, 15):
        assert WishRank(r) in actions
    assert DragonGive("left") in actions
    assert DragonGive("right") in actions


def test_four_bombs_one_per_rank():
    bombs = [a for a in CANONICAL_ACTIONS if isinstance(a, PlayFourBomb)]
    assert {b.rank for b in bombs} == set(range(2, 15))
    assert len(bombs) == 13


def test_pairs_have_natural_and_phoenix_per_rank():
    pairs = [a for a in CANONICAL_ACTIONS if isinstance(a, PlayPair)]
    seen = {(p.rank, p.with_phoenix) for p in pairs}
    assert seen == {(r, ph) for r in range(2, 15) for ph in (False, True)}


def test_triples_have_natural_and_phoenix_per_rank():
    triples = [a for a in CANONICAL_ACTIONS if isinstance(a, PlayTriple)]
    seen = {(t.rank, t.with_phoenix) for t in triples}
    assert seen == {(r, ph) for r in range(2, 15) for ph in (False, True)}


def test_full_house_pair_and_triple_ranks_differ():
    fhs = [a for a in CANONICAL_ACTIONS if isinstance(a, PlayFullHouse)]
    for fh in fhs:
        assert fh.triple_rank != fh.pair_rank
        assert fh.phoenix_position in ("none", "triple", "pair")


def test_straight_validity():
    straights = [a for a in CANONICAL_ACTIONS if isinstance(a, PlayStraight)]
    for s in straights:
        assert 1 <= s.start_rank <= 10
        assert 5 <= s.length <= 14
        assert s.start_rank + s.length - 1 <= 14
        if s.phoenix_position is not None:
            assert 0 <= s.phoenix_position < s.length
            if s.start_rank == 1:
                # Mahjong fills slot 0 — phoenix cannot replace it.
                assert s.phoenix_position != 0


def test_pair_step_validity():
    steps = [a for a in CANONICAL_ACTIONS if isinstance(a, PlayPairStep)]
    for ps in steps:
        assert ps.length >= 2
        assert 2 <= ps.start_rank
        assert ps.start_rank + ps.length - 1 <= 14
        if ps.phoenix_position is not None:
            assert 0 <= ps.phoenix_position < ps.length


def test_straight_flush_bomb_validity():
    bombs = [a for a in CANONICAL_ACTIONS if isinstance(a, PlayStraightFlushBomb)]
    for b in bombs:
        assert b.suit in ("jade", "sword", "pagoda", "star")
        assert b.length >= 5
        assert 2 <= b.start_rank
        assert b.start_rank + b.length - 1 <= 14


def test_singles_include_specials_and_phoenix_following():
    singles = [a for a in CANONICAL_ACTIONS if isinstance(a, PlaySingle)]
    # 52 normal cards
    naturals = [s for s in singles if s.special is None and s.phoenix_as_rank is None]
    assert len(naturals) == 52
    # Specials: mahjong, dog, dragon, phoenix-base
    specials = {s.special for s in singles if s.special is not None}
    assert specials == {"mahjong", "dog", "dragon", "phoenix"}
    # Phoenix-following: rank 2..14
    phoenix_following = [s for s in singles if s.phoenix_as_rank is not None]
    assert {s.phoenix_as_rank for s in phoenix_following} == set(range(2, 15))


def test_decode_out_of_range_raises():
    with pytest.raises(IndexError):
        decode(-1)
    with pytest.raises(IndexError):
        decode(ACTION_SPACE_SIZE)


def test_encode_unknown_action_raises():
    class NotAnAction:
        pass
    with pytest.raises(KeyError):
        encode(NotAnAction())  # type: ignore[arg-type]
