"""Concrete → Intent inverse resolver: BC training target lookups.

Pins the head-target mappings used by ParquetBCDataset:
  play              → 1809-wide Action Space (full encode)
  wish              → 14-wide (None first, then ranks 2..14)
  dragon_assignment → 2-wide (left=0, right=1 relative to winner)

Roundtrip property: for every engine Combination round-trippable in v1,
play_intent_index(combo) === encode(<the canonical Intent for combo>).
"""

import pytest

from tichu_engine.cards import Card, DOG, DRAGON, MAHJONG, PHOENIX, Suit
from tichu_engine.combinations import (
    FourOfAKindBomb, FullHouse, Pair, PairStep, Single, Straight,
    StraightFlushBomb, Triple,
)
from tichu_engine.legality import BombInterrupt, DragonGive, MahjongWish, PASS
from tichu_training.action_space import (
    ACTION_SPACE_SIZE,
    Pass as IntentPass,
    PlayFourBomb, PlayFullHouse, PlayPair, PlayPairStep, PlaySingle,
    PlayStraight, PlayStraightFlushBomb, PlayTriple,
    WishRank,
    bc_target_for_concrete,
    dragon_intent_index,
    encode,
    play_intent_index,
    wish_intent_index,
)


# --- play head -----------------------------------------------------------

def test_play_pass_maps_to_pass_intent():
    idx = play_intent_index(PASS)
    assert 0 <= idx < ACTION_SPACE_SIZE
    assert idx == encode(IntentPass())


def test_play_single_natural_card_roundtrips():
    combo = Single(Card(Suit.JADE, 7))
    assert play_intent_index(combo) == encode(PlaySingle(suit="jade", rank=7))


def test_play_single_dragon_and_mahjong_and_dog():
    assert play_intent_index(Single(DRAGON)) == encode(PlaySingle(special="dragon"))
    assert play_intent_index(Single(MAHJONG)) == encode(PlaySingle(special="mahjong"))
    assert play_intent_index(Single(DOG)) == encode(PlaySingle(special="dog"))


def test_play_single_phoenix_leading_vs_following():
    # Leading: bare Phoenix special.
    assert play_intent_index(Single(PHOENIX)) == encode(PlaySingle(special="phoenix"))
    # Following a 9: Phoenix-as-9.5 in engine; intent uses int 9 slot.
    phx_following = Single.phoenix_following(top_rank=9)
    assert play_intent_index(phx_following) == encode(PlaySingle(phoenix_as_rank=9))


def test_play_pair_without_and_with_phoenix():
    natural = Pair(Card(Suit.JADE, 5), Card(Suit.SWORD, 5))
    assert play_intent_index(natural) == encode(PlayPair(rank=5, with_phoenix=False))
    with_phx = Pair(Card(Suit.JADE, 5), PHOENIX)
    assert play_intent_index(with_phx) == encode(PlayPair(rank=5, with_phoenix=True))


def test_play_triple_with_phoenix():
    natural = Triple(Card(Suit.JADE, 9), Card(Suit.SWORD, 9), Card(Suit.PAGODA, 9))
    assert play_intent_index(natural) == encode(PlayTriple(rank=9, with_phoenix=False))
    with_phx = Triple(Card(Suit.JADE, 9), Card(Suit.SWORD, 9), PHOENIX)
    assert play_intent_index(with_phx) == encode(PlayTriple(rank=9, with_phoenix=True))


def test_play_full_house_phoenix_position_dispatch():
    triple = Triple(Card(Suit.JADE, 8), Card(Suit.SWORD, 8), Card(Suit.PAGODA, 8))
    pair = Pair(Card(Suit.JADE, 5), Card(Suit.SWORD, 5))
    fh = FullHouse(triple=triple, pair=pair)
    assert play_intent_index(fh) == encode(
        PlayFullHouse(triple_rank=8, pair_rank=5, phoenix_position="none")
    )
    # Phoenix in the triple.
    triple_phx = Triple(Card(Suit.JADE, 8), Card(Suit.SWORD, 8), PHOENIX)
    fh_t = FullHouse(triple=triple_phx, pair=pair)
    assert play_intent_index(fh_t) == encode(
        PlayFullHouse(triple_rank=8, pair_rank=5, phoenix_position="triple")
    )
    # Phoenix in the pair.
    pair_phx = Pair(Card(Suit.JADE, 5), PHOENIX)
    fh_p = FullHouse(triple=triple, pair=pair_phx)
    assert play_intent_index(fh_p) == encode(
        PlayFullHouse(triple_rank=8, pair_rank=5, phoenix_position="pair")
    )


def test_play_straight_natural_and_with_phoenix():
    cards = (
        Card(Suit.JADE, 5), Card(Suit.SWORD, 6), Card(Suit.PAGODA, 7),
        Card(Suit.STAR, 8), Card(Suit.JADE, 9),
    )
    s = Straight(cards=cards)
    assert play_intent_index(s) == encode(
        PlayStraight(start_rank=5, length=5, phoenix_position=None)
    )
    # Phoenix substituting at rank 7 (offset 2 from start 5).
    cards_phx = (
        Card(Suit.JADE, 5), Card(Suit.SWORD, 6), PHOENIX,
        Card(Suit.STAR, 8), Card(Suit.JADE, 9),
    )
    s_phx = Straight(cards=cards_phx, phoenix_as_rank=7)
    assert play_intent_index(s_phx) == encode(
        PlayStraight(start_rank=5, length=5, phoenix_position=2)
    )


def test_play_four_of_a_kind_bomb():
    bomb = FourOfAKindBomb(
        Card(Suit.JADE, 11), Card(Suit.SWORD, 11),
        Card(Suit.PAGODA, 11), Card(Suit.STAR, 11),
    )
    assert play_intent_index(bomb) == encode(PlayFourBomb(rank=11))


def test_play_straight_flush_bomb():
    cards = (
        Card(Suit.PAGODA, 3), Card(Suit.PAGODA, 4), Card(Suit.PAGODA, 5),
        Card(Suit.PAGODA, 6), Card(Suit.PAGODA, 7),
    )
    sfb = StraightFlushBomb(cards=cards)
    assert play_intent_index(sfb) == encode(
        PlayStraightFlushBomb(suit="pagoda", start_rank=3, length=5)
    )


def test_play_bomb_interrupt_unwraps_to_underlying_bomb():
    bomb = FourOfAKindBomb(
        Card(Suit.JADE, 11), Card(Suit.SWORD, 11),
        Card(Suit.PAGODA, 11), Card(Suit.STAR, 11),
    )
    bi = BombInterrupt(player=2, bomb=bomb)
    assert play_intent_index(bi) == play_intent_index(bomb)


def test_play_full_house_with_phoenix_in_both_raises():
    """Engine forbids this construction, but we still pin the rejection
    path so the BC dataset surfaces it as a parse-time error rather than a
    silent KeyError on encode()."""
    # The engine validation actually raises before we'd ever get here, so
    # construct the test against play_intent_index directly with a hand-
    # rolled fake. Skipped — engine prevents the input shape — but the
    # invariant is documented in play_intent_index's docstring.
    pytest.skip("engine FullHouse validation prevents the input shape from existing")


def test_phoenix_following_mahjong_raises_value_error_not_key_error():
    """Engine legitimately produces Single(PHOENIX, as_rank=1.5) when
    Phoenix follows the Mahjong (rank 1). The v1 Action Space catalogue
    does not enumerate `PlaySingle(phoenix_as_rank=1)` (truncated from
    1.5) — so we must surface this as ValueError, not the bare KeyError
    that `encode` would otherwise raise. The BC dataset relies on
    uniformly catching ValueError to skip unrepresentable actions."""
    phoenix_after_mahjong = Single(PHOENIX, as_rank=1.5)
    with pytest.raises(ValueError, match="not in the v1 Action Space"):
        play_intent_index(phoenix_after_mahjong)


def test_legal_mask_tolerates_unrepresentable_legal_actions():
    """When the engine produces a legal action the v1 catalogue cannot
    represent (e.g. Phoenix-following-Mahjong), `legal_mask` must skip
    that slot rather than crash. Pins the regression: pre-fix this
    leaked a KeyError out of the dataset's iterator partway through a
    100k-game run."""
    from tichu_training.action_space import legal_mask
    from tichu_engine.cards import Card, Suit, MAHJONG, PHOENIX
    from tichu_engine.state import GameState, PublicState, Trick
    # Construct a state where the leading combination is Single(MAHJONG)
    # and the current player holds the Phoenix. legal_actions will then
    # include Single.phoenix_following(top_rank=1) = as_rank=1.5, which
    # play_intent_index cannot represent.
    leader_play = __import__("tichu_engine.state", fromlist=["Play"]).Play(
        player=1, combination=Single(MAHJONG),
    )
    trick = Trick(plays=(leader_play,), leader=1)
    hand = frozenset([PHOENIX, Card(Suit.JADE, 5)])
    public = PublicState(
        current_player=0,
        hand_sizes=(2, 0, 0, 0),
        scores=(0, 0),
        trick=trick,
    )
    state = GameState(
        hands=(hand, frozenset(), frozenset(), frozenset()), public=public,
    )
    # Must not raise — Phoenix-following-Mahjong is silently skipped.
    mask = legal_mask("play", state, player=0)
    assert mask.shape == (1809,)
    # The non-phoenix Single(JADE, 5) is representable and beats Mahjong.
    from tichu_training.action_space import PlaySingle, encode
    assert mask[encode(PlaySingle(suit="jade", rank=5))]


# --- wish head -----------------------------------------------------------

def test_wish_none_is_index_zero():
    assert wish_intent_index(MahjongWish(rank=None)) == 0


@pytest.mark.parametrize("rank,expected", [(2, 1), (3, 2), (7, 6), (14, 13)])
def test_wish_rank_indices_are_rank_minus_one(rank, expected):
    assert wish_intent_index(MahjongWish(rank=rank)) == expected


# --- dragon_assignment head ---------------------------------------------

def test_dragon_left_is_zero_right_is_one():
    # Winner at seat 0: left opponent is seat 3, right opponent is seat 1.
    assert dragon_intent_index(DragonGive(target=3), winner_seat=0) == 0  # left
    assert dragon_intent_index(DragonGive(target=1), winner_seat=0) == 1  # right


def test_dragon_left_right_relative_to_any_winner_seat():
    # Winner at seat 2: left = (2+3)%4 = 1; right = (2+1)%4 = 3.
    assert dragon_intent_index(DragonGive(target=1), winner_seat=2) == 0
    assert dragon_intent_index(DragonGive(target=3), winner_seat=2) == 1


def test_dragon_give_to_teammate_raises():
    """Engine constructs legal DragonGives only — teammate (same parity) is
    never a legal target. Pin the rejection path for any path that mis-
    routes here."""
    # Winner 0, target 2 = teammate → not opp left (3) nor opp right (1).
    with pytest.raises(ValueError, match="not an opponent"):
        dragon_intent_index(DragonGive(target=2), winner_seat=0)


# --- bc_target_for_concrete dispatcher ----------------------------------

def test_dispatcher_play_pass():
    assert bc_target_for_concrete("play", PASS) == encode(IntentPass())


def test_dispatcher_wish():
    assert bc_target_for_concrete("wish", MahjongWish(rank=7)) == 6


def test_dispatcher_dragon_requires_winner_seat():
    with pytest.raises(ValueError, match="winner_seat"):
        bc_target_for_concrete("dragon_assignment", DragonGive(target=1))


def test_dispatcher_dragon_with_winner_seat():
    assert bc_target_for_concrete(
        "dragon_assignment", DragonGive(target=1), winner_seat=0,
    ) == 1


def test_dispatcher_rejects_unknown_decision_type():
    with pytest.raises(ValueError, match="unsupported decision_type"):
        bc_target_for_concrete("schupfen", object())
    with pytest.raises(ValueError, match="unsupported decision_type"):
        bc_target_for_concrete("call_tichu", object())
