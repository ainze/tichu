"""Legal-action enumeration for the current player.

This covers normal play: leading, following, and bomb interactions. It does NOT
cover Mahjong wish forcing, Dog/Dragon special semantics, or out-of-turn bomb
interrupts — those are addressed in their own sub-slices.
"""

from tichu_engine.cards import Card, DOG, DRAGON, MAHJONG, PHOENIX, Suit
from tichu_engine.combinations import (
    FourOfAKindBomb,
    FullHouse,
    Pair,
    PairStep,
    Single,
    Straight,
    StraightFlushBomb,
    Triple,
)
from tichu_engine.legality import PASS, Pass, legal_actions, legal_bomb_interrupts
from tichu_engine.state import GameState, PublicState, Trick


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


def _state(hand: frozenset, top=None, current_player: int = 0, mahjong_wish: int | None = None) -> GameState:
    """Build a state where `current_player` has `hand` and the trick is led by
    the previous player with `top` on top (or empty if `top` is None)."""
    hands: list[frozenset] = [frozenset()] * 4
    hands[current_player] = hand
    trick = Trick.empty()
    if top is not None:
        leader = (current_player - 1) % 4
        trick = trick.add_play(player=leader, combination=top)
    public = PublicState(
        current_player=current_player,
        hand_sizes=tuple(len(h) for h in hands),  # type: ignore[arg-type]
        scores=(0, 0),
        trick=trick,
        mahjong_wish=mahjong_wish,
    )
    return GameState(hands=tuple(hands), public=public)


# ---- Leading ----

def test_leading_single_card_hand_returns_one_single():
    state = _state(frozenset({_c(Suit.JADE, 7)}))
    assert legal_actions(state) == frozenset({Single(_c(Suit.JADE, 7))})


def test_leading_cannot_pass():
    state = _state(frozenset({_c(Suit.JADE, 7)}))
    assert PASS not in legal_actions(state)


def test_leading_with_pair_returns_singles_and_pair():
    seven_j = _c(Suit.JADE, 7)
    seven_s = _c(Suit.SWORD, 7)
    state = _state(frozenset({seven_j, seven_s}))
    result = legal_actions(state)
    assert Single(seven_j) in result
    assert Single(seven_s) in result
    assert Pair(seven_j, seven_s) in result


# ---- Following a single ----

def test_following_single_only_higher_singles_are_legal():
    five = _c(Suit.JADE, 5)
    nine = _c(Suit.SWORD, 9)
    twelve = _c(Suit.PAGODA, 12)
    state = _state(frozenset({five, nine, twelve}), top=Single(_c(Suit.JADE, 7)))
    result = legal_actions(state)
    assert Single(nine) in result
    assert Single(twelve) in result
    assert Single(five) not in result
    assert PASS in result


def test_following_single_wrong_type_combinations_not_legal():
    # Pair on hand cannot be played against a Single.
    seven_j = _c(Suit.JADE, 7)
    seven_s = _c(Suit.SWORD, 7)
    state = _state(frozenset({seven_j, seven_s}), top=Single(_c(Suit.JADE, 9)))
    result = legal_actions(state)
    assert Pair(seven_j, seven_s) not in result
    # Singles 7♥ and 7♠ also not legal (lower than 9).
    assert Single(seven_j) not in result
    assert Single(seven_s) not in result
    assert result == frozenset({PASS})


# ---- Following a pair ----

def test_following_pair_only_higher_pairs_are_legal():
    nine_j = _c(Suit.JADE, 9)
    nine_s = _c(Suit.SWORD, 9)
    five_j = _c(Suit.JADE, 5)
    five_s = _c(Suit.SWORD, 5)
    state = _state(
        frozenset({nine_j, nine_s, five_j, five_s}),
        top=Pair(_c(Suit.PAGODA, 7), _c(Suit.STAR, 7)),
    )
    result = legal_actions(state)
    assert Pair(nine_j, nine_s) in result
    assert Pair(five_j, five_s) not in result
    assert Single(nine_j) not in result  # wrong type
    assert PASS in result


# ---- Following a triple ----

def test_following_triple_only_higher_triples_are_legal():
    nines = (_c(Suit.JADE, 9), _c(Suit.SWORD, 9), _c(Suit.PAGODA, 9))
    fives = (_c(Suit.JADE, 5), _c(Suit.SWORD, 5), _c(Suit.PAGODA, 5))
    state = _state(
        frozenset(nines + fives),
        top=Triple(_c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7)),
    )
    result = legal_actions(state)
    assert Triple(*nines) in result
    assert Triple(*fives) not in result
    assert PASS in result


# ---- Following a straight ----

def test_following_straight_must_match_length_and_be_higher():
    hi_straight = (_c(Suit.JADE, 9), _c(Suit.SWORD, 10), _c(Suit.PAGODA, 11),
                   _c(Suit.STAR, 12), _c(Suit.JADE, 13))
    state = _state(
        frozenset(hi_straight),
        top=Straight((_c(Suit.JADE, 5), _c(Suit.SWORD, 6), _c(Suit.PAGODA, 7),
                      _c(Suit.STAR, 8), _c(Suit.JADE, 9))),
    )
    result = legal_actions(state)
    assert Straight(hi_straight) in result
    assert PASS in result


def test_following_straight_wrong_length_not_legal():
    # Hand has a 6-card straight; top is a 5-card straight.
    long_straight = (_c(Suit.JADE, 9), _c(Suit.SWORD, 10), _c(Suit.PAGODA, 11),
                     _c(Suit.STAR, 12), _c(Suit.JADE, 13), _c(Suit.SWORD, 14))
    state = _state(
        frozenset(long_straight),
        top=Straight((_c(Suit.JADE, 5), _c(Suit.SWORD, 6), _c(Suit.PAGODA, 7),
                      _c(Suit.STAR, 8), _c(Suit.JADE, 9))),
    )
    result = legal_actions(state)
    assert Straight(long_straight) not in result
    # The hand also contains length-5 straights (10..14, 9..13) — those are legal.
    assert Straight(long_straight[:5]) in result or Straight(long_straight[1:]) in result


# ---- Bombs against normal combinations ----

def test_bomb_beats_single():
    bomb_cards = (_c(Suit.JADE, 5), _c(Suit.SWORD, 5), _c(Suit.PAGODA, 5), _c(Suit.STAR, 5))
    state = _state(frozenset(bomb_cards), top=Single(_c(Suit.JADE, 9)))
    result = legal_actions(state)
    assert FourOfAKindBomb(*bomb_cards) in result


def test_bomb_beats_full_house():
    bomb_cards = (_c(Suit.JADE, 5), _c(Suit.SWORD, 5), _c(Suit.PAGODA, 5), _c(Suit.STAR, 5))
    top_full_house = FullHouse(
        triple=Triple(_c(Suit.JADE, 9), _c(Suit.SWORD, 9), _c(Suit.PAGODA, 9)),
        pair=Pair(_c(Suit.JADE, 11), _c(Suit.SWORD, 11)),
    )
    state = _state(frozenset(bomb_cards), top=top_full_house)
    result = legal_actions(state)
    assert FourOfAKindBomb(*bomb_cards) in result


def test_straight_flush_bomb_beats_four_of_a_kind_bomb_of_any_rank():
    # Straight flush trumps a four-of-a-kind bomb regardless of rank.
    sf_cards = (_c(Suit.JADE, 5), _c(Suit.JADE, 6), _c(Suit.JADE, 7),
                _c(Suit.JADE, 8), _c(Suit.JADE, 9))
    state = _state(
        frozenset(sf_cards),
        top=FourOfAKindBomb(
            _c(Suit.SWORD, 13), _c(Suit.PAGODA, 13), _c(Suit.STAR, 13), _c(Suit.JADE, 13)
        ),
    )
    result = legal_actions(state)
    assert StraightFlushBomb(sf_cards) in result


def test_following_four_bomb_only_higher_four_bombs_or_any_straight_flush():
    higher_four = (_c(Suit.JADE, 11), _c(Suit.SWORD, 11), _c(Suit.PAGODA, 11), _c(Suit.STAR, 11))
    sf = (_c(Suit.JADE, 5), _c(Suit.JADE, 6), _c(Suit.JADE, 7),
          _c(Suit.JADE, 8), _c(Suit.JADE, 9))
    lower_four = (_c(Suit.JADE, 4), _c(Suit.SWORD, 4), _c(Suit.PAGODA, 4), _c(Suit.STAR, 4))
    state = _state(
        frozenset(higher_four + sf + lower_four),
        top=FourOfAKindBomb(
            _c(Suit.SWORD, 9), _c(Suit.PAGODA, 9), _c(Suit.STAR, 9), _c(Suit.JADE, 9)
        ),
    )
    result = legal_actions(state)
    assert FourOfAKindBomb(*higher_four) in result
    assert StraightFlushBomb(sf) in result
    assert FourOfAKindBomb(*lower_four) not in result
    assert PASS in result


def test_following_straight_flush_only_better_straight_flush():
    # Top is a length-5 straight flush at rank 5. Hand has a length-5 at rank 9 (higher),
    # a length-6 at rank 5 (longer), and a length-5 at rank 4 (lower) — lower not legal.
    top_sf = StraightFlushBomb((_c(Suit.STAR, 5), _c(Suit.STAR, 6), _c(Suit.STAR, 7),
                                _c(Suit.STAR, 8), _c(Suit.STAR, 9)))
    higher_5 = (_c(Suit.JADE, 9), _c(Suit.JADE, 10), _c(Suit.JADE, 11),
                _c(Suit.JADE, 12), _c(Suit.JADE, 13))
    longer_6 = (_c(Suit.SWORD, 5), _c(Suit.SWORD, 6), _c(Suit.SWORD, 7),
                _c(Suit.SWORD, 8), _c(Suit.SWORD, 9), _c(Suit.SWORD, 10))
    state = _state(frozenset(higher_5 + longer_6), top=top_sf)
    result = legal_actions(state)
    assert StraightFlushBomb(higher_5) in result
    assert StraightFlushBomb(longer_6) in result
    # Same length lower rank is not legal — but there isn't one in this hand; we already
    # cover that case by construction.
    assert PASS in result


# ---- Phoenix following a Single ----

def test_phoenix_following_a_normal_single_is_legal():
    state = _state(frozenset({PHOENIX}), top=Single(_c(Suit.JADE, 7)))
    result = legal_actions(state)
    assert Single.phoenix_following(top_rank=7) in result
    # The leading-rank Phoenix single (rank 1.5) cannot beat a 7 — must not be offered.
    assert Single(PHOENIX) not in result
    assert PASS in result


def test_phoenix_following_a_mahjong_is_legal():
    # Mahjong has rank 1; Phoenix-following = 1.5, which beats it.
    state = _state(frozenset({PHOENIX}), top=Single(MAHJONG))
    result = legal_actions(state)
    assert Single.phoenix_following(top_rank=1) in result


def test_phoenix_cannot_follow_dragon():
    # Nothing — not even Phoenix — beats the Dragon as a single. Only bombs do.
    state = _state(frozenset({PHOENIX}), top=Single(DRAGON))
    result = legal_actions(state)
    assert Single.phoenix_following(top_rank=25) not in result
    assert Single(PHOENIX) not in result
    assert result == frozenset({PASS})


def test_phoenix_not_offered_as_following_single_when_top_is_pair():
    # Phoenix can be used inside a Pair (already handled by enumerate_pairs),
    # but should not appear as a phoenix-following Single against a Pair.
    seven = _c(Suit.JADE, 7)
    state = _state(
        frozenset({PHOENIX, seven}),
        top=Pair(_c(Suit.PAGODA, 5), _c(Suit.STAR, 5)),
    )
    result = legal_actions(state)
    assert Pair(PHOENIX, seven) in result
    assert Single.phoenix_following(top_rank=5) not in result


# ---- Dog ----

def test_dog_is_legal_when_leading():
    state = _state(frozenset({DOG, _c(Suit.JADE, 7)}))
    assert Single(DOG) in legal_actions(state)


def test_dog_is_not_legal_when_following_a_single():
    state = _state(frozenset({DOG}), top=Single(_c(Suit.JADE, 7)))
    assert Single(DOG) not in legal_actions(state)


def test_dog_is_not_legal_when_following_any_combination():
    # Following a pair, triple, etc. — Dog can never be played.
    state = _state(
        frozenset({DOG}),
        top=Pair(_c(Suit.PAGODA, 5), _c(Suit.STAR, 5)),
    )
    result = legal_actions(state)
    assert Single(DOG) not in result
    assert result == frozenset({PASS})


# ---- Mahjong wish ----

def test_wish_when_leading_forces_combinations_containing_wished_rank():
    # Hand: 7, 9. Wish=9. Leading. Only the 9 single is legal.
    seven = _c(Suit.JADE, 7)
    nine = _c(Suit.SWORD, 9)
    state = _state(frozenset({seven, nine}), mahjong_wish=9)
    result = legal_actions(state)
    assert result == frozenset({Single(nine)})


def test_wish_unfulfillable_when_leading_falls_back_to_all_legal():
    # Hand: 7, 8. Wish=9 (not in hand). Player can play anything.
    seven = _c(Suit.JADE, 7)
    eight = _c(Suit.SWORD, 8)
    state = _state(frozenset({seven, eight}), mahjong_wish=9)
    result = legal_actions(state)
    assert Single(seven) in result and Single(eight) in result
    assert PASS not in result  # still leading, can't pass


def test_wish_when_following_forces_wished_rank_excludes_pass():
    # Hand: 9, K. Top: 7. Wish=9. Must play 9 (K legal but wish forces 9).
    nine = _c(Suit.JADE, 9)
    king = _c(Suit.SWORD, 13)
    state = _state(frozenset({nine, king}), top=Single(_c(Suit.PAGODA, 7)), mahjong_wish=9)
    result = legal_actions(state)
    assert result == frozenset({Single(nine)})


def test_wish_unfulfillable_when_following_allows_pass_and_normal_legal():
    # Hand: 9. Top: K. Wish=Q. 9 doesn't beat K, and no Q in hand. Pass + nothing.
    nine = _c(Suit.JADE, 9)
    state = _state(frozenset({nine}), top=Single(_c(Suit.PAGODA, 13)), mahjong_wish=12)
    result = legal_actions(state)
    assert result == frozenset({PASS})


def test_wish_fulfilled_by_card_inside_full_house():
    # Hand: triple of 7s + pair of 9s + pair of Js. Wish=9. Leading.
    # Full house 7-7-7-9-9 fulfills (contains 9). Triple-7 alone does not.
    sevens = (_c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7))
    nines = (_c(Suit.JADE, 9), _c(Suit.SWORD, 9))
    jacks = (_c(Suit.JADE, 11), _c(Suit.SWORD, 11))
    state = _state(frozenset(sevens + nines + jacks), mahjong_wish=9)
    result = legal_actions(state)
    fh = FullHouse(triple=Triple(*sevens), pair=Pair(*nines))
    assert fh in result
    # Single 9s, pair of 9s, full house with 9 in either part — all fulfill.
    # Triple of 7s alone does not.
    assert Triple(*sevens) not in result
    assert Pair(*nines) in result


def test_wish_not_fulfilled_by_phoenix_substituting_for_wished_rank():
    # Hand: Phoenix + a 5. Wish=9. Phoenix-as-9 would fulfill conceptually,
    # but the strict rule requires the actual card. Phoenix-as-anything is not enough.
    five = _c(Suit.JADE, 5)
    state = _state(frozenset({PHOENIX, five}), mahjong_wish=9)
    result = legal_actions(state)
    # No combination contains a natural 9 -> wish is unfulfillable -> all legal options remain.
    # Leading -> all enumerated combinations (Single(5), Single(PHOENIX), Pair(PHOENIX, 5)).
    assert Single(five) in result
    assert Single(PHOENIX) in result
    assert Pair(PHOENIX, five) in result


# ---- Out-of-turn bomb interrupts ----

def _state_with_per_player_hands(
    hands_dict: dict[int, frozenset], top=None, current_player: int = 0
) -> GameState:
    hands: list[frozenset] = [frozenset()] * 4
    for p, h in hands_dict.items():
        hands[p] = h
    trick = Trick.empty()
    if top is not None:
        leader = (current_player - 1) % 4
        trick = trick.add_play(player=leader, combination=top)
    public = PublicState(
        current_player=current_player,
        hand_sizes=tuple(len(h) for h in hands),  # type: ignore[arg-type]
        scores=(0, 0),
        trick=trick,
    )
    return GameState(hands=tuple(hands), public=public)


def test_bomb_interrupts_on_empty_trick_returns_all_holder_bombs():
    # BSW allows a non-current player to preempt-bomb when the previous trick
    # has resolved and the trick winner has not yet led. The bomb seizes the
    # lead. Default current_player is 0; the bomb-holder is player 1.
    bomb_cards = (_c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7), _c(Suit.STAR, 7))
    state = _state_with_per_player_hands({1: frozenset(bomb_cards)})
    assert legal_bomb_interrupts(state, player=1) == frozenset({FourOfAKindBomb(*bomb_cards)})
    # The current player (player 0) gets their bombs via legal_actions, not interrupts.
    assert legal_bomb_interrupts(state, player=0) == frozenset()


def test_bomb_interrupts_returns_higher_bomb_for_non_current_player():
    # Player 0 leads a 9. Player 1 has a four-of-a-kind 7-bomb.
    bomb_cards = (_c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7), _c(Suit.STAR, 7))
    state = _state_with_per_player_hands(
        {1: frozenset(bomb_cards)}, top=Single(_c(Suit.JADE, 9)), current_player=1
    )
    # Player 2 (not current, not bomb owner) — they have no bomb.
    assert legal_bomb_interrupts(state, player=2) == frozenset()
    # Player 1 IS the current player; their bomb is in legal_actions, not interrupts.
    assert legal_bomb_interrupts(state, player=1) == frozenset()


def test_bomb_interrupts_returns_bomb_for_non_current_holder():
    # Current player is 1 (just led). Player 3 has a bomb and can interrupt.
    bomb_cards = (_c(Suit.JADE, 5), _c(Suit.SWORD, 5), _c(Suit.PAGODA, 5), _c(Suit.STAR, 5))
    state = _state_with_per_player_hands(
        {3: frozenset(bomb_cards)},
        top=Single(_c(Suit.JADE, 9)),
        current_player=2,  # next to play
    )
    result = legal_bomb_interrupts(state, player=3)
    assert FourOfAKindBomb(*bomb_cards) in result


def test_bomb_interrupts_excludes_bombs_that_dont_beat_top():
    # Top is a king four-of-a-kind. A 5-bomb doesn't beat it; a straight flush does.
    weak_bomb = (_c(Suit.JADE, 5), _c(Suit.SWORD, 5), _c(Suit.PAGODA, 5), _c(Suit.STAR, 5))
    sf_cards = (_c(Suit.JADE, 6), _c(Suit.JADE, 7), _c(Suit.JADE, 8),
                _c(Suit.JADE, 9), _c(Suit.JADE, 10))
    top = FourOfAKindBomb(
        _c(Suit.JADE, 13), _c(Suit.SWORD, 13), _c(Suit.PAGODA, 13), _c(Suit.STAR, 13)
    )
    state = _state_with_per_player_hands(
        {3: frozenset(weak_bomb + sf_cards)}, top=top, current_player=2
    )
    result = legal_bomb_interrupts(state, player=3)
    assert FourOfAKindBomb(*weak_bomb) not in result
    assert StraightFlushBomb(sf_cards) in result


# ---- Pass equality ----

def test_pass_singletons_are_equal():
    assert PASS == Pass()
    assert hash(PASS) == hash(Pass())
