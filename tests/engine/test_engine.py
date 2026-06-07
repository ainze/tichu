"""Engine state transitions: step() applies actions and resolves tricks.

Out-of-turn bomb interrupts, Dog-to-partner, Dragon-give, Mahjong-wish
declaration, and point scoring are covered in their own sub-slices.
"""

import pytest

from tichu_engine.cards import Card, DOG, DRAGON, MAHJONG, PHOENIX, Suit
from tichu_engine.combinations import FourOfAKindBomb, Pair, Single
from tichu_engine.engine import step
from tichu_engine.legality import BombInterrupt, DragonGive, MahjongWish, PASS, legal_actions
from tichu_engine.state import (
    DragonGivePending,
    GameState,
    MahjongWishPending,
    PublicState,
    Trick,
)


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


def _state(hands_dict: dict[int, frozenset], top=None, current_player: int = 0) -> GameState:
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


# ---- Playing a combination ----

def test_step_play_removes_cards_from_hand():
    seven = _c(Suit.JADE, 7)
    nine = _c(Suit.SWORD, 9)
    state = _state({0: frozenset({seven, nine}), 1: frozenset({_c(Suit.STAR, 10)})})
    next_state, _, _, _ = step(state, Single(seven))
    assert next_state.hands[0] == frozenset({nine})
    assert next_state.public.hand_sizes[0] == 1


def test_step_play_records_combination_on_trick():
    seven = _c(Suit.JADE, 7)
    state = _state(
        {
            0: frozenset({seven, _c(Suit.SWORD, 3)}),
            1: frozenset({_c(Suit.STAR, 10)}),
            2: frozenset({_c(Suit.PAGODA, 11)}),
            3: frozenset({_c(Suit.SWORD, 12)}),
        },
    )
    next_state, _, _, _ = step(state, Single(seven))
    assert next_state.public.trick.top_combination == Single(seven)
    assert next_state.public.trick.leader == 0


def test_step_play_advances_to_next_player():
    seven = _c(Suit.JADE, 7)
    state = _state(
        {0: frozenset({seven, _c(Suit.SWORD, 5)}), 1: frozenset({_c(Suit.STAR, 10)})},
    )
    next_state, _, _, _ = step(state, Single(seven))
    assert next_state.public.current_player == 1


def test_step_play_pair_removes_both_cards():
    seven_j = _c(Suit.JADE, 7)
    seven_s = _c(Suit.SWORD, 7)
    state = _state(
        {0: frozenset({seven_j, seven_s}), 1: frozenset({_c(Suit.STAR, 10), _c(Suit.JADE, 10)})},
    )
    next_state, _, _, _ = step(state, Pair(seven_j, seven_s))
    assert next_state.hands[0] == frozenset()
    assert next_state.public.hand_sizes[0] == 0


# ---- Passing ----

def test_step_pass_advances_to_next_player():
    # Player 0 led a 7. Player 1 is current and chooses to pass; turn goes to player 2.
    # All players still hold cards so the trick does not resolve yet.
    state = _state(
        {
            0: frozenset({_c(Suit.JADE, 14)}),
            1: frozenset({_c(Suit.STAR, 5)}),
            2: frozenset({_c(Suit.PAGODA, 11)}),
            3: frozenset({_c(Suit.SWORD, 12)}),
        },
        top=Single(_c(Suit.JADE, 7)),
        current_player=1,
    )
    next_state, _, _, _ = step(state, PASS)
    assert next_state.public.current_player == 2


def test_step_pass_keeps_top_combination():
    top_play = Single(_c(Suit.JADE, 7))
    state = _state(
        {0: frozenset(), 1: frozenset({_c(Suit.STAR, 5)}), 2: frozenset({_c(Suit.JADE, 11)})},
        top=top_play,
        current_player=1,
    )
    next_state, _, _, _ = step(state, PASS)
    assert next_state.public.trick.top_combination == top_play
    assert next_state.public.trick.leader == 0


def test_step_pass_records_pass_in_trick():
    state = _state(
        {0: frozenset(), 1: frozenset({_c(Suit.STAR, 5)}), 2: frozenset({_c(Suit.JADE, 11)})},
        top=Single(_c(Suit.JADE, 7)),
        current_player=1,
    )
    next_state, _, _, _ = step(state, PASS)
    assert 1 in next_state.public.trick.passes


# ---- Trick resolution ----

def test_trick_resolves_after_all_others_pass():
    # Leader is 0 (just played a 7). Players 1, 2, 3 all pass — trick resolves.
    # Start state: player 1 is current, others have passed.
    leader_card = _c(Suit.JADE, 7)
    state = GameState(
        hands=(
            frozenset({_c(Suit.JADE, 14)}),  # leader has remaining card
            frozenset({_c(Suit.STAR, 5)}),
            frozenset({_c(Suit.STAR, 6)}),
            frozenset({_c(Suit.STAR, 3)}),
        ),
        public=PublicState(
            current_player=3,
            hand_sizes=(1, 1, 1, 1),
            scores=(0, 0),
            trick=Trick(
                plays=(),
                leader=0,
                passes=frozenset({1, 2}),
            ).add_play(player=0, combination=Single(leader_card)).add_pass(player=1).add_pass(player=2),
        ),
    )
    # Player 3 passes; trick should resolve back to player 0.
    next_state, _, _, _ = step(state, PASS)
    assert next_state.public.current_player == 0
    assert next_state.public.trick.top_combination is None  # trick cleared
    assert next_state.public.trick.leader is None


def test_trick_advancement_skips_players_with_empty_hands():
    # Player 1 is out (empty hand). Player 0 plays, turn should jump to player 2.
    seven = _c(Suit.JADE, 7)
    state = GameState(
        hands=(
            frozenset({seven, _c(Suit.SWORD, 5)}),
            frozenset(),  # player 1 is out
            frozenset({_c(Suit.STAR, 10)}),
            frozenset({_c(Suit.PAGODA, 3)}),
        ),
        public=PublicState(
            current_player=0,
            hand_sizes=(2, 0, 1, 1),
            scores=(0, 0),
            trick=Trick.empty(),
        ),
    )
    next_state, _, _, _ = step(state, Single(seven))
    assert next_state.public.current_player == 2


# ---- Validation ----

def test_step_rejects_illegal_action():
    seven = _c(Suit.JADE, 7)
    state = _state({0: frozenset({seven})})
    # Trying to play a single that isn't in our hand.
    with pytest.raises(ValueError):
        step(state, Single(_c(Suit.SWORD, 9)))


def test_step_rejects_pass_when_leading():
    state = _state({0: frozenset({_c(Suit.JADE, 7)})})  # leading, can't pass
    with pytest.raises(ValueError):
        step(state, PASS)


# ---- Dog ----

def test_step_dog_passes_lead_to_partner():
    # Player 0 plays Dog while leading; partner is player 2.
    state = _state(
        {
            0: frozenset({DOG, _c(Suit.JADE, 7)}),
            1: frozenset({_c(Suit.STAR, 5)}),
            2: frozenset({_c(Suit.PAGODA, 11)}),
            3: frozenset({_c(Suit.SWORD, 12)}),
        }
    )
    next_state, _, _, _ = step(state, Single(DOG))
    assert next_state.public.current_player == 2
    # Trick is cleared — no top combination, no leader.
    assert next_state.public.trick.top_combination is None
    assert next_state.public.trick.leader is None


def test_step_dog_removes_dog_from_hand():
    state = _state(
        {
            0: frozenset({DOG, _c(Suit.JADE, 7)}),
            1: frozenset({_c(Suit.STAR, 5)}),
            2: frozenset({_c(Suit.PAGODA, 11)}),
            3: frozenset({_c(Suit.SWORD, 12)}),
        }
    )
    next_state, _, _, _ = step(state, Single(DOG))
    assert DOG not in next_state.hands[0]
    assert next_state.public.hand_sizes[0] == 1


def test_step_dog_skips_partner_if_partner_is_out():
    # Player 1 leads with Dog; partner (player 3) has empty hand;
    # lead should pass to next clockwise player with cards, which is player 0.
    state = _state(
        {
            0: frozenset({_c(Suit.JADE, 7)}),
            1: frozenset({DOG, _c(Suit.SWORD, 5)}),
            2: frozenset({_c(Suit.PAGODA, 11)}),
            3: frozenset(),  # partner out
        },
        current_player=1,
    )
    next_state, _, _, _ = step(state, Single(DOG))
    assert next_state.public.current_player == 0


# ---- Point scoring on trick resolution ----

def _resolved_state_after_trick(top_play_player: int, top_combination) -> GameState:
    """Construct a state where `top_play_player` led `top_combination`, the next
    two players have already passed, and the last player is about to pass — so
    one more PASS resolves the trick to top_play_player."""
    hands_dict = {p: frozenset({_c(Suit.JADE, 2)}) for p in range(4)}
    hands_dict[top_play_player] = frozenset()
    hands: list[frozenset] = [hands_dict[p] for p in range(4)]
    trick = Trick.empty().add_play(player=top_play_player, combination=top_combination)
    # Players in order after top_play_player pass first.
    a = (top_play_player + 1) % 4
    b = (top_play_player + 2) % 4
    c = (top_play_player + 3) % 4
    trick = trick.add_pass(a).add_pass(b)
    public = PublicState(
        current_player=c,
        hand_sizes=tuple(len(h) for h in hands),  # type: ignore[arg-type]
        scores=(0, 0),
        trick=trick,
    )
    return GameState(hands=tuple(hands), public=public)


def test_trick_resolution_with_no_point_cards_does_not_change_score():
    state = _resolved_state_after_trick(0, Single(_c(Suit.JADE, 7)))
    next_state, _, _, _ = step(state, PASS)
    assert next_state.public.scores == (0, 0)


def test_trick_resolution_with_a_five_scores_five_to_winner_team():
    # Player 0 (team 0) wins a trick containing a 5.
    state = _resolved_state_after_trick(0, Single(_c(Suit.JADE, 5)))
    next_state, _, _, _ = step(state, PASS)
    assert next_state.public.scores == (5, 0)


def test_trick_resolution_with_a_ten_scores_ten():
    state = _resolved_state_after_trick(0, Single(_c(Suit.JADE, 10)))
    next_state, _, _, _ = step(state, PASS)
    assert next_state.public.scores == (10, 0)


def test_trick_resolution_with_a_king_scores_ten():
    state = _resolved_state_after_trick(0, Single(_c(Suit.JADE, 13)))
    next_state, _, _, _ = step(state, PASS)
    assert next_state.public.scores == (10, 0)


def test_trick_resolution_with_dragon_defers_to_pending_decision():
    # Dragon-led trick: scoring is deferred until the winner chooses an opponent.
    state = _resolved_state_after_trick(0, Single(DRAGON))
    next_state, _, _, _ = step(state, PASS)
    assert next_state.public.scores == (0, 0)
    assert next_state.public.pending_decision == DragonGivePending(winner=0, points=25)


def test_trick_resolution_with_phoenix_scores_minus_twenty_five():
    # Phoenix as a leading single — Phoenix-leading rank is 1.5, not enough to
    # naturally lead alone, but it's a legal lead. Construct directly.
    state = _resolved_state_after_trick(0, Single(PHOENIX))
    next_state, _, _, _ = step(state, PASS)
    assert next_state.public.scores == (-25, 0)


def test_trick_resolution_credits_correct_team():
    # Player 1 (team 1) wins a trick with a 10.
    state = _resolved_state_after_trick(1, Single(_c(Suit.JADE, 10)))
    next_state, _, _, _ = step(state, PASS)
    assert next_state.public.scores == (0, 10)


# ---- Dragon-give (G3) ----

def test_dragon_give_legal_actions_are_the_two_opponents():
    # Winner is player 0 (team 0); opponents are players 1 and 3.
    state = _resolved_state_after_trick(0, Single(DRAGON))
    after_resolve, _, _, _ = step(state, PASS)
    options = legal_actions(after_resolve)
    assert options == frozenset({DragonGive(target=1), DragonGive(target=3)})


def test_dragon_give_credits_target_team_with_points():
    state = _resolved_state_after_trick(0, Single(DRAGON))
    after_resolve, _, _, _ = step(state, PASS)
    final, _, _, _ = step(after_resolve, DragonGive(target=1))
    # 25 points to team 1.
    assert final.public.scores == (0, 25)
    assert final.public.pending_decision is None


def test_dragon_give_to_other_opponent_credits_other_team_still():
    # Whether the winner picks player 1 or player 3, both are on team 1.
    state = _resolved_state_after_trick(0, Single(DRAGON))
    after_resolve, _, _, _ = step(state, PASS)
    final, _, _, _ = step(after_resolve, DragonGive(target=3))
    assert final.public.scores == (0, 25)


def test_dragon_give_cannot_target_partner():
    state = _resolved_state_after_trick(0, Single(DRAGON))
    after_resolve, _, _, _ = step(state, PASS)
    # Player 0's partner is player 2 — not a legal target.
    options = legal_actions(after_resolve)
    assert DragonGive(target=0) not in options
    assert DragonGive(target=2) not in options


# ---- Mahjong wish (G4) ----

def test_playing_mahjong_sets_wish_pending():
    state = _state({
        0: frozenset({MAHJONG, _c(Suit.JADE, 7)}),
        1: frozenset({_c(Suit.STAR, 10)}),
        2: frozenset({_c(Suit.PAGODA, 11)}),
        3: frozenset({_c(Suit.SWORD, 12)}),
    })
    next_state, _, _, _ = step(state, Single(MAHJONG))
    assert next_state.public.pending_decision == MahjongWishPending(player=0)
    # Turn has not advanced yet.
    assert next_state.public.current_player == 0


def test_wish_pending_legal_actions_are_ranks_2_through_14_and_none():
    state = _state({
        0: frozenset({MAHJONG, _c(Suit.JADE, 7)}),
        1: frozenset({_c(Suit.STAR, 10)}),
        2: frozenset({_c(Suit.PAGODA, 11)}),
        3: frozenset({_c(Suit.SWORD, 12)}),
    })
    after_mahjong, _, _, _ = step(state, Single(MAHJONG))
    options = legal_actions(after_mahjong)
    expected = {MahjongWish(rank=r) for r in range(2, 15)} | {MahjongWish(rank=None)}
    assert options == expected


def test_mahjong_wish_with_rank_sets_wish_and_advances_turn():
    state = _state(
        {
            0: frozenset({MAHJONG, _c(Suit.JADE, 7)}),
            1: frozenset({_c(Suit.SWORD, 9)}),
            2: frozenset({_c(Suit.PAGODA, 10)}),
            3: frozenset({_c(Suit.STAR, 11)}),
        }
    )
    after_mahjong, _, _, _ = step(state, Single(MAHJONG))
    final, _, _, _ = step(after_mahjong, MahjongWish(rank=9))
    assert final.public.mahjong_wish == 9
    assert final.public.pending_decision is None
    assert final.public.current_player == 1


def test_mahjong_wish_with_none_leaves_wish_unset():
    state = _state(
        {
            0: frozenset({MAHJONG, _c(Suit.JADE, 7)}),
            1: frozenset({_c(Suit.SWORD, 9)}),
            2: frozenset({_c(Suit.PAGODA, 10)}),
            3: frozenset({_c(Suit.STAR, 11)}),
        }
    )
    after_mahjong, _, _, _ = step(state, Single(MAHJONG))
    final, _, _, _ = step(after_mahjong, MahjongWish(rank=None))
    assert final.public.mahjong_wish is None
    assert final.public.pending_decision is None
    assert final.public.current_player == 1


# ---- Bomb interrupts (G5) ----

def test_bomb_interrupt_makes_bomb_the_new_top():
    # Player 0 led a 7. Player 3 has a four-7 bomb (wait, 7s already on table — use 5s).
    bomb_cards = (_c(Suit.JADE, 5), _c(Suit.SWORD, 5), _c(Suit.PAGODA, 5), _c(Suit.STAR, 5))
    state = _state(
        {
            0: frozenset({_c(Suit.JADE, 14)}),
            1: frozenset({_c(Suit.JADE, 11)}),
            2: frozenset({_c(Suit.JADE, 12)}),
            3: frozenset(bomb_cards),
        },
        top=Single(_c(Suit.JADE, 7)),
        current_player=1,
    )
    next_state, _, _, _ = step(state, BombInterrupt(player=3, bomb=FourOfAKindBomb(*bomb_cards)))
    assert next_state.public.trick.top_combination == FourOfAKindBomb(*bomb_cards)
    assert next_state.public.trick.leader == 3


def test_bomb_interrupt_removes_bomb_from_bomber_hand():
    bomb_cards = (_c(Suit.JADE, 5), _c(Suit.SWORD, 5), _c(Suit.PAGODA, 5), _c(Suit.STAR, 5))
    state = _state(
        {
            0: frozenset({_c(Suit.JADE, 14)}),
            1: frozenset({_c(Suit.JADE, 11)}),
            2: frozenset({_c(Suit.JADE, 12)}),
            3: frozenset(bomb_cards + (_c(Suit.JADE, 13),)),
        },
        top=Single(_c(Suit.JADE, 7)),
        current_player=1,
    )
    next_state, _, _, _ = step(state, BombInterrupt(player=3, bomb=FourOfAKindBomb(*bomb_cards)))
    assert next_state.hands[3] == frozenset({_c(Suit.JADE, 13)})
    assert next_state.public.hand_sizes[3] == 1


def test_bomb_interrupt_advances_turn_clockwise_from_bomber():
    bomb_cards = (_c(Suit.JADE, 5), _c(Suit.SWORD, 5), _c(Suit.PAGODA, 5), _c(Suit.STAR, 5))
    state = _state(
        {
            0: frozenset({_c(Suit.JADE, 14)}),
            1: frozenset({_c(Suit.JADE, 11)}),
            2: frozenset({_c(Suit.JADE, 12)}),
            3: frozenset(bomb_cards),
        },
        top=Single(_c(Suit.JADE, 7)),
        current_player=1,
    )
    next_state, _, _, _ = step(state, BombInterrupt(player=3, bomb=FourOfAKindBomb(*bomb_cards)))
    # Player 3 bombed; next to act is player 0.
    assert next_state.public.current_player == 0


def test_bomb_interrupt_clears_previous_passes():
    # Set up a state where player 1 has already passed. Then player 3 bombs.
    bomb_cards = (_c(Suit.JADE, 5), _c(Suit.SWORD, 5), _c(Suit.PAGODA, 5), _c(Suit.STAR, 5))
    state = _state(
        {
            0: frozenset({_c(Suit.JADE, 14)}),
            1: frozenset({_c(Suit.JADE, 11)}),
            2: frozenset({_c(Suit.JADE, 12)}),
            3: frozenset(bomb_cards),
        },
        top=Single(_c(Suit.JADE, 7)),
        current_player=2,
    )
    # Inject a prior pass from player 1 by editing the trick.
    state = GameState(
        hands=state.hands,
        public=PublicState(
            current_player=state.public.current_player,
            hand_sizes=state.public.hand_sizes,
            scores=state.public.scores,
            trick=state.public.trick.add_pass(1),
        ),
    )
    next_state, _, _, _ = step(state, BombInterrupt(player=3, bomb=FourOfAKindBomb(*bomb_cards)))
    assert next_state.public.trick.passes == frozenset()


def test_bomb_interrupt_illegal_if_bomb_does_not_beat_top():
    # Top is a king-bomb. A 5-bomb does not beat it.
    weak_bomb = FourOfAKindBomb(
        _c(Suit.JADE, 5), _c(Suit.SWORD, 5), _c(Suit.PAGODA, 5), _c(Suit.STAR, 5)
    )
    strong_top = FourOfAKindBomb(
        _c(Suit.JADE, 13), _c(Suit.SWORD, 13), _c(Suit.PAGODA, 13), _c(Suit.STAR, 13)
    )
    state = _state(
        {
            0: frozenset({_c(Suit.JADE, 14)}),
            1: frozenset({_c(Suit.JADE, 11)}),
            2: frozenset({_c(Suit.JADE, 12)}),
            3: frozenset(weak_bomb.a.__class__(suit=Suit.JADE, rank=r) for r in (2,)) | {
                _c(Suit.JADE, 5), _c(Suit.SWORD, 5), _c(Suit.PAGODA, 5), _c(Suit.STAR, 5)
            },
        },
        top=strong_top,
        current_player=1,
    )
    with pytest.raises(ValueError):
        step(state, BombInterrupt(player=3, bomb=weak_bomb))


# ---- Round-end detection (G6) ----

def test_round_not_done_when_no_player_is_out():
    state = _state({0: frozenset({_c(Suit.JADE, 7)}), 1: frozenset({_c(Suit.SWORD, 9)})})
    _, _, done, _ = step(state, Single(_c(Suit.JADE, 7)))
    # Player 0 went out, but only 3 players were in the deal — this state has
    # only 2 non-empty hands to start, so after the play, 3 are empty.
    # Use a fuller setup instead.
    state = _state(
        {p: frozenset({_c(Suit.JADE, 2 + p)}) for p in range(4)},
        current_player=0,
    )
    next_state, _, done, _ = step(state, Single(_c(Suit.JADE, 2)))
    # One player is now out; round not done yet.
    assert next_state.public.hand_sizes == (0, 1, 1, 1)
    assert done is False


def test_round_done_when_three_players_are_out():
    # Players 1, 2 already out; player 0 plays their last card. 3 out -> done.
    state = GameState(
        hands=(
            frozenset({_c(Suit.JADE, 7)}),
            frozenset(),
            frozenset(),
            frozenset({_c(Suit.STAR, 9)}),
        ),
        public=PublicState(
            current_player=0,
            hand_sizes=(1, 0, 0, 1),
            scores=(0, 0),
            trick=Trick.empty(),
        ),
    )
    _, _, done, _ = step(state, Single(_c(Suit.JADE, 7)))
    assert done is True


def test_round_end_credits_in_progress_trick_to_leader_team():
    # Players 1 (team 1) and 2 (team 0) already out; player 0 (team 0) leads the
    # final trick with a Single 10 as their last card and becomes 3rd-out. Player
    # 3 (team 1) is last-in. BSW credits the in-progress trick to the leader's
    # team (10 -> team 0); the engine must too.
    state = GameState(
        hands=(
            frozenset({_c(Suit.JADE, 10)}),
            frozenset(),
            frozenset(),
            frozenset({_c(Suit.SWORD, 4)}),
        ),
        public=PublicState(
            current_player=0,
            hand_sizes=(1, 0, 0, 1),
            scores=(0, 0),
            trick=Trick.empty(),
            out_order=(1, 2),
        ),
    )
    next_state, _, done, _ = step(state, Single(_c(Suit.JADE, 10)))
    assert done is True
    assert next_state.public.scores == (10, 0)


def test_round_end_credits_whole_in_progress_trick_not_just_last_play():
    # Players 2 (team 0) and 3 (team 1) already out. Trick history: player 0 led
    # with Single 5, player 1 beat with Single 7, player 0 now plays Single K
    # (their last card -> 3rd-out, makes player 0 the new trick leader). Trick
    # values: 5 + 7 + K = 5 + 0 + 10 = 15 points, credited to team 0.
    state = GameState(
        hands=(
            frozenset({_c(Suit.JADE, 13)}),  # King
            frozenset({_c(Suit.STAR, 9)}),
            frozenset(),
            frozenset(),
        ),
        public=PublicState(
            current_player=0,
            hand_sizes=(1, 1, 0, 0),
            scores=(0, 0),
            trick=Trick.empty()
                .add_play(player=0, combination=Single(_c(Suit.JADE, 5)))
                .add_play(player=1, combination=Single(_c(Suit.STAR, 7))),
            out_order=(2, 3),
        ),
    )
    next_state, _, done, _ = step(state, Single(_c(Suit.JADE, 13)))
    assert done is True
    assert next_state.public.scores == (15, 0)


def test_round_end_with_dragon_as_last_card_sets_dragon_give_pending():
    # Players 1 (team 1) and 2 (team 0) already out; player 0 plays Single Dragon
    # as their last card -> 3rd-out. Round must NOT finalise yet: we owe a
    # Dragon-give decision. Then DragonGive(target=3) routes 25 to team 1.
    state = GameState(
        hands=(
            frozenset({DRAGON}),
            frozenset(),
            frozenset(),
            frozenset({_c(Suit.SWORD, 4)}),
        ),
        public=PublicState(
            current_player=0,
            hand_sizes=(1, 0, 0, 1),
            scores=(0, 0),
            trick=Trick.empty(),
            out_order=(1, 2),
        ),
    )
    after_play, _, _, _ = step(state, Single(DRAGON))
    assert isinstance(after_play.public.pending_decision, DragonGivePending)
    assert after_play.public.pending_decision.winner == 0
    assert after_play.public.pending_decision.points == 25
    assert after_play.public.scores == (0, 0)
    final, _, done, _ = step(after_play, DragonGive(target=3))
    assert done is True
    assert final.public.scores == (0, 25)


def test_round_end_with_dragon_during_slam_nets_to_slam_bonus_only():
    # Player 0 (team 0) is 1st-out. Player 2 (team 0, partner) plays Single
    # Dragon as their last card -> 2nd-out -> SLAM. The Dragon's +25 must NOT
    # leak: slam pays exactly +200 to team 0.
    state = GameState(
        hands=(
            frozenset(),
            frozenset({_c(Suit.STAR, 9)}),
            frozenset({DRAGON}),
            frozenset({_c(Suit.SWORD, 4)}),
        ),
        public=PublicState(
            current_player=2,
            hand_sizes=(0, 1, 1, 1),
            scores=(0, 0),
            trick=Trick.empty(),
            out_order=(0,),
        ),
    )
    next_state, _, _, _ = step(state, Single(DRAGON))
    assert next_state.public.scores == (200, 0)
    assert next_state.public.pending_decision is None


def test_step_reports_done_on_slam():
    # Player 0 (team 0) is 1st-out; partner player 2 plays their last card ->
    # 2nd-out -> SLAM, round over. Regression: _finalise_round resets out_order,
    # so re-evaluating _round_done_state AFTER finalising wrongly returned
    # done=False for slams (fewer than 3 hands empty). step must report done=True.
    state = GameState(
        hands=(
            frozenset(),
            frozenset({_c(Suit.STAR, 9)}),
            frozenset({_c(Suit.STAR, 14)}),
            frozenset({_c(Suit.SWORD, 4)}),
        ),
        public=PublicState(
            current_player=2,
            hand_sizes=(0, 1, 1, 1),
            scores=(0, 0),
            trick=Trick.empty(),
            out_order=(0,),
        ),
    )
    next_state, _, done, _ = step(state, Single(_c(Suit.STAR, 14)))
    assert done is True
    assert next_state.public.scores == (200, 0)  # slam bonus; finalised exactly once


def test_slam_undoes_nonzero_midround_card_points_and_pays_only_200():
    # Doppelsieg must NOT double-count card points: any card points credited to
    # scores mid-round are undone, and the team gets exactly +200. Existing slam
    # tests only exercise the trivial zero-points case, so the undo arithmetic
    # (scores - round_team_points) was never actually tested against a non-zero
    # accumulation. Here team 0 collected 40 and team 1 collected 30 mid-round
    # (mirrored in round_points_by_player, as the engine always keeps them); the
    # partner pair (0, 2) then goes out 1-2. Net result: exactly (200, 0).
    state = GameState(
        hands=(
            frozenset(),
            frozenset({_c(Suit.STAR, 9)}),
            frozenset({_c(Suit.STAR, 9)}),  # seat 2's 0-point last card
            frozenset({_c(Suit.SWORD, 4)}),
        ),
        public=PublicState(
            current_player=2,
            hand_sizes=(0, 1, 1, 1),
            scores=(40, 30),                         # mid-round card points credited
            trick=Trick.empty(),
            out_order=(0,),
            round_points_by_player=(40, 0, 0, 30),   # team0=40, team1=30 (mirror)
        ),
    )
    next_state, _, done, _ = step(state, Single(_c(Suit.STAR, 9)))
    assert done is True
    assert next_state.public.scores == (200, 0)


def test_slam_preserves_prior_baseline_and_stacks_tichu_bust():
    # A slam on top of a prior-round baseline (300, 150), with a busted Tichu by
    # the losing team. Mid-round team 0 collected 40 (undone by the slam). The
    # +200 slam and the -100 Tichu bust (seat 1 is not first-out) must both land
    # on top of the preserved baseline: team0 = 300 + 200 = 500, team1 = 150 - 100 = 50.
    state = GameState(
        hands=(
            frozenset(),
            frozenset({_c(Suit.STAR, 9)}),
            frozenset({_c(Suit.STAR, 9)}),
            frozenset({_c(Suit.SWORD, 4)}),
        ),
        public=PublicState(
            current_player=2,
            hand_sizes=(0, 1, 1, 1),
            scores=(340, 150),                       # 300 baseline + 40 mid-round (team0)
            trick=Trick.empty(),
            out_order=(0,),
            round_points_by_player=(40, 0, 0, 0),
            tichu_callers=frozenset({1}),            # seat 1 (team 1) busts: not first out
        ),
    )
    next_state, _, done, _ = step(state, Single(_c(Suit.STAR, 9)))
    assert done is True
    assert next_state.public.scores == (500, 50)


def test_bomb_interrupt_legal_on_empty_trick():
    # When the previous trick has just resolved and the winner (current_player)
    # hasn't led yet, a non-current player holding a bomb can preempt-bomb to
    # seize the lead. BSW allows this; the engine must too.
    from tichu_engine.legality import legal_bomb_interrupts
    bomb_cards = (_c(Suit.JADE, 5), _c(Suit.SWORD, 5), _c(Suit.PAGODA, 5), _c(Suit.STAR, 5))
    state = _state(
        {
            0: frozenset({_c(Suit.JADE, 14)}),
            1: frozenset({_c(Suit.JADE, 11)}),
            2: frozenset({_c(Suit.JADE, 12)}),
            3: frozenset(bomb_cards),
        },
        current_player=0,  # player 0 has the lead, hasn't played yet
    )
    # Player 3 should be able to bomb-interrupt the empty trick.
    bombs = legal_bomb_interrupts(state, player=3)
    assert FourOfAKindBomb(*bomb_cards) in bombs
    # And the actual step should succeed and make the bomb the new top.
    next_state, _, _, _ = step(state, BombInterrupt(player=3, bomb=FourOfAKindBomb(*bomb_cards)))
    assert next_state.public.trick.top_combination == FourOfAKindBomb(*bomb_cards)
    assert next_state.public.trick.leader == 3


def test_round_end_via_bomb_interrupt_credits_bombed_trick_to_bomber_team():
    # Players 1 (team 1) and 2 (team 0) already out. Trick contains player 2's
    # final play (Single 9, leader=2). Current is player 3. Player 0 (team 0)
    # bomb-interrupts with FourOfAKindBomb(10s) as their last cards -> 3rd-out.
    # Without the fix the bombed trick (40 points) is dropped; with the fix it
    # credits to team 0.
    bomb_cards = [_c(s, 10) for s in (Suit.JADE, Suit.STAR, Suit.PAGODA, Suit.SWORD)]
    bomb = FourOfAKindBomb(*bomb_cards)
    state = GameState(
        hands=(
            frozenset(bomb_cards),
            frozenset(),
            frozenset(),
            frozenset({_c(Suit.SWORD, 4)}),
        ),
        public=PublicState(
            current_player=3,
            hand_sizes=(4, 0, 0, 1),
            scores=(0, 0),
            trick=Trick.empty().add_play(player=2, combination=Single(_c(Suit.JADE, 9))),
            out_order=(1, 2),
        ),
    )
    next_state, _, _, _ = step(state, BombInterrupt(player=0, bomb=bomb))
    assert next_state.public.scores == (40, 0)


def test_round_done_persists_after_dragon_give():
    # Configure a state where a Dragon-give resolves and the giver was the third
    # to go out (round done before the give; done remains true after).
    state = GameState(
        hands=(
            frozenset(),  # already out
            frozenset(),
            frozenset(),
            frozenset({_c(Suit.JADE, 3)}),  # last player with a card
        ),
        public=PublicState(
            current_player=0,
            hand_sizes=(0, 0, 0, 1),
            scores=(0, 0),
            trick=Trick.empty(),
            pending_decision=DragonGivePending(winner=0, points=25),
        ),
    )
    _, _, done, _ = step(state, DragonGive(target=1))
    assert done is True


# ---- Return contract ----

def test_step_returns_four_tuple():
    state = _state({0: frozenset({_c(Suit.JADE, 7)})})
    result = step(state, Single(_c(Suit.JADE, 7)))
    assert len(result) == 4
    _, reward, done, info = result
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
