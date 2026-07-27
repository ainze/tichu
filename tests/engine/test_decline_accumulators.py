"""v6 (ADR-0038 / ADR-0028 B-core): the engine maintains per-seat cross-Trick
decline accumulators that the featurizer reads — declined_top_by_player,
lead_summary_by_player, pass_stats_by_player. These are now the CANONICAL
definition of the ADR-0028 B-core channels: the belief-side replay accumulator
that once mirrored them is gone (it duplicated the v6 featurizer and, unlike
these, missed the synthetic PASSes the BSW replay steps through the engine)."""

from tichu_engine.cards import Card, DOG, MAHJONG, Suit
from tichu_engine.combinations import FourOfAKindBomb, Single, Pair
from tichu_engine.engine import step, _finalise_round
from tichu_engine.legality import BombInterrupt, PASS
from tichu_engine.state import GameState, PublicState, Trick


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


def test_pass_increments_pass_stats_for_passer():
    eight = _c(Suit.SWORD, 8)
    state = _state(
        {0: frozenset({_c(Suit.JADE, 7)}), 1: frozenset({_c(Suit.STAR, 10)})},
        top=Single(eight), current_player=1,
    )
    ns, _, _, _ = step(state, PASS)
    # (n_passes, n_play_decisions)
    assert ns.public.pass_stats_by_player[1] == (1, 1)
    assert ns.public.pass_stats_by_player[0] == (0, 0)


def test_play_counts_as_decision_only():
    seven = _c(Suit.JADE, 7)
    state = _state({0: frozenset({seven, _c(Suit.SWORD, 3)}), 1: frozenset({_c(Suit.STAR, 10)})})
    ns, _, _, _ = step(state, Single(seven))
    assert ns.public.pass_stats_by_player[0] == (0, 1)


def test_pass_records_declined_top_rank_for_its_type():
    king = _c(Suit.SWORD, 13)
    state = _state(
        {0: frozenset({_c(Suit.JADE, 7)}), 1: frozenset({_c(Suit.STAR, 10)})},
        top=Single(king), current_player=1,
    )
    ns, _, _, _ = step(state, PASS)
    # Single = type index 0; King = rank 13 (raw; featurizer normalises /14).
    assert ns.public.declined_top_by_player[1][0] == 13
    assert ns.public.declined_top_by_player[1][1] == 0  # Pair type untouched


def test_lead_play_increments_lead_count_and_lowest_single():
    four = _c(Suit.JADE, 4)
    state = _state({0: frozenset({four, _c(Suit.SWORD, 9)}), 1: frozenset({_c(Suit.STAR, 10)})})
    ns, _, _, _ = step(state, Single(four))
    assert ns.public.lead_summary_by_player[0] == (1, 4)  # led 1 trick, lowest single 4


def test_dog_lead_counts_as_decision_and_lead_without_lowest_single():
    state = _state({
        0: frozenset({DOG}), 1: frozenset({_c(Suit.STAR, 10)}),
        2: frozenset({_c(Suit.PAGODA, 11)}), 3: frozenset({_c(Suit.SWORD, 12)}),
    })
    ns, _, _, _ = step(state, Single(DOG))
    assert ns.public.pass_stats_by_player[0] == (0, 1)
    assert ns.public.lead_summary_by_player[0] == (1, 0)  # Dog is not a natural single


def test_mahjong_play_counts_as_decision_and_lead():
    state = _state({
        0: frozenset({MAHJONG, _c(Suit.JADE, 5)}), 1: frozenset({_c(Suit.STAR, 10)}),
        2: frozenset({_c(Suit.PAGODA, 11)}), 3: frozenset({_c(Suit.SWORD, 12)}),
    })
    ns, _, _, _ = step(state, Single(MAHJONG))
    assert ns.public.pass_stats_by_player[0] == (0, 1)
    assert ns.public.lead_summary_by_player[0] == (1, 0)  # rank 1, not a 2..14 natural


def test_bomb_interrupt_counts_as_decision_for_bomber():
    sevens = [_c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7), _c(Suit.STAR, 7)]
    state = _state(
        {
            0: frozenset({_c(Suit.STAR, 14)}),
            2: frozenset(sevens + [_c(Suit.STAR, 8)]),
            1: frozenset({_c(Suit.STAR, 9)}),
            3: frozenset({_c(Suit.STAR, 11)}),
        },
        top=Single(_c(Suit.JADE, 10)), current_player=1,
    )
    ns, _, _, _ = step(state, BombInterrupt(player=2, bomb=FourOfAKindBomb(*sevens)))
    # The bomber made a play decision; not a fresh lead (interrupting an open trick).
    assert ns.public.pass_stats_by_player[2] == (0, 1)
    assert ns.public.lead_summary_by_player[2] == (0, 0)


def test_finalise_resets_decline_accumulators():
    public = PublicState(
        current_player=0, hand_sizes=(0, 0, 0, 2), scores=(0, 0),
        trick=Trick.empty(), out_order=(0, 1, 2),
        declined_top_by_player=((0,) * 6, (13, 0, 0, 0, 0, 0), (0,) * 6, (0,) * 6),
        lead_summary_by_player=((1, 4), (0, 0), (0, 0), (0, 0)),
        pass_stats_by_player=((0, 1), (1, 1), (0, 0), (0, 0)),
    )
    hands = (frozenset(), frozenset(), frozenset(),
             frozenset({_c(Suit.STAR, 4), _c(Suit.STAR, 5)}))
    fin = _finalise_round(GameState(hands=hands, public=public))
    assert fin.public.declined_top_by_player == ((0,) * 6,) * 4
    assert fin.public.lead_summary_by_player == ((0, 0),) * 4
    assert fin.public.pass_stats_by_player == ((0, 0),) * 4
