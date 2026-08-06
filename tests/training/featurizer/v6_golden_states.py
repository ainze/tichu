"""Representative PrivateStates for the v7 additive-prefix golden test.

ADR-0044 makes `featurize(...)[:591]` byte-identical to the v6 Feature Vector a
load-bearing invariant: it is what lets the served `cpfix3328` Checkpoint stay in
the co-train rollout and the promotion gate behind a **v6-View Prefix** slice,
instead of a frozen second featurizer plus a second `featurize()` call in the
rollout's hottest loop. It is also what makes the v6-equivalent control arm
provable rather than asserted.

A prefix invariant is only worth what its inputs exercise, so these states are
built to populate EVERY v6 section — including the ones a fresh state leaves at
zero (`played_by`, `schupfen_received`, the ADR-0028 B-core accumulators,
`trick_leader`, `out_order`, `round_points`, `mahjong_wish`, both caller sets).
A golden captured on empty states would pass while the prefix silently rotted.

The builders use only pre-v7 constructor fields, so they keep working after v7
adds its own (defaulted) `PublicState` field — the same property that let
`featurizer_v5_frozen`'s tests survive the v6 bump.
"""

from tichu_engine.cards import Card, DOG, DRAGON, MAHJONG, PHOENIX, Suit
from tichu_engine.combinations import Pair
from tichu_engine.state import PrivateState, PublicState, Trick


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


def _fresh_deal() -> PrivateState:
    """Round start: every v6 accumulator at its default. The floor case."""
    hand = frozenset({
        MAHJONG, DOG,
        _c(Suit.JADE, 3), _c(Suit.JADE, 5), _c(Suit.JADE, 7), _c(Suit.JADE, 9),
        _c(Suit.SWORD, 4), _c(Suit.SWORD, 6), _c(Suit.SWORD, 8),
        _c(Suit.PAGODA, 10), _c(Suit.PAGODA, 12), _c(Suit.PAGODA, 14),
        _c(Suit.STAR, 11), _c(Suit.STAR, 13),
    })
    public = PublicState(
        current_player=0,
        hand_sizes=(14, 14, 14, 14),
        scores=(0, 0),
        trick=Trick.empty(),
    )
    return PrivateState(player=0, hand=hand, public=public)


def _mid_round_contested() -> PrivateState:
    """Mid-round, seat 2 to act into a contested Trick led by seat 1.

    Populates every v6 channel at once: per-player provenance, both caller sets,
    an active Mahjong Wish, a non-empty Trick with passes and a leader, all three
    B-core accumulators, and self-only schupfen provenance.
    """
    hand = frozenset({
        _c(Suit.JADE, 8), _c(Suit.SWORD, 8), _c(Suit.PAGODA, 11),
        _c(Suit.STAR, 12), PHOENIX, _c(Suit.JADE, 2),
    })
    trick = (
        Trick.empty()
        .add_play(player=1, combination=Pair(_c(Suit.JADE, 10), _c(Suit.SWORD, 10)))
        .add_pass(player=3)
    )
    public = PublicState(
        current_player=2,
        hand_sizes=(4, 5, 6, 7),
        scores=(35, 60),
        trick=trick,
        mahjong_wish=9,
        round_points_by_player=(15, 20, 5, 40),
        out_order=(3,),
        tichu_callers=frozenset({1}),
        grand_tichu_callers=frozenset({3}),
        played_cards_by_player=(
            frozenset({_c(Suit.JADE, 4), DOG}),
            frozenset({_c(Suit.JADE, 10), _c(Suit.SWORD, 10)}),
            frozenset({_c(Suit.PAGODA, 3)}),
            frozenset({DRAGON, _c(Suit.STAR, 5), MAHJONG}),
        ),
        declined_top_by_player=(
            (13, 0, 0, 0, 0, 0),
            (0, 10, 0, 0, 0, 0),
            (7, 0, 0, 0, 0, 9),
            (0, 0, 12, 0, 0, 0),
        ),
        lead_summary_by_player=((3, 4), (2, 0), (1, 11), (5, 2)),
        pass_stats_by_player=((2, 9), (0, 6), (4, 7), (1, 8)),
    )
    return PrivateState(
        player=2,
        hand=hand,
        public=public,
        schupfen_received=(_c(Suit.STAR, 12), PHOENIX, _c(Suit.JADE, 2)),
    )


def _endgame_leading() -> PrivateState:
    """Late Round, seat 3 leading a fresh Trick with two seats already out."""
    hand = frozenset({_c(Suit.SWORD, 14), DRAGON})
    public = PublicState(
        current_player=3,
        hand_sizes=(0, 3, 0, 2),
        scores=(80, 20),
        trick=Trick.empty(),
        mahjong_wish=None,
        round_points_by_player=(50, 10, 30, 10),
        out_order=(0, 2),
        tichu_callers=frozenset({0, 3}),
        grand_tichu_callers=frozenset(),
        played_cards_by_player=(
            frozenset({_c(Suit.JADE, 6), _c(Suit.SWORD, 6), MAHJONG}),
            frozenset({_c(Suit.PAGODA, 7)}),
            frozenset({DOG, _c(Suit.STAR, 9)}),
            frozenset({PHOENIX}),
        ),
        declined_top_by_player=(
            (0, 0, 0, 0, 0, 0),
            (14, 13, 0, 0, 0, 0),
            (0, 0, 0, 11, 0, 0),
            (5, 0, 0, 0, 8, 0),
        ),
        lead_summary_by_player=((6, 2), (0, 0), (4, 14), (7, 3)),
        pass_stats_by_player=((0, 12), (7, 11), (3, 10), (2, 13)),
    )
    return PrivateState(
        player=3,
        hand=hand,
        public=public,
        schupfen_received=(DRAGON, None, _c(Suit.SWORD, 14)),
    )


def golden_states() -> list[PrivateState]:
    """Deterministic, order-stable. The golden fixture is a row per state, so
    appending here invalidates the fixture — regenerate it deliberately."""
    return [_fresh_deal(), _mid_round_contested(), _endgame_leading()]
