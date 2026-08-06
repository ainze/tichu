"""v7 (ADR-0044): the engine maintains the Rich History Block accumulator that
the featurizer reads — the canonical, single definition shared by training and
inference.

This is the ADR-0038 pattern. The alternative — each caller accumulating its own
history featurizer-side — is exactly the shape that produced the shipped
`team_scores` and schupfen `current_player` train/inference skews, so the state
lives on `PublicState` and nowhere else.
"""

from dataclasses import replace

from tichu_engine.cards import Card, Suit
from tichu_engine.combinations import FourOfAKindBomb, Pair, PairStep, Single
from tichu_engine.engine import _finalise_round, step
from tichu_engine.legality import PASS
from tichu_engine.rich_history import (
    CTX_OPPONENT_WINNING,
    RichHistory,
    CTX_PARTNER_WINNING,
    INTENT_PAIR_STEP,
    INTENT_SINGLE,
)
from tichu_engine.state import GameState, PublicState, Trick


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


def _state(hands_dict: dict[int, frozenset], *, top=None, leader: int = 0,
           current_player: int = 1) -> GameState:
    hands: list[frozenset] = [frozenset()] * 4
    for p, h in hands_dict.items():
        hands[p] = h
    trick = Trick.empty()
    if top is not None:
        trick = trick.add_play(player=leader, combination=top)
    public = PublicState(
        current_player=current_player,
        hand_sizes=tuple(len(h) for h in hands),  # type: ignore[arg-type]
        scores=(0, 0),
        trick=trick,
    )
    return GameState(hands=tuple(hands), public=public)


def test_pass_records_declined_rank_under_opponent_winning():
    """Seat 1 declines a King led by seat 0 — an OPPONENT (teams are seat % 2).

    Declining while an opponent is winning is real evidence about the hand: the
    seat had a reason to take the trick and could not. v6's `declined_top` records
    the rank but not this context, which is the whole point of the channel.
    """
    king = _c(Suit.SWORD, 13)
    state = _state(
        {0: frozenset({_c(Suit.JADE, 2)}), 1: frozenset({_c(Suit.STAR, 10)})},
        top=Single(king), leader=0, current_player=1,
    )
    ns, _, _, _ = step(state, PASS)
    rh = ns.public.rich_history
    assert rh.declined_rank(1, INTENT_SINGLE, CTX_OPPONENT_WINNING) == 13
    assert rh.declined_rank(1, INTENT_SINGLE, CTX_PARTNER_WINNING) == 0


def test_leading_under_an_active_wish_without_the_rank_proves_a_void():
    """The one CERTAINTY in the block.

    A seat leading a fresh Trick may play any Combination in hand, so if it held
    the wished rank at all, legality would have FORCED a wish-fulfilling lead.
    Leading something else therefore proves the rank is absent — not "suggests",
    proves. Seat 1 leads a King under an active wish for 9.
    """
    state = _state(
        {
            0: frozenset({_c(Suit.JADE, 2)}),
            1: frozenset({_c(Suit.SWORD, 13), _c(Suit.STAR, 4)}),
            2: frozenset({_c(Suit.PAGODA, 3)}),
        },
        top=None, current_player=1,
    )
    state = GameState(
        hands=state.hands,
        public=replace(state.public, mahjong_wish=9),
    )
    ns, _, _, _ = step(state, Single(_c(Suit.SWORD, 13)))
    assert ns.public.rich_history.is_wish_void(1, 9) is True


def test_passing_under_an_active_wish_does_not_prove_a_void():
    """Seat 1 HOLDS the wished 9 and still passes — legally, because a 9 cannot
    beat a King. Marking this a void would be flatly false.

    Soundness is the channel's entire value: a false void silently corrupts every
    inference downstream of it. This pins the negative case with a hand that makes
    the falseness demonstrable rather than hypothetical.
    """
    state = _state(
        {
            0: frozenset({_c(Suit.JADE, 2)}),
            1: frozenset({_c(Suit.JADE, 9), _c(Suit.STAR, 4)}),
            2: frozenset({_c(Suit.PAGODA, 3)}),
        },
        top=Single(_c(Suit.SWORD, 13)), leader=0, current_player=1,
    )
    state = GameState(hands=state.hands,
                      public=replace(state.public, mahjong_wish=9))
    ns, _, _, _ = step(state, PASS)
    assert ns.public.rich_history.is_wish_void(1, 9) is False


def test_following_without_the_wished_rank_does_not_prove_a_void():
    """Seat 1 holds the wished 9 but only its King beats the Queen on top, so the
    wish restriction — which applies only AMONG already-legal actions — does not
    bite. Playing the King is honest and proves nothing about the 9."""
    state = _state(
        {
            0: frozenset({_c(Suit.JADE, 2)}),
            1: frozenset({_c(Suit.JADE, 9), _c(Suit.SWORD, 13)}),
            2: frozenset({_c(Suit.PAGODA, 3)}),
        },
        top=Single(_c(Suit.STAR, 12)), leader=0, current_player=1,
    )
    state = GameState(hands=state.hands,
                      public=replace(state.public, mahjong_wish=9))
    ns, _, _, _ = step(state, Single(_c(Suit.SWORD, 13)))
    assert ns.public.rich_history.is_wish_void(1, 9) is False


def test_passing_on_a_bomb_counts_as_a_declined_bomb():
    """v6 drops this entirely: `combo_type_and_rank` returns None for a Bomb, so
    `declined_top` records nothing and the fact that a seat could not (or would
    not) over-bomb is lost. It is strong evidence about the hand — bombs are rare
    and a seat holding one almost always uses it here.
    """
    sevens = [_c(Suit.JADE, 7), _c(Suit.SWORD, 7),
              _c(Suit.PAGODA, 7), _c(Suit.STAR, 7)]
    state = _state(
        {
            0: frozenset({_c(Suit.STAR, 14)}),
            1: frozenset({_c(Suit.STAR, 9)}),
            2: frozenset({_c(Suit.STAR, 11)}),
        },
        top=FourOfAKindBomb(*sevens), leader=3, current_player=1,
    )
    ns, _, _, _ = step(state, PASS)
    rh = ns.public.rich_history
    assert rh.declined_bomb_count(1) == 1
    assert rh.declined_bomb_count(0) == 0
    # And it must NOT leak into the ranked decline channel — a Bomb has no
    # (type, rank) in the six non-bomb Intent types.
    assert rh.declined_rank(1, INTENT_SINGLE, CTX_OPPONENT_WINNING) == 0


def test_finalise_resets_the_rich_history_accumulator():
    """Round-only state. A void proven in Round N says nothing in Round N+1 —
    the hands are new. Mirrors `test_finalise_resets_decline_accumulators`.
    """
    rh = (
        RichHistory()
        .with_decline(1, INTENT_SINGLE, CTX_OPPONENT_WINNING, 13)
        .with_proven_void(2, 9)
        .with_declined_bomb(3)
    )
    public = PublicState(
        current_player=0, hand_sizes=(0, 0, 0, 2), scores=(0, 0),
        trick=Trick.empty(), out_order=(0, 1, 2), rich_history=rh,
    )
    hands = (frozenset(), frozenset(), frozenset(),
             frozenset({_c(Suit.STAR, 4), _c(Suit.STAR, 5)}))
    fin = _finalise_round(GameState(hands=hands, public=public))
    assert fin.public.rich_history == RichHistory()


def test_declining_records_frequency_length_and_stakes():
    """v6 keeps only the max rank declined. The same Pass also says HOW OFTEN the
    seat has declined this type, how LONG a combination it could not beat, and
    how many card points it let go — the last being the strongest of the three,
    since declining a fat Trick is far more informative than declining an empty
    one.

    Seat 1 declines a 3-long PairStep (10-J-Q) worth 20 points — the two Tens.
    """
    pair_step = PairStep((
        Pair(_c(Suit.JADE, 10), _c(Suit.SWORD, 10)),
        Pair(_c(Suit.JADE, 11), _c(Suit.SWORD, 11)),
        Pair(_c(Suit.JADE, 12), _c(Suit.SWORD, 12)),
    ))
    state = _state(
        {
            0: frozenset({_c(Suit.STAR, 2)}),
            1: frozenset({_c(Suit.STAR, 9)}),
            2: frozenset({_c(Suit.PAGODA, 3)}),
        },
        top=pair_step, leader=0, current_player=1,
    )
    ns, _, _, _ = step(state, PASS)
    rh = ns.public.rich_history
    assert rh.declined_count(1, INTENT_PAIR_STEP) == 1
    assert rh.declined_length(1, INTENT_PAIR_STEP) == 3
    assert rh.declined_stakes(1) == 20  # two Tens
    assert rh.declined_count(0, INTENT_PAIR_STEP) == 0


def test_declining_under_a_wish_is_recorded_as_evidence_but_not_proof():
    """The same Pass feeds two channels with different epistemic status.

    `wish_soft` says "this seat had a chance to fulfil the wish and did not" —
    real evidence, and usually right. `wish_void` says "this seat cannot hold the
    rank" — a certainty. Keeping them apart is what lets a consumer weigh the
    guess without ever being lied to by the proof.

    Seat 1 holds the wished 9 but cannot beat a King, so it passes: evidence yes,
    proof no.
    """
    state = _state(
        {
            0: frozenset({_c(Suit.JADE, 2)}),
            1: frozenset({_c(Suit.JADE, 9), _c(Suit.STAR, 4)}),
            2: frozenset({_c(Suit.PAGODA, 3)}),
        },
        top=Single(_c(Suit.SWORD, 13)), leader=0, current_player=1,
    )
    state = GameState(hands=state.hands,
                      public=replace(state.public, mahjong_wish=9))
    ns, _, _, _ = step(state, PASS)
    rh = ns.public.rich_history
    assert rh.wish_evidence(1, 9) is True
    assert rh.is_wish_void(1, 9) is False


def test_lead_records_the_highest_single_as_well_as_the_lowest():
    """v6's `lead_summary` keeps only the LOWEST single lead. The highest is the
    other half of a seat's lead profile: leading Aces reads very differently from
    leading Threes, and a seat that has led both is describing its shape."""
    state = _state(
        {
            0: frozenset({_c(Suit.JADE, 2)}),
            1: frozenset({_c(Suit.SWORD, 14), _c(Suit.STAR, 4)}),
            2: frozenset({_c(Suit.PAGODA, 3)}),
        },
        top=None, current_player=1,
    )
    ns, _, _, _ = step(state, Single(_c(Suit.SWORD, 14)))
    assert ns.public.rich_history.lead_max_single(1) == 14
    assert ns.public.lead_summary_by_player[1] == (1, 14)  # v6 low-lead unchanged
    assert ns.public.rich_history.lead_max_single(0) == 0


def test_play_order_is_recorded_per_card():
    """WHEN a card appeared, not just THAT it appeared. v6's `played_by` is a
    position-less set per seat, so an Ace dumped on trick one and an Ace held to
    the endgame are identical to it."""
    state = _state(
        {
            0: frozenset({_c(Suit.JADE, 2)}),
            1: frozenset({_c(Suit.SWORD, 14), _c(Suit.STAR, 4)}),
            2: frozenset({_c(Suit.PAGODA, 3)}),
        },
        top=None, current_player=1,
    )
    ace = _c(Suit.SWORD, 14)
    ns, _, _, _ = step(state, Single(ace))
    rh = ns.public.rich_history
    assert rh.play_order(ace) == 1                      # the first play
    assert rh.play_order(_c(Suit.STAR, 4)) == 0         # never played
    assert rh.plays_so_far == 1
