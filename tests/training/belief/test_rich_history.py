"""Rich play-history projections for the Belief Model (ADR-0041 follow-up).

v6's B-core keeps 27 numbers of negative information for a whole Round —
`declined_top` is `max(rank)` per opponent per combo type, with no context, no
length, no bombs, and no record of a wish-forced void. This module tests the
richer projections that upper-bound how much a better representation could buy.
"""

import numpy as np

from tichu_engine.combinations import Pair, Single
from tichu_engine.deck import Card, Suit
from tichu_engine.legality import Pass
from tichu_training.belief.rich_history import wish_void_deduction


def _single(rank):
    return Single(card=Card(suit=Suit.JADE, rank=rank))


def _pair(rank):
    return Pair(a=Card(suit=Suit.JADE, rank=rank), b=Card(suit=Suit.SWORD, rank=rank))


def test_a_lead_that_ducks_an_active_wish_proves_the_rank_is_void():
    """When LEADING, every combination in hand is a candidate, so holding the
    wished rank would force it (`_apply_wish`). Leading something else is a
    rules-derived certainty — the same kind of fact as Unseen Cards, and the one
    piece of card information v6 discards entirely."""
    assert wish_void_deduction(_single(9), wish=7, is_lead=True) == 7
    assert wish_void_deduction(_pair(4), wish=7, is_lead=True) == 7


def test_fulfilling_the_wish_proves_nothing():
    assert wish_void_deduction(_single(7), wish=7, is_lead=True) is None
    assert wish_void_deduction(_pair(7), wish=7, is_lead=True) is None


def test_following_and_passing_are_evidence_but_never_proof():
    """Following restricts candidates to combinations that BEAT the top, so a
    player may hold the wished rank and be unable to play it; a Pass means no
    legal action existed at all. Encoding either as a void would inject a false
    certainty into the features."""
    assert wish_void_deduction(_single(9), wish=7, is_lead=False) is None
    assert wish_void_deduction(Pass(), wish=7, is_lead=True) is None
    assert wish_void_deduction(Pass(), wish=7, is_lead=False) is None


def test_no_active_wish_yields_no_deduction():
    assert wish_void_deduction(_single(9), wish=None, is_lead=True) is None


def test_rich_emit_appends_history_and_stays_causal():
    """Rich examples are the v6 Feature Vector plus the recovered channels. The
    block must reflect only Decisions made BEFORE the one being featurised —
    folding in the actor's own current action would leak the answer."""
    from tichu_eval.full_position_pool import generate_full_position_pool
    from tichu_ml.rule_agent import RuleAgent
    from tichu_training.belief.rich_history import RICH_HISTORY_DIM
    from tichu_training.belief.selfplay_emit import belief_examples_for_selfplay_round
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM

    position = generate_full_position_pool(seed=11, n=1)[0]
    agents = [RuleAgent() for _ in range(4)]
    plain = belief_examples_for_selfplay_round(agents, position)
    rich = belief_examples_for_selfplay_round(
        [RuleAgent() for _ in range(4)], position, rich_history=True,
    )

    assert len(rich) == len(plain)
    assert rich[0].features.shape == (FEATURIZER_OUTPUT_DIM + RICH_HISTORY_DIM,)
    # The v6 prefix is untouched — rich history is strictly additive.
    for got, want in zip(rich, plain):
        np.testing.assert_array_equal(
            got.features[:FEATURIZER_OUTPUT_DIM], want.features
        )

    # Nothing has happened before the first Decision; something has by the end.
    assert not rich[0].features[FEATURIZER_OUTPUT_DIM:].any()
    assert rich[-1].features[FEATURIZER_OUTPUT_DIM:].any()


def test_a_bomb_interrupt_is_folded_as_the_bomb_it_plays():
    """A non-current player can seize a Trick with a Bomb. `BombInterrupt` wraps
    the bomb rather than being a Combination itself, so the accumulator must
    unwrap it — self-play never produces one, only the BSW replay does."""
    from tichu_engine.combinations import FourOfAKindBomb
    from tichu_engine.legality import BombInterrupt
    from tichu_training.belief.rich_history import action_cards

    bomb = FourOfAKindBomb(
        a=Card(suit=Suit.JADE, rank=9), b=Card(suit=Suit.SWORD, rank=9),
        c=Card(suit=Suit.PAGODA, rank=9), d=Card(suit=Suit.STAR, rank=9),
    )
    interrupt = BombInterrupt(player=2, bomb=bomb)

    assert {c.rank for c in action_cards(interrupt)} == {9}
    assert len(action_cards(interrupt)) == 4
    # A Pass plays nothing, and an unrecognised action must degrade to empty
    # rather than raise — the accumulator is a measurement, not a validator.
    assert action_cards(Pass()) == []
    assert action_cards(object()) == []


def test_wish_checks_never_raise_on_a_bomb_interrupt():
    """`_fulfills_wish` calls `_cards_in`, which raises on `BombInterrupt` — so
    every wish check must go through the same unwrap as the card extraction. The
    soft (following) branch bypassed it once and aborted a corpus scan."""
    from tichu_engine.combinations import FourOfAKindBomb
    from tichu_engine.legality import BombInterrupt
    from tichu_training.belief.rich_history import fulfills_wish

    bomb = FourOfAKindBomb(
        a=Card(suit=Suit.JADE, rank=9), b=Card(suit=Suit.SWORD, rank=9),
        c=Card(suit=Suit.PAGODA, rank=9), d=Card(suit=Suit.STAR, rank=9),
    )
    assert fulfills_wish(BombInterrupt(player=2, bomb=bomb), 9) is True
    assert fulfills_wish(BombInterrupt(player=2, bomb=bomb), 7) is False
    assert fulfills_wish(_single(7), 7) is True
    assert fulfills_wish(Pass(), 7) is False
