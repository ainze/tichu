"""Faithfulness of the FROZEN v5 featurizer running on a v6 engine state.

The cross-version tournament (cotrain_v6 vs cotrain_wish_v5) seats a v5-trained
net at the same table as a v6 net. The v5 net must be fed v5 (224-dim) features —
computed by `featurizer_v5_frozen` — but those features are derived from a v6
`PrivateState`. This is faithful ONLY if the v5 featurizer's output is *invariant*
to the fields v6 ADDED (ADR-0038): `played_cards_by_player`, `declined_top_by_player`,
`lead_summary_by_player`, `pass_stats_by_player` on PublicState, and
`schupfen_received` on PrivateState. v6 is a strict superset (existing fields
unchanged), so a faithful frozen v5 reads only pre-v6 fields and ignores the rest.

These tests pin exactly that: dim/version are the v5 contract, and populating the
v6-only fields does NOT move the v5 vector by a single bit.
"""

import numpy as np

from tichu_engine.cards import Card, Suit
from tichu_engine.state import GameState, PublicState, Trick
from tichu_training import featurizer_v5_frozen as fz5


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


def _base_state(*, played: frozenset = frozenset()) -> GameState:
    """A plain mid-round playing-phase v6 state with all v6-added fields at their
    defaults (empty)."""
    hands = (
        frozenset({_c(Suit.JADE, 7), _c(Suit.SWORD, 3), _c(Suit.STAR, 14)}),
        frozenset({_c(Suit.STAR, 10), _c(Suit.PAGODA, 5)}),
        frozenset({_c(Suit.SWORD, 9)}),
        frozenset({_c(Suit.PAGODA, 11)}),
    )
    public = PublicState(
        current_player=0,
        hand_sizes=tuple(len(h) for h in hands),  # type: ignore[arg-type]
        scores=(0, 0),
        trick=Trick.empty(),
        played_cards_this_round=played,
    )
    return GameState(hands=hands, public=public)


def test_version_and_dim_are_the_v5_contract():
    assert fz5.FEATURIZER_VERSION == "v5"
    assert fz5.FEATURIZER_OUTPUT_DIM == 224


def test_featurize_v6_state_yields_v5_shape():
    vec = fz5.featurize(_base_state().private_view(0))
    assert vec.shape == (224,)
    assert vec.dtype == np.float32


def test_invariant_to_v6_only_public_fields():
    """Populating the v6-added PublicState fields must not change the v5 vector —
    proof the frozen featurizer reads none of them."""
    base = _base_state(played=frozenset({_c(Suit.JADE, 7)}))
    base_vec = fz5.featurize(base.private_view(0))

    # Same pre-v6 state, but every v6-added PublicState field populated to a
    # non-default value (these are exactly the ADR-0038 additions).
    enriched_public = base.public.__class__(
        current_player=base.public.current_player,
        hand_sizes=base.public.hand_sizes,
        scores=base.public.scores,
        trick=base.public.trick,
        played_cards_this_round=base.public.played_cards_this_round,
        played_cards_by_player=(
            frozenset({_c(Suit.JADE, 7)}), frozenset(), frozenset(), frozenset()),
        declined_top_by_player=((9, 0, 0, 0, 0, 0),) * 4,
        lead_summary_by_player=((3, 5),) * 4,
        pass_stats_by_player=((2, 4),) * 4,
    )
    enriched = GameState(hands=base.hands, public=enriched_public)
    enriched_vec = fz5.featurize(enriched.private_view(0))

    assert np.array_equal(base_vec, enriched_vec)


def test_invariant_to_v6_schupfen_received():
    """The re-added self-only `schupfen_received` (v6, ADR-0038) was a documented
    rules-violation removal in v3 — the v5 featurizer must not read it."""
    base = _base_state()
    base_vec = fz5.featurize(base.private_view(0))

    received = (
        _c(Suit.SWORD, 2), _c(Suit.STAR, 13), _c(Suit.PAGODA, 8),  # seat 0 got 3 cards
        None, None, None, None, None, None, None, None, None,
    )[:4]
    enriched = GameState(
        hands=base.hands, public=base.public,
        schupfen_received=(received[:3], None, None, None),
    )
    enriched_vec = fz5.featurize(enriched.private_view(0))

    assert np.array_equal(base_vec, enriched_vec)
