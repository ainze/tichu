"""PrivateState JSON codec — round-trip."""

import pytest

from tichu_engine.cards import DRAGON, MAHJONG, PHOENIX, Card, Suit
from tichu_engine.combinations import Pair, Single
from tichu_engine.legality import DragonGive, MahjongWish, PASS, SchupfenPass
from tichu_engine.rich_history import (
    CTX_OPPONENT_WINNING, INTENT_SINGLE, RichHistory,
)
from tichu_engine.state import (
    DragonGivePending,
    GameState,
    MahjongWishPending,
    Play,
    PrivateState,
    PublicState,
    SchupfenPending,
    Trick,
    deal_initial_state,
    deal_for_schupfen,
)

from tichu_inference.codec import (
    action_to_json,
    private_state_from_json,
    private_state_to_json,
    public_state_from_json,
    public_state_to_json,
)


def test_deal_initial_round_trip():
    state = deal_initial_state(seed=0)
    ps = state.private_view(0)
    blob = private_state_to_json(ps)
    restored = private_state_from_json(blob)
    assert restored == ps


def test_schupfen_pending_round_trip():
    state = deal_for_schupfen(seed=0)
    ps = state.private_view(0)
    blob = private_state_to_json(ps)
    restored = private_state_from_json(blob)
    assert restored == ps


def test_mid_trick_round_trip():
    state = deal_initial_state(seed=0)
    public = state.public
    # Synthesise a mid-trick state with a top Pair using whichever rank
    # appears at least twice in seat 0's hand.
    by_rank: dict[int, list[Card]] = {}
    for c in state.hands[0]:
        if isinstance(c, Card):
            by_rank.setdefault(c.rank, []).append(c)
    a, b = next(cards for cards in by_rank.values() if len(cards) >= 2)[:2]
    pair = Pair(a, b)
    trick = Trick(plays=(Play(player=0, combination=pair),), leader=0)
    public = PublicState(
        current_player=1,
        hand_sizes=public.hand_sizes,
        scores=public.scores,
        trick=trick,
        round_points_by_player=(1, 2, 3, 4),
        out_order=(),
        tichu_callers=frozenset({2}),
        grand_tichu_callers=frozenset(),
        mahjong_wish=7,
    )
    ps = state.private_view(1).__class__(player=1, hand=state.hands[1], public=public)
    blob = private_state_to_json(ps)
    restored = private_state_from_json(blob)
    assert restored == ps


def test_v6_provenance_and_decline_fields_round_trip():
    """v6 (ADR-0038): the new per-player provenance / decline accumulators
    (PublicState) and self-only schupfen_received (PrivateState) survive the
    wire round-trip — else the served featurizer sees them as empty."""
    hand = frozenset({Card(Suit.SWORD, r) for r in range(2, 15)})  # 13 cards
    public = PublicState(
        current_player=1, hand_sizes=(13, 14, 14, 14), scores=(0, 0),
        trick=Trick.empty(),
        played_cards_by_player=(
            frozenset({Card(Suit.JADE, 7)}), frozenset({DRAGON}),
            frozenset(), frozenset({PHOENIX}),
        ),
        declined_top_by_player=((0,) * 6, (13, 7, 0, 0, 0, 0), (0,) * 6, (0,) * 6),
        lead_summary_by_player=((1, 4), (0, 0), (2, 9), (0, 0)),
        pass_stats_by_player=((0, 1), (3, 6), (0, 0), (2, 2)),
    )
    ps = PrivateState(
        player=0, hand=hand, public=public,
        schupfen_received=(Card(Suit.STAR, 3), MAHJONG, Card(Suit.STAR, 5)),
    )
    restored = private_state_from_json(private_state_to_json(ps))
    assert restored == ps


def test_pre_v6_blob_without_new_keys_deserializes_to_defaults():
    """Backward-compat: a client that omits the v6 keys yields a valid state with
    the zeroed/empty accumulator defaults (no decode error)."""
    ps = deal_initial_state(seed=0).private_view(0)
    blob = private_state_to_json(ps)
    del blob["schupfen_received"]
    for k in ("played_cards_by_player", "declined_top_by_player",
              "lead_summary_by_player", "pass_stats_by_player"):
        del blob["public"][k]
    restored = private_state_from_json(blob)
    assert restored == ps  # all new fields fall back to their defaults


def test_dragon_give_pending_round_trip():
    state = deal_initial_state(seed=0)
    public = state.public
    public = PublicState(
        current_player=public.current_player,
        hand_sizes=public.hand_sizes,
        scores=public.scores,
        trick=public.trick,
        pending_decision=DragonGivePending(winner=2, points=25),
    )
    ps = state.private_view(0).__class__(player=0, hand=state.hands[0], public=public)
    blob = private_state_to_json(ps)
    restored = private_state_from_json(blob)
    assert restored == ps


def test_mahjong_wish_pending_round_trip():
    state = deal_initial_state(seed=0)
    public = state.public
    public = PublicState(
        current_player=public.current_player,
        hand_sizes=public.hand_sizes,
        scores=public.scores,
        trick=public.trick,
        pending_decision=MahjongWishPending(player=public.current_player),
    )
    ps = state.private_view(0).__class__(player=0, hand=state.hands[0], public=public)
    blob = private_state_to_json(ps)
    restored = private_state_from_json(blob)
    assert restored == ps


def test_action_to_json_single():
    blob = action_to_json(Single(Card(suit=Suit.JADE, rank=5)))
    assert blob["kind"] == "Single"


def test_action_to_json_pass():
    assert action_to_json(PASS) == {"kind": "Pass"}


def test_action_to_json_dragon_give():
    assert action_to_json(DragonGive(target=2)) == {"kind": "DragonGive", "target": 2}


def test_action_to_json_mahjong_wish():
    assert action_to_json(MahjongWish(rank=7)) == {"kind": "MahjongWish", "rank": 7}
    assert action_to_json(MahjongWish(rank=None)) == {"kind": "MahjongWish", "rank": None}


def test_action_to_json_schupfen_pass():
    a = Card(Suit.JADE, 4)
    b = Card(Suit.SWORD, 5)
    c = Card(Suit.PAGODA, 6)
    blob = action_to_json(SchupfenPass(to_next=a, to_partner=b, to_previous=c))
    assert blob["kind"] == "SchupfenPass"
    assert "to_next" in blob and "to_partner" in blob and "to_previous" in blob


def test_malformed_payload_raises_helpful_error():
    with pytest.raises(ValueError):
        private_state_from_json({"hand": [0, 1, 2]})  # missing player + public.


def test_special_cards_roundtrip():
    state = deal_initial_state(seed=0)
    # Find which player holds Dragon, Phoenix, Mahjong — ensure they survive round-trip.
    for ps_player in range(4):
        ps = state.private_view(ps_player)
        if not (DRAGON in ps.hand or PHOENIX in ps.hand or MAHJONG in ps.hand):
            continue
        blob = private_state_to_json(ps)
        restored = private_state_from_json(blob)
        assert restored.hand == ps.hand


def test_v7_rich_history_round_trips():
    """v7 (ADR-0044): the Rich History Block accumulator must survive the wire,
    or the served featurizer computes 233 dims of zeros while training saw real
    values — the `team_scores` bug class, at 28% of the vector."""
    rich = (
        RichHistory()
        .with_decline(2, INTENT_SINGLE, CTX_OPPONENT_WINNING, 13,
                      length=0, stakes=20)
        .with_proven_void(1, 9)
        .with_wish_evidence(3, 11)
        .with_declined_bomb(2)
        .with_lead_single(3, 14)
        .with_play([0, 5])
    )
    public = PublicState(
        current_player=1, hand_sizes=(13, 14, 14, 14), scores=(0, 0),
        trick=Trick.empty(), rich_history=rich,
    )
    ps = PrivateState(
        player=0,
        hand=frozenset({Card(Suit.SWORD, r) for r in range(2, 15)}),
        public=public,
    )
    restored = private_state_from_json(private_state_to_json(ps))
    assert restored.public.rich_history == rich
    assert restored == ps


def test_strict_mode_rejects_a_payload_without_the_v7_block():
    """The v6 accumulators default to zeros on a missing key, which is why the
    served agent can silently run on 42% zeroed inputs today. v7 keeps the
    lenient default so the CURRENT client keeps working, but adds an explicit
    strict switch: once the client is updated, flipping this turns a silent
    degradation into a loud failure. That switch is the whole point — a
    half-updated client must not be able to corrupt inputs undetectably.
    """
    ps = deal_initial_state(seed=0).private_view(0)
    blob = public_state_to_json(ps.public)
    del blob["rich_history"]

    lenient = public_state_from_json(blob)
    assert lenient.rich_history == RichHistory()  # today's behaviour, documented

    with pytest.raises(ValueError, match="rich_history"):
        public_state_from_json(blob, require_v7=True)
