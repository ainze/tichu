"""Regression: a mid-round Mahjong-wish *decline* must be recorded as a real
`wish` Decision so BC can train the no-wish target (wish-head index 0).

BSW logs a decline as the *absence* of a "Wunsch:" line after a Mahjong play
(~20% of Mahjong plays). The replay layer auto-declines such a pending wish,
but used to step it through the engine WITHOUT appending it to
`ReplayResult.decisions` — so the no-wish target never reached any BC emit
path and the served wish head only ever saw rank wishes. See the wish-decline
diagnosis.

`sample/2417500.tch` rounds 4 and 6 are real mid-round declines (seat 1 plays
the Mahjong and no "Wunsch:" line follows, but the round continues).
"""

from pathlib import Path

from tichu_engine.legality import MahjongWish
from tichu_engine.state import MahjongWishPending
from tichu_training.action_space import (
    bc_target_for_concrete,
    legal_mask,
    wish_intent_index,
)
from tichu_training.bsw.parser import parse_tch
from tichu_training.bsw.replay import replay_round

_SAMPLES = Path(__file__).resolve().parents[3] / "sample"


def _round(filename: str, index: int):
    game = parse_tch((_SAMPLES / filename).read_text(encoding="utf-8"), game_id=filename[:-4])
    return game.rounds[index]


def _wish_declines(replay):
    """The (parsed, concrete, pre_state, legal) tuples for recorded declines."""
    return [
        (parsed, concrete, pre_state, legal)
        for (parsed, concrete), pre_state, legal in zip(
            replay.decisions, replay.pre_decision_states, replay.legal_actions_at,
        )
        if parsed.kind == "wish" and parsed.wish_rank is None
    ]


def test_mid_round_decline_is_recorded_as_a_wish_decision():
    replay = replay_round(_round("2417500.tch", 4))
    assert replay.final_state is not None, "round must replay legally after the fix"

    declines = _wish_declines(replay)
    assert len(declines) == 1, "round 4 has exactly one mid-round decline (seat 1)"

    parsed, concrete, pre_state, legal = declines[0]
    # The decline carries enough to become a BC example.
    assert pre_state is not None
    assert isinstance(pre_state.public.pending_decision, MahjongWishPending)
    wisher = pre_state.public.pending_decision.player
    assert parsed.player == wisher == 1
    assert concrete == MahjongWish(rank=None)
    # The recorded legal set is the full wish universe (every slot legal).
    assert MahjongWish(rank=None) in legal


def test_recorded_decline_maps_to_the_no_wish_bc_target():
    """The decline must produce the no-wish label (wish-head index 0) and that
    slot must be legal — i.e. it survives `_emit_bc_examples_for_round`'s
    `mask[target]` filter rather than being silently dropped."""
    replay = replay_round(_round("2417500.tch", 4))
    parsed, concrete, pre_state, _ = _wish_declines(replay)[0]

    target = bc_target_for_concrete("wish", concrete)
    assert target == 0 == wish_intent_index(MahjongWish(rank=None))

    mask = legal_mask("wish", pre_state, parsed.player)
    assert mask[target], "no-wish target must be legal or BC would drop it"


def test_every_recorded_decline_has_a_pending_wish_pre_state():
    """Guard against recording moot/end-of-round declines (where the round is
    ending and the wish is meaningless). Every recorded decline across the
    sample games must sit on a genuine MahjongWishPending pre-state."""
    for filename in ("2417500.tch", "2417501.tch"):
        game = parse_tch(
            (_SAMPLES / filename).read_text(encoding="utf-8"), game_id=filename[:-4],
        )
        for rnd in game.rounds:
            replay = replay_round(rnd)
            for _parsed, _concrete, pre_state, _legal in _wish_declines(replay):
                assert pre_state is not None
                assert isinstance(pre_state.public.pending_decision, MahjongWishPending)
