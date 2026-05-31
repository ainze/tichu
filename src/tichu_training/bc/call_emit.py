"""Per-round Call-example emit — the single source of truth shared by the
standalone `ParquetCallDataset` and the consolidated `parse_bsw` pass.

[ADR-0020](../../../docs/adr/0020-one-parse-pass-many-task-bundles.md) chose
*share-not-mirror* for the call tasks: rather than copy the emit loop into the
consolidated pass (as ADR-0016 did for BC), both paths import these functions.
Living in their own module keeps the conflict surface in `call_training.py`
small for the in-flight Tichu train/val-split work.

* `grand_tichu_examples_for_round` — parse-bound: a synthetic deal-time
  `GameState` from `pre_deal_hands` (8 cards), one example per seat. No replay.
* `tichu_examples_for_round` — replay-bound: takes a **passed-in**
  `ReplayResult` and featurises each seat at its first non-Pass Play state
  ([ADR-0018](../../../docs/adr/0018-tichu-call-featurises-at-first-non-pass-play.md)).
  The consolidated pass hands it the round's *full* replay (already computed
  for Parquet validation); the standalone dataset hands it an early-stopped
  replay. Both yield the same examples — the first-non-Pass-Play states are
  identical whether or not the replay continued past them (pinned by
  `test_call_emit.py`).
"""

from __future__ import annotations

from tichu_training.bc.call_training import CallExample


def grand_tichu_examples_for_round(
    parsed_round,
    *,
    skill_lookup: dict[str, int],
    neutral_decile: int,
    sample_weight: float,
    game_id: str = "",
    round_id: int = 0,
) -> list[CallExample]:
    """One CallExample per seat at the synthetic deal-time (8-card) state.

    `current_player=0` is arbitrary — no engine step runs against this state;
    the featurizer reads PrivateState fields directly. `game_id`/`round_id`
    flow onto each example (game-level split key + bundle provenance)."""
    from tichu_engine.state import GameState, PublicState, Trick
    from tichu_training.featurizer import featurize

    public = PublicState(
        current_player=0,
        hand_sizes=(8, 8, 8, 8),
        scores=(0, 0),
        trick=Trick.empty(),
        pending_decision=None,
    )
    state = GameState(hands=parsed_round.pre_deal_hands, public=public)
    callers = parsed_round.grand_tichu_callers
    out: list[CallExample] = []
    for seat in range(4):
        private = state.private_view(seat)
        features = featurize(private)
        handle = parsed_round.handles[seat]
        skill = skill_lookup.get(handle, neutral_decile)
        if skill is None:  # nullable lookup (consolidated pass); default neutral
            skill = neutral_decile
        out.append(CallExample(
            features=features,
            target=int(seat in callers),
            skill_decile=skill,
            sample_weight=sample_weight,
            game_id=game_id,
            round_id=round_id,
        ))
    return out


def tichu_examples_for_round(
    parsed_round,
    replay,
    *,
    skill_lookup: dict[str, int],
    neutral_decile: int,
    sample_weight: float,
    game_id: str = "",
    round_id: int = 0,
) -> list[CallExample]:
    """One CallExample per seat, featurised at the seat's first non-Pass Play
    state (ADR-0018). Consumes the `replay` it is given — the caller decides
    whether that is a full or early-stopped `ReplayResult`; the first
    non-Pass-Play states are identical either way."""
    from tichu_training.featurizer import featurize, mask_self_tichu_call

    callers = parsed_round.tichu_callers
    first_play_state_for_seat: dict[int, object] = {}
    for (parsed_action, _concrete), pre_state in zip(
        replay.decisions, replay.pre_decision_states,
    ):
        if pre_state is None:
            continue  # phantom pass / call-passthrough
        if parsed_action.kind != "play":
            continue  # excludes pass (the C-wide rule), schupfen, wish, …
        seat = parsed_action.player
        if seat in first_play_state_for_seat:
            continue
        first_play_state_for_seat[seat] = pre_state
    out: list[CallExample] = []
    for seat in range(4):
        pre_state = first_play_state_for_seat.get(seat)
        if pre_state is None:
            continue  # seat never made a non-Pass Play this round (rare)
        private = pre_state.private_view(seat)  # type: ignore[attr-defined]
        features = featurize(private)
        # `pub.tichu_callers` at the first-play snapshot already contains this
        # seat's own call (the regular Tichu window closes when the seat plays
        # its first card). Slot `tichu_callers[seat]` IS the label — mask it
        # out or the classifier trivially solves the task. (Ported from main's
        # label-leak fix; PR #27.)
        mask_self_tichu_call(features, seat)
        handle = parsed_round.handles[seat]
        skill = skill_lookup.get(handle, neutral_decile)
        if skill is None:  # nullable lookup (consolidated pass); default neutral
            skill = neutral_decile
        out.append(CallExample(
            features=features,
            target=int(seat in callers),
            skill_decile=skill,
            sample_weight=sample_weight,
            game_id=game_id,
            round_id=round_id,
        ))
    return out
