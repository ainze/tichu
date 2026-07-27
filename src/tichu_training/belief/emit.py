"""Per-round Belief-example emit — replay-derived (ADR-0021).

For each Play Decision in a validated round, build one `BeliefExample` from
the acting player's view: features = `featurize(private_view(seat))` (the same
vector the BC `play` example uses), labels = the three opponents' actual Hands
at that engine state in relative-seat order (next / partner / previous),
mask = the card-level "unknown" set (cards held by some opponent).

The features are the policy Feature Vector *unextended*. ADR-0028's History
block used to be appended here; v6 folded its surviving channels (B-core:
declined_top / lead_summary / pass_pressure) into the featurizer itself, so
appending them again was pure duplication — see `input_spec.py`. Reading them
off the live engine accumulators is also strictly more faithful than the old
replay-side accumulation, which missed the synthetic PASSes `replay._sync_for`
steps through the engine without recording in `replay.decisions`.

The labels are *free here*: the replay already holds every seat's Hand at each
`pre_decision_state`. This is the share-not-mirror emit the consolidated parse
pass calls; there is no standalone replay dataset for belief (the shipped
`SyntheticBeliefDataset` is a smoke placeholder).
"""

from __future__ import annotations

import numpy as np

from tichu_training.belief.dataset import BeliefExample


_NUM_OPPONENTS = 3
_NUM_CARDS = 56
# A Play Decision is the in-trick choice over the Play action space, which
# includes Pass — so both `play` and `pass` parsed kinds are sampled.
_PLAY_KINDS = frozenset({"play", "pass"})


def belief_examples_for_round(
    parsed_round,
    replay,
    *,
    game_id: int = 0,
    round_id: int = 0,
) -> list[BeliefExample]:
    """One BeliefExample per Play Decision (acting seat's view). Labels read
    from the same `pre_decision_state` that produces the features, so they are
    always fresh to the featurise moment."""
    from tichu_training.card_slots import card_slot
    from tichu_training.featurizer import featurize

    out: list[BeliefExample] = []
    for (parsed_action, _concrete), pre_state in zip(
        replay.decisions, replay.pre_decision_states,
    ):
        if pre_state is None:
            continue
        if parsed_action.kind not in _PLAY_KINDS:
            continue
        seat = parsed_action.player
        if 0 <= seat < 4:
            # `pre_state` is the state BEFORE this action, so its cross-Trick
            # decline / lead / pressure accumulators carry every earlier
            # decision but not this one — the ADR-0028 read-out convention.
            features = featurize(pre_state.private_view(seat))
            hands = pre_state.hands
            cards_played = _NUM_CARDS - sum(len(h) for h in hands)
            # Opponents in relative-seat order: next / partner / previous.
            rel_seats = ((seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4)
            labels = np.zeros((_NUM_OPPONENTS, _NUM_CARDS), dtype=np.float32)
            for opp_idx, opp_seat in enumerate(rel_seats):
                for card in hands[opp_seat]:
                    labels[opp_idx, card_slot(card)] = 1.0
            # A card is "unknown" iff some opponent holds it (≡ not own, not played).
            card_mask = labels.any(axis=0)
            mask = np.broadcast_to(card_mask, (_NUM_OPPONENTS, _NUM_CARDS)).copy()
            out.append(BeliefExample(
                features=features,
                labels=labels,
                mask=mask,
                cards_played=int(cards_played),
                game_id=game_id,
                round_id=round_id,
            ))
    return out
