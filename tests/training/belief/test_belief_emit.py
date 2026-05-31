"""Correctness of the replay-derived belief emit (ADR-0021): one example per
Play Decision, labels = the three opponents' actual Hands in relative-seat
order, mask = cards held by some opponent."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tichu_training.belief.emit import belief_examples_for_round
from tichu_training.bsw.parser import parse_tch
from tichu_training.bsw.replay import replay_round
from tichu_training.card_slots import card_slot


_SAMPLES = Path(__file__).resolve().parents[3] / "sample"
_PLAY_KINDS = {"play", "pass"}


def test_belief_labels_match_opponent_hands_in_relative_order():
    seen = 0
    for tch in sorted(_SAMPLES.glob("*.tch")):
        gid = int(tch.stem)
        game = parse_tch(tch.read_bytes().decode("utf-8"), game_id=tch.stem)
        for rnd in game.rounds:
            replay = replay_round(rnd)
            if replay.final_state is None:
                continue
            examples = belief_examples_for_round(
                rnd, replay, game_id=gid, round_id=rnd.round_index,
            )
            expected = [
                (pa.player, ps)
                for (pa, _c), ps in zip(replay.decisions, replay.pre_decision_states)
                if ps is not None and pa.kind in _PLAY_KINDS and 0 <= pa.player < 4
            ]
            assert len(examples) == len(expected)
            for ex, (seat, ps) in zip(examples, expected):
                assert ex.game_id == gid and ex.round_id == rnd.round_index
                rel_seats = ((seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4)
                for opp_idx, opp_seat in enumerate(rel_seats):
                    got = {s for s in range(56) if ex.labels[opp_idx, s]}
                    want = {card_slot(c) for c in ps.hands[opp_seat]}
                    assert got == want
                # mask = cards held by some opponent, broadcast over the 3 rows.
                expect_mask = np.broadcast_to(ex.labels.any(axis=0), (3, 56))
                np.testing.assert_array_equal(ex.mask, expect_mask)
                # The acting seat's own cards are never labelled.
                own = {card_slot(c) for c in ps.hands[seat]}
                assert not (own & {s for s in range(56) if ex.labels.any(axis=0)[s]})
                assert ex.cards_played == 56 - sum(len(h) for h in ps.hands)
                seen += 1
    assert seen > 0, "no belief examples produced from sample data — fixture broken"
