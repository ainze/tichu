"""Behavioral Drift Benchmark — the per-Decision recorder.

The recorder is a pure side channel over `play_full_round`: it sees every one of
the 6 Decisions (Grand-Tichu Call, Schupfen, Tichu Call, Play, Wish, Dragon
Assignment) and never changes how the Round plays out. See CONTEXT.md
§"Behavioral Drift Benchmark".
"""

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_eval.play_full import play_full_round
from tichu_ml.rule_agent import RuleAgent

from tests.eval.test_play_full import ScriptedCaller


def _positions(n=6):
    return generate_full_position_pool(seed=0, n=n)


def test_decision_observer_is_a_pure_side_channel():
    for pos in _positions():
        mk = lambda: (ScriptedCaller(tichu=True), RuleAgent(), RuleAgent(), RuleAgent())
        seen = []
        observed = play_full_round(
            mk(), pos.state, pos.grand_prefixes,
            decision_observer=lambda *a: seen.append(a),
        )
        plain = play_full_round(mk(), pos.state, pos.grand_prefixes)
        assert observed == plain
        assert seen  # it did fire


def test_record_round_logs_every_decision_of_every_seat():
    from collections import Counter

    from tichu_eval.drift_recorder import record_round

    for i, pos in enumerate(_positions()):
        agents = (ScriptedCaller(grand=True), RuleAgent(), ScriptedCaller(tichu=True), RuleAgent())
        log = record_round(agents, pos, deal=i, half=0, subject_seats=(0, 2))
        kinds = Counter((r["seat"], r["kind"]) for r in log.decisions)
        for seat in range(4):
            assert kinds[(seat, "grand")] == 1       # every seat is asked, once
            assert kinds[(seat, "schupfen")] == 1
            assert kinds[(seat, "tichu")] <= 1       # asked at its first non-Pass Play
            assert kinds[(seat, "play")] >= 1
        assert kinds[(0, "tichu")] == 0              # a Grand caller is never asked
        # One Wish iff the Mahjong was played; at most one Dragon Assignment, and
        # only if the Dragon was played (a Bomb can still take the Trick).
        n_wish = sum(1 for r in log.decisions if r["kind"] == "wish")
        n_dragon = sum(1 for r in log.decisions if r["kind"] == "dragon")
        assert n_wish == int(log.round["mahjong_played"])
        assert n_dragon <= int(log.round["dragon_played"])
        assert {r["deal"] for r in log.decisions} == {i}
        assert all(r["is_subject"] == (r["seat"] in (0, 2)) for r in log.decisions)


# --- Play-row situation facts, on hand-built mid-Trick states -----------------

from types import SimpleNamespace

import pytest

from tichu_engine.cards import Card, Suit
from tichu_engine.combinations import Single
from tichu_engine.state import GameState, Play, PublicState, Trick


def _c(rank, suit=Suit.JADE):
    return Card(suit, rank)


def _mid_trick(*, holder, current, current_hand, tichu_callers=frozenset()):
    """A 4-seat state mid-Trick: `holder` has just played a lone King (10 points)
    and holds the Trick; `current` is to act holding `current_hand`."""
    filler = iter([_c(r, s) for s in (Suit.SWORD, Suit.PAGODA) for r in (4, 5, 6, 7, 8, 9)])
    hands = []
    for seat in range(4):
        hands.append(frozenset(current_hand) if seat == current
                     else frozenset({next(filler), next(filler)}))
    trick = Trick(plays=(Play(player=holder, combination=Single(_c(13, Suit.STAR))),),
                  leader=holder)
    public = PublicState(
        current_player=current, hand_sizes=tuple(len(h) for h in hands),
        scores=(0, 0), trick=trick, tichu_callers=frozenset(tichu_callers),
    )
    state = GameState(hands=tuple(hands), public=public)
    return SimpleNamespace(state=state, grand_prefixes=(frozenset(),) * 4)


def _first_play_row(pos):
    agents = tuple(RuleAgent() for _ in range(4))
    from tichu_eval.drift_recorder import record_round

    log = record_round(agents, pos, deal=0, half=0, subject_seats=(0, 2))
    return next(r for r in log.decisions if r["kind"] == "play")


def test_play_row_partner_holds_trick_with_a_legal_beat():
    # Seat 2 follows its partner (seat 0, a Tichu caller) holding an Ace.
    row = _first_play_row(_mid_trick(holder=0, current=2, current_hand={_c(14), _c(3)},
                                     tichu_callers={0}))
    assert row["seat"] == 2
    assert row["holder"] == "partner"
    assert row["has_beat"] and not row["forced"] and not row["bomb_legal"]
    assert row["partner_called"] and not row["opponent_called"]
    assert row["trick_points"] == 10


def test_play_row_opponent_holds_trick_and_only_pass_is_legal():
    # Seat 1 follows an opponent (seat 0) holding nothing above a King: forced Pass.
    row = _first_play_row(_mid_trick(holder=0, current=1, current_hand={_c(2), _c(3)},
                                     tichu_callers={0}))
    assert row["holder"] == "opponent"
    assert not row["has_beat"] and row["forced"]
    assert row["opponent_called"] and not row["partner_called"]
    assert row["action"] == "pass"


def test_play_row_leading_an_empty_trick():
    pos = _mid_trick(holder=0, current=2, current_hand={_c(14), _c(3)})
    empty = pos.state.public.__class__(**{**pos.state.public.__dict__, "trick": Trick.empty()})
    pos.state = GameState(hands=pos.state.hands, public=empty)
    row = _first_play_row(pos)
    assert row["holder"] == "none"
    assert row["action"] == "single"   # the lead's Combination type


def test_round_row_records_the_outcome():
    from tichu_eval.drift_recorder import record_round

    slams = 0
    for i, pos in enumerate(generate_full_position_pool(seed=0, n=40)):
        mk = lambda: (ScriptedCaller(grand=True), RuleAgent(), RuleAgent(), ScriptedCaller(tichu=True))
        log = record_round(mk(), pos, deal=i, half=1, subject_seats=(1, 3))
        ref = play_full_round(mk(), pos.state, pos.grand_prefixes)
        r = log.round
        assert (r["total_0"], r["total_1"]) == ref.total
        assert (r["call_bonus_0"], r["call_bonus_1"]) == ref.call_bonus
        assert r["subject_team"] == 1
        assert r["grand_callers"] == (0,)
        assert r["tichu_callers"] in ((), (3,))   # seat 3 is asked iff it ever plays
        out = r["out_order"]
        assert len(set(out)) == len(out) and set(out) <= {0, 1, 2, 3}
        slam_team = out[0] % 2 if len(out) >= 2 and out[0] % 2 == out[1] % 2 else None
        assert r["slam_team"] == slam_team
        slams += slam_team is not None
    assert slams > 0   # the sample actually exercises a Slam


# --- Call / Schupfen / Wish / Dragon rows ---------------------------------------

from tichu_engine.cards import DOG, DRAGON, MAHJONG, PHOENIX


def _record(agents, pos, subject=(0, 2)):
    from tichu_eval.drift_recorder import record_round

    return record_round(agents, pos, deal=0, half=0, subject_seats=subject)


def test_call_rows_carry_the_answer_and_the_hand_power():
    for pos in _positions(10):
        agents = (ScriptedCaller(grand=True), RuleAgent(), ScriptedCaller(tichu=True), RuleAgent())
        log = _record(agents, pos)
        for r in log.decisions:
            if r["kind"] == "grand":
                hand = pos.grand_prefixes[r["seat"]]
                assert r["called"] == (r["seat"] == 0)
                assert r["partner_called"] is False and r["opponent_called"] is False
            elif r["kind"] == "tichu":
                assert r["called"] == (r["seat"] == 2)
                # Only seat 2's partner (seat 0) called — a Grand caller, visible.
                assert r["partner_called"] == (r["seat"] == 2)
                assert r["opponent_called"] == (r["seat"] in (1, 3))
                hand = None
            else:
                continue
            if hand is not None:
                power = sum(1 for c in hand if c in (DRAGON, PHOENIX)
                            or getattr(c, "rank", 0) == 14)
                assert r["power"] == power


def test_schupfen_row_names_the_card_given_in_each_direction():
    for pos in _positions(10):
        log = _record(tuple(RuleAgent() for _ in range(4)), pos)
        for r in (r for r in log.decisions if r["kind"] == "schupfen"):
            hand = pos.state.hands[r["seat"]]
            labels = {_label(c) for c in hand}
            gifts = (r["give_next"], r["give_partner"], r["give_previous"])
            assert set(gifts) <= labels
            assert r["holds_dog"] == (DOG in hand)
            assert r["holds_dragon"] == (DRAGON in hand)


def _label(card):
    from tichu_eval.drift_recorder import card_label

    return card_label(card)


def test_card_labels():
    from tichu_eval.drift_recorder import card_label

    assert [card_label(_c(r)) for r in (2, 10, 11, 12, 13, 14)] == ["2", "10", "J", "Q", "K", "A"]
    assert [card_label(c) for c in (DOG, MAHJONG, PHOENIX, DRAGON)] == [
        "dog", "mahjong", "phoenix", "dragon"]


def _pending_state(pending, *, current, trick, hand_sizes, tichu_callers=frozenset()):
    filler = iter([_c(r, s) for s in (Suit.SWORD, Suit.PAGODA, Suit.STAR) for r in range(2, 14)])
    hands = tuple(frozenset(next(filler) for _ in range(n)) for n in hand_sizes)
    public = PublicState(current_player=current, hand_sizes=tuple(hand_sizes), scores=(0, 0),
                         trick=trick, pending_decision=pending,
                         tichu_callers=frozenset(tichu_callers))
    return SimpleNamespace(state=GameState(hands=hands, public=public),
                           grand_prefixes=(frozenset(),) * 4)


def test_wish_row_names_the_wished_rank():
    from tichu_engine.state import MahjongWishPending

    trick = Trick(plays=(Play(player=2, combination=Single(MAHJONG)),), leader=2)
    pos = _pending_state(MahjongWishPending(player=2), current=2, trick=trick,
                         hand_sizes=(3, 3, 3, 3))
    log = _record(tuple(RuleAgent() for _ in range(4)), pos)
    (row,) = [r for r in log.decisions if r["kind"] == "wish"]
    assert row["seat"] == 2
    ranks = {"none", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"}
    assert row["wish"] in ranks


def test_dragon_row_relates_the_receiver_to_the_other_opponent():
    # Reference: the Dragon-give GameStates seen directly through the hook, on the
    # same deterministic games the recorder plays.
    seen = 0
    for pos in _positions(80):
        mk = lambda: (ScriptedCaller(tichu=True), RuleAgent(), RuleAgent(), RuleAgent())
        states = []
        play_full_round(mk(), pos.state, pos.grand_prefixes,
                        decision_observer=lambda s_, k, gs, a: k == "dragon" and states.append(gs))
        rows = [r for r in _record(mk(), pos).decisions if r["kind"] == "dragon"]
        assert len(rows) == len(states)
        for row, gs in zip(rows, states):
            seat = row["seat"]
            target = (seat + 1) % 4 if row["target"] == "next" else (seat + 3) % 4
            other = (seat + 3) % 4 if row["target"] == "next" else (seat + 1) % 4
            sizes = gs.public.hand_sizes
            expect = ("more" if sizes[target] > sizes[other]
                      else "fewer" if sizes[target] < sizes[other] else "equal")
            assert row["target_cards"] == expect
            callers = gs.public.tichu_callers | gs.public.grand_tichu_callers
            assert row["target_called"] == (target in callers)
            seen += 1
    assert seen >= 3


def test_play_row_hand_and_caller_facts():
    bomb = {_c(9, s) for s in Suit}
    row = _first_play_row(_mid_trick(holder=0, current=2, current_hand={DOG, _c(14), *bomb},
                                     tichu_callers={2}))
    assert row["holds_dog"] and row["holds_bomb"] and row["self_called"]
    assert row["top_rank"] == 13
    row = _first_play_row(_mid_trick(holder=0, current=2, current_hand={_c(14), _c(3)}))
    assert not row["holds_dog"] and not row["holds_bomb"] and not row["self_called"]


def test_play_row_flags_the_specials_it_plays_and_round_row_counts_tricks():
    dragons = 0
    for i, pos in enumerate(_positions(30)):
        log = _record(tuple(RuleAgent() for _ in range(4)), pos)
        plays = [r for r in log.decisions if r["kind"] == "play"]
        n_dragon = sum(r["plays_dragon"] for r in plays)
        assert n_dragon == int(log.round["dragon_played"])
        assert sum(r["plays_phoenix"] for r in plays) <= 1
        dragons += n_dragon
        r = log.round
        assert sum(r["tricks_won"]) == r["tricks_total"] > 0
        assert r["wish_fulfilled"] is None or log.round["mahjong_played"]
    assert dragons > 0


def test_recorder_agrees_with_behavioral_telemetry_on_the_same_rounds():
    # Two independent implementations of the same facts must agree: the in-loop
    # BehavioralProfile counters vs the recorder's raw rows (all seats, no
    # forced filter — BehavioralProfile's own definitions).
    from tichu_eval.behavioral import profile_agent_self_play

    builder = lambda: ScriptedCaller(tichu=True)
    positions = _positions(40)
    profile = profile_agent_self_play(builder, positions)

    plays, rounds = [], []
    for i, pos in enumerate(positions):
        log = _record(tuple(builder() for _ in range(4)), pos)
        plays += [r for r in log.decisions if r["kind"] == "play"]
        rounds.append(log.round)

    follow_partner = [r for r in plays if r["holder"] == "partner" and r["has_beat"]]
    caller_vs_opp = [r for r in plays
                     if r["holder"] == "opponent" and r["has_beat"] and r["self_called"]]
    assert profile.partner_steal_opportunities == len(follow_partner)
    assert profile.partner_steal_events == sum(r["action"] != "pass" for r in follow_partner)
    assert profile.partner_steal_caller_opportunities == sum(r["partner_called"] for r in follow_partner)
    assert profile.caller_pass_opportunities == len(caller_vs_opp)
    assert profile.caller_pass_events == sum(r["action"] == "pass" for r in caller_vs_opp)
    assert profile.bomb_legal_decisions == sum(r["bomb_legal"] for r in plays)
    assert profile.bombs_played == sum(r["action"] == "bomb" for r in plays)
    assert profile.slam_for == 2 * sum(r["slam_team"] is not None for r in rounds)
    assert profile.tichu_called == sum(len(r["tichu_callers"]) for r in rounds)
    assert profile.caller_pass_opportunities > 0 and profile.slam_for > 0
