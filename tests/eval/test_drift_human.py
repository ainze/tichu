"""Behavioral Drift Benchmark — the human reference, from replayed BSW games.

Human Rounds are replayed through the engine and every Decision is logged by the
SAME row builder the agents use, so a human rate is the same definition as the
agents' rate. See CONTEXT.md §"Behavioral Drift Benchmark".
"""

from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from tichu_training.bsw.parser import parse_tch

from tichu_eval.drift_human import human_log, record_human_round

_DATA = Path(__file__).resolve().parents[1] / "training" / "bsw" / "data"


def _games():
    return [parse_tch(f.read_text(encoding="utf-8", errors="replace"), game_id=f.stem)
            for f in sorted(_DATA.glob("*.tch"))]


def _logs():
    for g, game in enumerate(_games()):
        for r in game.rounds:
            logs = record_human_round(r, deal=r.round_index, game=g, subject_teams=(0, 1))
            if logs is not None:
                yield r, logs


def test_every_seat_logs_its_decisions_once():
    n = 0
    for parsed, (log0, _) in _logs():
        kinds = Counter((d["seat"], d["kind"]) for d in log0.decisions)
        for seat in range(4):
            assert kinds[(seat, "grand")] == 1 and kinds[(seat, "schupfen")] == 1
            assert kinds[(seat, "tichu")] <= 1
            if seat in parsed.grand_tichu_callers:
                assert kinds[(seat, "tichu")] == 0
        for d in log0.decisions:
            if d["kind"] == "tichu":
                assert d["called"] == (d["seat"] in parsed.tichu_callers)
            if d["kind"] == "grand":
                assert d["called"] == (d["seat"] in parsed.grand_tichu_callers)
        n += 1
    assert n >= 60   # nearly every fixture Round replays cleanly


def test_a_tichu_call_is_invisible_before_its_call_line():
    # The replay pre-populates EVERY Tichu caller from the start; a naive recorder
    # would show a partner as "called" before they called. Callers must appear in
    # log order.
    checked = 0
    for parsed, (log0, _) in _logs():
        plays = [d for d in log0.decisions if d["kind"] == "play"]
        for caller in parsed.tichu_callers:
            partner = (caller + 2) % 4
            # Non-Pass Play rows correspond one-to-one to the partner's "play" lines.
            partner_plays = [d for d in plays if d["seat"] == partner and d["action"] != "pass"]
            called_line = next(i for i, a in enumerate(parsed.plays)
                               if a.kind == "tichu" and a.player == caller)
            n_before = sum(1 for a in parsed.plays[:called_line]
                           if a.kind == "play" and a.player == partner)
            for d in partner_plays[:n_before]:
                assert not d["partner_called"]
                checked += 1
            for d in partner_plays[n_before:]:
                assert d["partner_called"]
    assert checked > 0


def test_round_row_matches_the_logged_result():
    slams = 0
    for parsed, (log0, log1) in _logs():
        r = log0.round
        assert (r["total_0"], r["total_1"]) == parsed.ergebnis
        assert r["subject_team"] == 0 and log1.round["subject_team"] == 1
        if r["slam_team"] is not None:
            slams += 1
            card = [parsed.ergebnis[t] - r[f"call_bonus_{t}"] for t in (0, 1)]
            assert sorted(card) == [0, 200]
        assert r["tricks_won"] is None   # replay cannot attribute Tricks
    assert slams > 0


def test_only_teams_meeting_the_skill_floor_are_measured():
    games = _games()
    handles = {h for g in games for r in g.rounds for h in r.handles if h}
    strong = set(sorted(handles)[::2])
    decile = {h: (9 if h in strong else 3) for h in handles}
    log = human_log(games, name="top", decile_of=decile, min_decile=9, n_rounds=10_000)
    assert len(log.rounds)
    by_round = {(g.game_id, r.round_index): r for g in games for r in g.rounds}
    for row in log.rounds.itertuples():
        parsed = by_round[(row.game, row.round_index)]
        team = row.subject_team
        assert all(decile.get(parsed.handles[s]) == 9 for s in (team, team + 2))
    assert set(log.rounds.arm) == {"top"}
    subject = log.decisions[log.decisions.is_subject]
    assert set(subject.seat % 2 == subject.half) == {True}


def test_all_humans_measures_both_teams_and_caps_the_sample():
    log = human_log(_games(), name="humans", n_rounds=20)
    assert log.rounds.deal.nunique() == 20
    assert set(log.rounds.half) == {0, 1}


def test_summarise_adds_the_reference_without_touching_deltas():
    from functools import partial

    from tichu_eval.drift_arms import run_drift_arms
    from tichu_eval.drift_metrics import METRICS, summarise
    from tichu_eval.full_position_pool import generate_full_position_pool
    from tichu_ml.rule_agent import RuleAgent

    from tests.eval.test_play_full import ScriptedCaller

    log = run_drift_arms(partial(ScriptedCaller, tichu=True), RuleAgent,
                         generate_full_position_pool(seed=0, n=10))
    games = _games()
    humans = human_log(games, name="humans", n_rounds=10_000)
    plain = summarise(log, n_boot=50, seed=0, floor=0)
    both = summarise(log, n_boot=50, seed=0, floor=0, references=[humans])

    ref = both[both.arm == "humans"]
    assert set(ref.metric) == {m.name for m in METRICS} | {"game"}
    agent = both[both.arm != "humans"].reset_index(drop=True)
    pd.testing.assert_frame_equal(agent[plain.columns], plain, check_dtype=False)

    # Trick attribution is unavailable from a replay: reported, but empty.
    assert ref[ref.metric == "trick_share"].suppressed.all()
    # Rounds per Game is the real Game length, not a Synthetic Game.
    rpg = ref[(ref.metric == "game") & (ref.cell == "rounds_per_game")].iloc[0]
    lengths = {}
    for g in games:
        t = [0, 0]
        for i, r in enumerate(g.rounds, 1):
            t = [t[0] + r.ergebnis[0], t[1] + r.ergebnis[1]]
            if max(t) >= 1000 and t[0] != t[1]:
                lengths[g.game_id] = i
                break
    assert rpg.rate == pytest.approx(sum(lengths.values()) / len(lengths))
