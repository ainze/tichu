"""Behavioral Drift Benchmark — Synthetic Games.

Every current net is score-blind (round-local scores only), so a Game is a
sequence of independent Rounds: chaining benchmark Round results until a team
reaches 1000 is exact. See CONTEXT.md §"Synthetic Game".
"""

from functools import partial

import pytest

from tichu_eval.drift_arms import run_drift_arms
from tichu_eval.drift_games import synthetic_games
from tichu_eval.drift_metrics import summarise
from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_ml.rule_agent import RuleAgent

from tests.eval.test_play_full import ScriptedCaller


def test_a_game_ends_on_the_round_a_team_reaches_1000():
    games = synthetic_games([(600, 0), (500, 0), (10, 90)])
    assert [(g.rounds, g.subject_won, g.margin) for g in games] == [(2, True, 1100)]


def test_higher_total_wins_when_both_teams_cross_1000():
    (g,) = synthetic_games([(900, 0), (200, 1200)])
    assert (g.rounds, g.subject_won, g.margin) == (2, False, 1100 - 1200)


def test_a_tie_at_or_above_1000_plays_on():
    (g,) = synthetic_games([(500, 500), (500, 500), (100, 0)])
    assert (g.rounds, g.subject_won, g.margin) == (3, True, 100)


def test_an_unfinished_trailing_game_is_dropped():
    assert len(synthetic_games([(600, 0), (500, 0), (100, 0)])) == 1


def test_summarise_reports_game_metrics_with_a_paired_delta():
    log = run_drift_arms(partial(ScriptedCaller, tichu=True), RuleAgent,
                         generate_full_position_pool(seed=0, n=40))
    s = summarise(log, n_boot=100, seed=0, floor=0)
    g = s[s.metric == "game"]
    assert set(g.cell) == {"rounds_per_game", "win_rate", "margin"}
    assert set(g.arm) == {"bc", "subject", "delta"}
    rpg = g[(g.cell == "rounds_per_game") & (g.arm == "bc")].iloc[0]
    assert 1 < rpg.rate < 40 and rpg.rate_lo <= rpg.rate <= rpg.rate_hi


def test_each_deal_contributes_one_round_to_the_games():
    # Regression: the two Seat-Swap halves of a deal are the same cards with the
    # teams swapped — near-mirror images (exact mirrors in the BC arm). Chaining
    # both made consecutive Rounds cancel and games run ~1.2 Rounds long (BC 10.67
    # vs human 9.49). A real Game never replays a deal: one Round per deal.
    import pandas as pd

    from tichu_eval.drift_games import summarise_games

    rows = [{"arm": arm, "deal": d, "half": h, "subject_team": h,
             "total_0": 400 if h == 0 else 0, "total_1": 0 if h == 0 else 400}
            for arm in ("bc", "subject") for d in range(6) for h in (0, 1)]
    s = summarise_games(pd.DataFrame(rows), n_boot=20, seed=0)
    rpg = s[(s.cell == "rounds_per_game") & (s.arm == "bc")].iloc[0]
    # 6 deals -> 6 Rounds of +400 -> 2 Games of 3 Rounds (both halves: 4 Games).
    assert rpg.rate == 3 and rpg.opportunities == 2
