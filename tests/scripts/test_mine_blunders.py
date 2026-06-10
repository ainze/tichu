"""Pure-helper tests for scripts/mine_blunders.py (candidate selection + clustering)."""

import pandas as pd

from scripts.mine_blunders import cluster_report, select_tier2_candidates


def _tier1_df():
    return pd.DataFrame(
        [
            # round 0, turn 3: two alternatives — only the best should survive dedup
            dict(round_idx=0, turn=3, delta=40.0, role="follow", phase="mid",
                 caller_ctx="none", partner_leads=False, chosen_kind="Pass",
                 alt_kind="Single"),
            dict(round_idx=0, turn=3, delta=25.0, role="follow", phase="mid",
                 caller_ctx="none", partner_leads=False, chosen_kind="Pass",
                 alt_kind="Pair"),
            # below the delta bar
            dict(round_idx=0, turn=9, delta=5.0, role="lead", phase="late",
                 caller_ctx="none", partner_leads=False, chosen_kind="Single",
                 alt_kind="Pair"),
            # another round, clears the bar
            dict(round_idx=1, turn=2, delta=80.0, role="follow", phase="open",
                 caller_ctx="opp", partner_leads=True, chosen_kind="Pass",
                 alt_kind="Bomb:FourOfAKindBomb"),
        ]
    )


def test_select_tier2_candidates_dedups_and_filters():
    out = select_tier2_candidates(_tier1_df(), min_delta=15.0, top=10)
    assert len(out) == 2
    assert set(zip(out["round_idx"], out["turn"])) == {(0, 3), (1, 2)}
    # best alternative kept for the duplicated decision
    assert out[out["turn"] == 3]["delta"].iloc[0] == 40.0


def test_select_tier2_candidates_caps_at_top():
    out = select_tier2_candidates(_tier1_df(), min_delta=15.0, top=1)
    assert len(out) == 1
    assert out["delta"].iloc[0] == 80.0


def test_cluster_report_groups_and_ranks():
    verified = pd.DataFrame(
        [
            dict(role="follow", phase="mid", caller_ctx="none", partner_leads=False,
                 chosen_kind="Pass", alt_kind="Single", mean_delta=30.0,
                 alt_win_rate=0.9),
            dict(role="follow", phase="mid", caller_ctx="none", partner_leads=False,
                 chosen_kind="Pass", alt_kind="Single", mean_delta=50.0,
                 alt_win_rate=0.8),
            dict(role="lead", phase="late", caller_ctx="self", partner_leads=False,
                 chosen_kind="Single", alt_kind="Pair", mean_delta=20.0,
                 alt_win_rate=1.0),
        ]
    )
    report = cluster_report(verified)
    assert len(report) == 2
    top = report.iloc[0]
    assert top["n"] == 2 and top["mean_delta"] == 40.0 and top["alt_kind"] == "Single"
