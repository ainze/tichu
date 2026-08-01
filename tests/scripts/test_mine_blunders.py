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


# --- tier-2 reproducibility ------------------------------------------------
#
# A tier-2 verdict IS the label the correction corpus is built from (ADR-0042),
# so the same candidate must score the same way on every run. The failure this
# guards is invisible in-process: builtin hash() of a str is PYTHONHASHSEED-
# salted, and `spawn` re-draws the salt in every worker, so a salted seed looks
# perfectly deterministic until it crosses a process boundary.

import os
import subprocess
import sys

_VERDICT_PROBE = """
import json, random
from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_engine.legality import legal_actions_for
from tichu_ml.rule_agent import RuleAgent
import scripts.mine_blunders as mb
from tichu_training.search.blunder_miner import record_round

agents = [RuleAgent() for _ in range(4)]
mb._WORKER["agents"] = agents
pos = generate_full_position_pool(seed=11, n=1)[0]
_, decisions = record_round(agents, pos)
d = next(x for x in decisions
         if len(legal_actions_for(x.state.private_view(x.seat))) > 1)
alt = next(a for a in legal_actions_for(d.state.private_view(d.seat))
           if a != d.chosen)
cand = {"turn": d.turn, "chosen": repr(d.chosen), "alt": repr(alt)}
out = mb._tier2_task((5, pos, [cand], 3))
print(json.dumps([{k: r[k] for k in ("alt_win_rate", "mean_delta")} for r in out]))
"""


def _tier2_verdict_under(hash_seed: str) -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    env["PYTHONPATH"] = (os.path.join(root, "src") + os.pathsep + root
                         + os.pathsep + env.get("PYTHONPATH", ""))
    proc = subprocess.run([sys.executable, "-c", _VERDICT_PROBE],
                          capture_output=True, text=True, env=env, cwd=root)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()[-1]


def test_tier2_verdicts_reproduce_across_hash_salts():
    assert _tier2_verdict_under("0") == _tier2_verdict_under("12345")
