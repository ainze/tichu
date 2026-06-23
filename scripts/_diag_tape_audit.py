"""[DEBUG-dog] Audit real /act requests from the champion tape against the
fields training produces. Verifies on REAL inference traffic:
  * team_scores range (round-local 0..125 vs game-cumulative)
  * current_player == acting player (per phase)
  * whether the v6 accumulators (played_by / declined_top / lead_summary /
    pass_stats / schupfen_received) are populated mid-game by the real client.
"""
import json
import sys
from collections import Counter

TAPE = r"C:\workbench\tichu\data\runs\cotrain_v6_pbrs_resid_wish_gated\champion_tape.log"
PREFIX = "replay:"

reqs = []
with open(TAPE, encoding="utf-8") as fh:
    for line in fh:
        i = line.find(PREFIX)
        if i == -1:
            continue
        reqs.append(json.loads(line[i + len(PREFIX):].strip()))

print(f"parsed {len(reqs)} /act requests\n")

phase_ct = Counter()
score_max = 0
score_vals = set()
cp_mismatch = Counter()
cp_total = Counter()
acc_populated = Counter()   # decisions where each accumulator is non-empty
acc_eligible = 0            # play decisions with >=1 card already played somewhere


def phase_of(pub):
    pd = pub.get("pending_decision")
    return pd["kind"] if pd else "play"


for r in reqs:
    ps = r["private_state"]
    pub = ps["public"]
    player = ps["player"]
    ph = phase_of(pub)
    phase_ct[ph] += 1

    s0, s1 = pub["scores"]
    score_max = max(score_max, abs(s0), abs(s1))
    score_vals.add((s0, s1))

    cp_total[ph] += 1
    if pub["current_player"] != player:
        cp_mismatch[ph] += 1

    # accumulator population (only meaningful once play is underway)
    played = pub.get("played_cards_by_player", [[], [], [], []])
    any_played = sum(len(x) for x in played)
    if ph == "play" and any_played > 0:
        acc_eligible += 1
        if any_played > 0:
            acc_populated["played_cards_by_player"] += 1
        if any(any(v) for v in pub.get("declined_top_by_player", [])):
            acc_populated["declined_top_by_player"] += 1
        if any(any(v) for v in pub.get("lead_summary_by_player", [])):
            acc_populated["lead_summary_by_player"] += 1
        if any(any(v) for v in pub.get("pass_stats_by_player", [])):
            acc_populated["pass_stats_by_player"] += 1
        if any(v is not None for v in ps.get("schupfen_received", [None, None, None])):
            acc_populated["schupfen_received"] += 1

print("phase distribution:", dict(phase_ct))
print(f"\nteam_scores: max |score| seen = {score_max}; distinct score pairs = {len(score_vals)}")
print("  sample score pairs:", sorted(score_vals)[:12])
print("\ncurrent_player == actor:")
for ph in cp_total:
    print(f"  {ph:<14} mismatches {cp_mismatch[ph]}/{cp_total[ph]}")
print(f"\nv6 accumulators on play decisions with cards already played (n={acc_eligible}):")
for k in ("played_cards_by_player", "declined_top_by_player", "lead_summary_by_player",
          "pass_stats_by_player", "schupfen_received"):
    print(f"  {k:<26} populated in {acc_populated[k]}/{acc_eligible}")
