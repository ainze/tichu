"""[DEBUG-dog] Confirm the retrained schupfen head fixed the current_player bug.

Compares champion (old) vs cpfix (new) schupfen on the flagged state:
  * P(Dog->partner) by WHO called grand (partner/self/opp) at correct current_player
  * P(Dog->partner) sweep over current_player
A fixed head should: drop sharply for opponent-callers, and VARY with current_player.
"""
import copy
import dataclasses
import json

from tichu_engine.cards import DOG
from tichu_inference.codec import private_state_from_json
from tichu_inference.ml_agent import MLAgent

CHAMP = r"C:\workbench\tichu\data\runs\cotrain_v6_pbrs_resid_wish_gated\export\champion"
CPFIX = r"C:\workbench\tichu\data\runs\schupfen_full_corpus_v6_resid_cpfix\export\schupfen.pt"

BASE = json.loads(r'''
{"player":2,"hand":[39,42,53,32,9,40,49,52,6,51,3,43,54,11],"public":{"current_player":2,"hand_sizes":[14,14,14,14],"scores":[225,575],"trick":{"plays":[],"leader":null,"passes":[]},"mahjong_wish":null,"pending_decision":{"kind":"schupfen","submitted":[[26,2,13],[28,31,16],null,null]},"round_points_by_player":[0,0,0,0],"out_order":[],"tichu_callers":[],"grand_tichu_callers":[0],"played_cards_by_player":[[],[],[],[]],"declined_top_by_player":[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],"lead_summary_by_player":[[0,0],[0,0],[0,0],[0,0]],"pass_stats_by_player":[[0,0],[0,0],[0,0],[0,0]]}}
''')

champ = MLAgent(checkpoint_path=fr"{CHAMP}\policy.pt", schupfen_path=fr"{CHAMP}\schupfen.pt", skill_decile=9)
fixed = MLAgent(checkpoint_path=fr"{CHAMP}\policy.pt", schupfen_path=CPFIX, skill_decile=9)


def dog_partner(agent, grand, cp):
    blob = copy.deepcopy(BASE)
    blob["public"]["grand_tichu_callers"] = list(grand)
    blob["public"]["current_player"] = cp
    ps = private_state_from_json(blob)
    sc = agent.schupfen_action_scores(ps)
    return dict((str(c), p) for c, p in sc["partner"]).get(str(DOG), 0.0)


print("P(Dog -> partner=player0).  player 2 schupfs; cp = acting seat = 2\n")
print(f"{'who called grand':<22} {'champion(old)':>14} {'cpfix(new)':>12}")
for label, grand in [("partner (p0)", [0]), ("self (p2)", [2]), ("opp (p1)", [1]), ("opp (p3)", [3]), ("nobody", [])]:
    print(f"{label:<22} {dog_partner(champ,grand,2):>14.3f} {dog_partner(fixed,grand,2):>12.3f}")

print("\nsweep current_player (grand=partner p0):")
print(f"{'cp':<22} {'champion(old)':>14} {'cpfix(new)':>12}")
for cp in range(4):
    print(f"{cp:<22} {dog_partner(champ,[0],cp):>14.3f} {dog_partner(fixed,[0],cp):>12.3f}")
