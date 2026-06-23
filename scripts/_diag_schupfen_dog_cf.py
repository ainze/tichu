"""[DEBUG-dog] Counterfactual probe: does the partner head CONDITION on the
grand-tichu call, or dump the Dog on partner unconditionally?

Varies who (if anyone) called grand tichu and reports the partner head's Dog
probability + whether the feature vector actually changes.
"""
import copy
import json

import numpy as np

from tichu_engine.cards import DOG
from tichu_inference.codec import private_state_from_json
from tichu_inference.ml_agent import MLAgent
from tichu_training import featurizer as fz

BUNDLE = r"C:\workbench\tichu\data\runs\cotrain_v6_pbrs\export\iter_01200_resid_heads"

BASE = json.loads(r'''
{"player":2,"hand":[39,42,53,32,9,40,49,52,6,51,3,43,54,11],"public":{"current_player":2,"hand_sizes":[14,14,14,14],"scores":[225,575],"trick":{"plays":[],"leader":null,"passes":[]},"mahjong_wish":null,"pending_decision":{"kind":"schupfen","submitted":[[26,2,13],[28,31,16],null,null]},"round_points_by_player":[0,0,0,0],"out_order":[],"tichu_callers":[],"grand_tichu_callers":[0],"played_cards_by_player":[[],[],[],[]],"declined_top_by_player":[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],"lead_summary_by_player":[[0,0],[0,0],[0,0],[0,0]],"pass_stats_by_player":[[0,0],[0,0],[0,0],[0,0]]},"schupfen_received":[null,null,null]}
''')

agent = MLAgent(
    checkpoint_path=fr"{BUNDLE}\policy.pt",
    schupfen_path=fr"{BUNDLE}\schupfen.pt",
    skill_decile=9,
)


def variant(label, grand_callers, tichu_callers=()):
    blob = copy.deepcopy(BASE)
    blob["public"]["grand_tichu_callers"] = list(grand_callers)
    blob["public"]["tichu_callers"] = list(tichu_callers)
    ps = private_state_from_json(blob)
    feats = fz.featurize(ps)
    scores = agent.schupfen_action_scores(ps)
    partner = dict((str(c), p) for c, p in scores["partner"])
    dog_p = partner.get(str(DOG), 0.0)
    top = scores["partner"][0]
    return label, feats, dog_p, str(top[0]), top[1]


rows = [
    variant("grand: partner(p0) called  [ORIGINAL]", grand_callers=[0]),
    variant("grand: nobody called", grand_callers=[]),
    variant("grand: SELF(p2) called", grand_callers=[2]),
    variant("grand: opp(p1) called", grand_callers=[1]),
    variant("grand: opp(p3) called", grand_callers=[3]),
    variant("small tichu: partner(p0) called", grand_callers=[], tichu_callers=[0]),
]

base_feats = rows[0][1]
print(f"{'variant':<42} {'P(Dog->partner)':>16}  {'partner top':<22} feat_delta_vs_orig")
for label, feats, dog_p, top_card, top_p in rows:
    delta = int(np.count_nonzero(feats != base_feats))
    print(f"{label:<42} {dog_p:>16.4f}  {top_card:<22} {delta}")
