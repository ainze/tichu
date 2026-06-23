"""[DEBUG-dog] Train/inference skew on current_player for the schupfen head.

Training pins public.current_player=0 for all 4 seats (schupfen_training.py),
while the engine advances it so at inference current_player == acting seat.
grand_tichu_callers is encoded ABSOLUTE, so the only way to localise the caller
relative to 'me' is (grand_caller - current_player) % 4. If the head is
insensitive to current_player, it cannot tell "partner called" from "opponent
called" -- it just keys on "grand present".

This probe holds seat=2, grand=[0] and sweeps public.current_player.
"""
import copy
import dataclasses
import json

from tichu_engine.cards import DOG
from tichu_inference.codec import private_state_from_json
from tichu_inference.ml_agent import MLAgent

BUNDLE = r"C:\workbench\tichu\data\runs\cotrain_v6_pbrs\export\iter_01200_resid_heads"

REQ = json.loads(r'''
{"player":2,"hand":[39,42,53,32,9,40,49,52,6,51,3,43,54,11],"public":{"current_player":2,"hand_sizes":[14,14,14,14],"scores":[225,575],"trick":{"plays":[],"leader":null,"passes":[]},"mahjong_wish":null,"pending_decision":{"kind":"schupfen","submitted":[[26,2,13],[28,31,16],null,null]},"round_points_by_player":[0,0,0,0],"out_order":[],"tichu_callers":[],"grand_tichu_callers":[0],"played_cards_by_player":[[],[],[],[]],"declined_top_by_player":[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],"lead_summary_by_player":[[0,0],[0,0],[0,0],[0,0]],"pass_stats_by_player":[[0,0],[0,0],[0,0],[0,0]]}}
''')
ps = private_state_from_json(REQ)

agent = MLAgent(
    checkpoint_path=fr"{BUNDLE}\policy.pt",
    schupfen_path=fr"{BUNDLE}\schupfen.pt",
    skill_decile=9,
)


def with_cp(ps, cp):
    pub2 = dataclasses.replace(ps.public, current_player=cp)
    return dataclasses.replace(ps, public=pub2)


def dog_partner_p(ps):
    sc = agent.schupfen_action_scores(ps)
    partner = dict((str(c), p) for c, p in sc["partner"])
    top = sc["partner"][0]
    return partner.get(str(DOG), 0.0), str(top[0]), top[1]


print("seat=2, grand_tichu_callers=[0] (=partner of seat 2)")
print(f"{'public.current_player':<24} {'P(Dog->partner)':>16}  partner-top")
for cp in range(4):
    dp, top_card, top_p = dog_partner_p(with_cp(ps, cp))
    note = ""
    if cp == 0:
        note = "  <- TRAINING moment (pinned)"
    if cp == 2:
        note = "  <- INFERENCE moment (engine-advanced; =acting seat)"
    print(f"{cp:<24} {dp:>16.4f}  {top_card}{note}")
