"""[DEBUG-dog] Reproduce the schupfen 'give Dog to grand-tichu partner' decision.

Loads the MASTER tier (skill_decile=9) schupfen net from the residual bundle and
replays the exact /act request the user flagged. Prints:
  * per-direction head ranking (next / partner / previous)
  * the SchupfenPass the agent actually emits, decoded to player ids + card names.
"""
import json
import sys

from tichu_inference.codec import private_state_from_json, card_to_id, id_to_card
from tichu_inference.ml_agent import MLAgent

BUNDLE = r"C:\workbench\tichu\data\runs\cotrain_v6_pbrs\export\iter_01200_resid_heads"

REQ = json.loads(r'''
{"difficulty":"master","private_state":{"player":2,"hand":[39,42,53,32,9,40,49,52,6,51,3,43,54,11],"public":{"current_player":2,"hand_sizes":[14,14,14,14],"scores":[225,575],"trick":{"plays":[],"leader":null,"passes":[]},"mahjong_wish":null,"pending_decision":{"kind":"schupfen","submitted":[[26,2,13],[28,31,16],null,null]},"round_points_by_player":[0,0,0,0],"out_order":[],"tichu_callers":[],"grand_tichu_callers":[0],"played_cards_by_player":[[],[],[],[]],"declined_top_by_player":[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],"lead_summary_by_player":[[0,0],[0,0],[0,0],[0,0]],"pass_stats_by_player":[[0,0],[0,0],[0,0],[0,0]]},"schupfen_received":[null,null,null]}}
''')

ps = private_state_from_json(REQ["private_state"])
player = ps.player
DIR_TO_PLAYER = {
    "next": (player + 1) % 4,
    "partner": (player + 2) % 4,
    "previous": (player + 3) % 4,
}

agent = MLAgent(
    checkpoint_path=fr"{BUNDLE}\policy.pt",
    schupfen_path=fr"{BUNDLE}\schupfen.pt",
    skill_decile=9,
)

print(f"player={player}  partner=player {DIR_TO_PLAYER['partner']}  "
      f"grand_tichu_callers={sorted(ps.public.grand_tichu_callers)}")
print("hand:", [f"{c}({card_to_id(c)})" for c in sorted(ps.hand, key=card_to_id)])
print()

scores = agent.schupfen_action_scores(ps)
for name in ("next", "partner", "previous"):
    tgt = DIR_TO_PLAYER[name]
    print(f"--- head '{name}' -> player {tgt} ---")
    for card, p in scores[name][:6]:
        print(f"    {str(card):<24} id={card_to_id(card):>2}  p={p:.4f}")
    print()

action = agent._act_schupfen(ps)
print("CHOSEN SchupfenPass:")
for name, card in (("next", action.to_next),
                   ("partner", action.to_partner),
                   ("previous", action.to_previous)):
    tgt = DIR_TO_PLAYER[name]
    print(f"    to_{name:<8} -> player {tgt}: {card} (id={card_to_id(card)})")
