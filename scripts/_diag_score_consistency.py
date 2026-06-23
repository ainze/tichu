"""[DEBUG-dog] Severity of the team_scores train/inference mismatch.

ALL training paths seed scores=(0,0) and play ONE round (replay._build_initial_state,
deal_initial_state, starting_position_pool), so team_scores is round-local (~0..125)
in training. At inference /act carries the GAME-cumulative score (e.g. 225/575).

If the served policy reacts to team_scores, the OOD input shifts play decisions.
This sweeps public.scores on a real post-schupfen state and reports whether the
top play action / its probability move.
"""
import dataclasses

from tichu_engine.state import deal_initial_state
from tichu_inference.ml_agent import MLAgent

BUNDLE = r"C:\workbench\tichu\data\runs\cotrain_v6_pbrs\export\iter_01200_resid_heads"

agent = MLAgent(checkpoint_path=fr"{BUNDLE}\policy.pt", skill_decile=9)

# A real post-schupfen play state (mahjong holder leads, pending=None, scores 0).
state = deal_initial_state(0)
seat = state.public.current_player
ps0 = state.private_view(seat)


def with_scores(ps, s):
    pub = dataclasses.replace(ps.public, scores=s)
    return dataclasses.replace(ps, public=pub)


SCORES = [(0, 0), (100, 0), (225, 575), (575, 225), (900, 50), (50, 900), (990, 990)]
print(f"play state: seat {seat} leads, {len(ps0.hand)} cards, pending=None\n")
print(f"{'public.scores':<16} {'top play action':<40} {'p_top':>7}  argmax_changed")
ref = None
for s in SCORES:
    scored = agent.play_action_scores(with_scores(ps0, s))
    top_a, top_p = scored[0]
    label = str(top_a)
    if len(label) > 38:
        label = label[:37] + "…"
    changed = "" if ref is None else ("YES" if label != ref else "no")
    if ref is None:
        ref = label
        note = "  <- TRAINING dist (round-local)"
    else:
        note = "  <- /act game-cumulative" if s == (225, 575) else ""
    print(f"{str(s):<16} {label:<40} {top_p:>7.4f}  {changed}{note}")
