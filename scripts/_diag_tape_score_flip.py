"""[DEBUG-dog] Does the team_scores mismatch change REAL served decisions?

Replays every play-decision /act from the champion tape through the policy
twice: once with the real game-cumulative scores (what /act sends, OOD for the
net) and once with scores zeroed (the training distribution). Counts how many
top actions flip — i.e. decisions the score mismatch actually altered.
"""
import dataclasses
import json

from tichu_inference.codec import private_state_from_json
from tichu_inference.ml_agent import MLAgent

TAPE = r"C:\workbench\tichu\data\runs\cotrain_v6_pbrs_resid_wish_gated\champion_tape.log"
BUNDLE = r"C:\workbench\tichu\data\runs\cotrain_v6_pbrs\export\iter_01200_resid_heads"

agent = MLAgent(checkpoint_path=fr"{BUNDLE}\policy.pt", skill_decile=9)

plays = []
with open(TAPE, encoding="utf-8") as fh:
    for line in fh:
        i = line.find("replay:")
        if i == -1:
            continue
        req = json.loads(line[i + len("replay:"):].strip())
        ps_blob = req["private_state"]
        if ps_blob["public"].get("pending_decision") is not None:
            continue  # play decisions only
        plays.append(ps_blob)


def zero_scores(ps):
    pub = dataclasses.replace(ps.public, scores=(0, 0))
    return dataclasses.replace(ps, public=pub)


flips = 0
nonzero_score = 0
flip_examples = []
for blob in plays:
    ps_real = private_state_from_json(blob)
    if any(ps_real.public.scores):
        nonzero_score += 1
    real_top = agent.play_action_scores(ps_real)[0]
    zero_top = agent.play_action_scores(zero_scores(ps_real))[0]
    if str(real_top[0]) != str(zero_top[0]):
        flips += 1
        if len(flip_examples) < 6:
            flip_examples.append(
                (ps_real.public.scores, str(real_top[0]), real_top[1],
                 str(zero_top[0]), zero_top[1])
            )

print(f"play decisions: {len(plays)}  (with non-zero score: {nonzero_score})")
print(f"top action FLIPS when score corrected to training dist: {flips}/{len(plays)} "
      f"({100.0 * flips / len(plays):.1f}%)\n")
for sc, ra, rp, za, zp in flip_examples:
    print(f"  scores={sc[0]}/{sc[1]}")
    print(f"    /act (real score):   {ra[:46]:<46} p={rp:.2f}")
    print(f"    zeroed (train dist): {za[:46]:<46} p={zp:.2f}")
