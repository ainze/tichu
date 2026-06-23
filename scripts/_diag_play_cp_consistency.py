"""[DEBUG-dog] Does the PLAY pipeline have schupfen's current_player skew?

For schupfen, training pinned public.current_player=0 while the acting seat
varied -> the head could not localise the grand caller. Play examples instead
come from real engine REPLAY (bc/dataset.py), so current_player should already
equal the acting player at every decision boundary (matching inference).

This replays real archive rounds and, for every state BC would featurize,
checks current_player == acting player, broken down by decision type. A 100%
match for play/wish/dragon means play has NO schupfen-style skew. Bomb
interrupts are out-of-turn BY DESIGN (the engine presents the same state at
inference), so they are reported separately, not as a defect.
"""
import sys
from collections import Counter

from tichu_training.bsw.archive import iter_archive
from tichu_training.bsw.parser import parse_tch
from tichu_training.bsw.replay import replay_round
from tichu_engine.legality import BombInterrupt

ARCHIVE = r"C:\workbench\tichu\data\archive.zst"
N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 300

_KIND = {"play": "play", "wish": "wish", "dragon_give": "dragon"}

match = Counter()
mismatch = Counter()
bomb = Counter()
n_games = 0

for stem, text in iter_archive(ARCHIVE):
    try:
        game = parse_tch(text, game_id=stem)
    except Exception:
        continue
    n_games += 1
    for rnd in game.rounds:
        rep = replay_round(rnd)
        if rep.final_state is None:
            continue
        for (parsed, concrete), pre in zip(rep.decisions, rep.pre_decision_states):
            if pre is None:
                continue
            dt = _KIND.get(parsed.kind)
            if dt is None:
                continue
            cp = pre.public.current_player
            actor = parsed.player
            if isinstance(concrete, BombInterrupt):
                bomb["bomb (out-of-turn by design)"] += 1
                continue
            if cp == actor:
                match[dt] += 1
            else:
                mismatch[dt] += 1
    if n_games >= N_GAMES:
        break

print(f"replayed {n_games} games\n")
print(f"{'decision type':<32} {'cp==actor':>10} {'cp!=actor':>10} {'mismatch %':>11}")
for dt in ("play", "wish", "dragon"):
    m, x = match[dt], mismatch[dt]
    tot = m + x
    pct = (100.0 * x / tot) if tot else 0.0
    print(f"{dt:<32} {m:>10} {x:>10} {pct:>10.3f}%")
for label, c in bomb.items():
    print(f"{label:<32} {'-':>10} {c:>10} {'(n/a)':>11}")
