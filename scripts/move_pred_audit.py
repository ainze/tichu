"""Decile-9 move-prediction audit (ADR-0032 final instrument).

For every decision a decile-9 BSW human actually made (filtered via ratings_full),
replay the position and ask the v5 master what it would do. Reports top-1 / top-5
agreement per decision type, PLUS a disagreement map (how the master's move diverges
from the human's), so a systematic divergence can be located and then EV-probed.

Imitation fidelity != strength: a disagreement may be the master playing better.
A divergence is a candidate, confirmed only by a follow-up EV-probe. Caveat: games are
sampled from the corpus (likely in-sample for bc_full_corpus_v5), which *inflates*
agreement — so any divergence found is a conservative (real) signal.
"""
import sys
from collections import defaultdict

import pandas as pd

from tichu_engine.combinations import (
    FourOfAKindBomb, FullHouse, Pair, PairStep, Single, Straight, StraightFlushBomb, Triple,
)
from tichu_engine.legality import Pass
from tichu_training.bsw.archive import iter_archive, list_game_ids
from tichu_training.bsw.parser import parse_tch
from tichu_training.bsw.replay import replay_round

ARCHIVE = r"C:\workbench\tichu\data\archive.zst"
RATINGS = r"C:\workbench\tichu\data\ratings_full.parquet"
CKPT = r"C:\workbench\tichu\data\export\bc_full_corpus_v5_memmap_unshuffled"
RESULT = r"C:\workbench\tichu\data\runs\move_pred_audit\result.txt"
N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
TARGET_DECILE = 9

_KIND_TO_TYPE = {"play": "play", "pass": "play", "wish": "wish",
                 "dragon_give": "dragon_assignment", "schupfen": "schupfen"}


def _klass(a) -> str:
    if isinstance(a, Pass):
        return "Pass"
    if isinstance(a, Single):
        return "Single"
    if isinstance(a, Pair):
        return "Pair"
    if isinstance(a, Triple):
        return "Triple"
    if isinstance(a, FullHouse):
        return "FullHouse"
    if isinstance(a, Straight):
        return "Straight"
    if isinstance(a, PairStep):
        return "PairStep"
    if isinstance(a, (FourOfAKindBomb, StraightFlushBomb)):
        return "Bomb"
    return type(a).__name__


def _write(lines) -> None:
    import os
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w") as f:
        f.write("\n".join(lines) + "\n")
        f.flush()


def main() -> None:
    _write(["status=started loading-ratings"])
    ratings = pd.read_parquet(RATINGS, columns=["player_handle", "skill_decile"])
    decile = dict(zip(ratings.player_handle, ratings.skill_decile))

    _write(["status=loading-master"])
    from tichu_inference.ml_agent import MLAgent
    master = MLAgent(
        rf"{CKPT}\policy.pt", skill_decile=9,
        schupfen_path=rf"{CKPT}\schupfen.pt",
        tichu_call_path=rf"{CKPT}\tichu_call.pt",
        grand_call_path=rf"{CKPT}\grand_tichu_call.pt",
    )

    n = defaultdict(int)
    top1 = defaultdict(int)
    top5 = defaultdict(int)
    skipped = defaultdict(int)
    confusion = defaultdict(int)  # (human_klass, master_klass) for play disagreements

    game_ids = list_game_ids(ARCHIVE)[:N_GAMES]
    wanted = set(game_ids)
    done = 0
    for game_id, text in iter_archive(ARCHIVE, game_ids=wanted):
        done += 1
        try:
            game = parse_tch(text, game_id=game_id)
        except Exception:
            continue
        for rnd in game.rounds:
            try:
                result = replay_round(rnd)
            except Exception:
                continue
            for (parsed, human_action), state in zip(result.decisions, result.pre_decision_states):
                if state is None:
                    continue
                dt = _KIND_TO_TYPE.get(parsed.kind)
                if dt is None:
                    continue
                handle = rnd.handles[parsed.player]
                if decile.get(handle) != TARGET_DECILE:
                    continue
                pv = state.private_view(parsed.player)
                try:
                    chosen = master.act(pv)
                    ranked = master.rank_actions(pv)
                except Exception:
                    skipped[dt] += 1
                    continue
                n[dt] += 1
                hit1 = chosen == human_action
                if hit1:
                    top1[dt] += 1
                if ranked is not None and human_action in ranked[:5]:
                    top5[dt] += 1
                if dt == "play" and not hit1:
                    confusion[(_klass(human_action), _klass(chosen))] += 1
        if done % 200 == 0:
            tot = sum(n.values())
            _write([f"status=running games={done}/{len(game_ids)} decile9_decisions={tot}"])

    lines = [f"status=DONE games={done} target_decile={TARGET_DECILE}", "",
             "decision_type   n       top1     top5     skipped"]
    for dt in sorted(set(n) | set(skipped)):
        nn = n[dt]
        t1 = f"{top1[dt]/nn:.3f}" if nn else "n/a"
        t5 = f"{top5[dt]/nn:.3f}" if nn else "n/a"
        lines.append(f"{dt:<14s} {nn:<7d} {t1:<8s} {t5:<8s} {skipped[dt]}")
    lines += ["", "play disagreements (human_move -> master_move), top 20:"]
    for (hk, mk), c in sorted(confusion.items(), key=lambda kv: -kv[1])[:20]:
        lines.append(f"  {hk:>10s} -> {mk:<10s}  {c}")
    _write(lines)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        _write(["status=ERROR", traceback.format_exc()])
        raise
