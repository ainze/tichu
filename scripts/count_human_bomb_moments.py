r"""Issue #110: in human (BSW) games, when are out-of-turn bombs played?

Replays rounds through the engine and sorts every out-of-turn bomb by the moment
it came in: right after a Play (the Bomb Window contract covers it), after a
Pass, or into an empty Trick (both uncovered).

  py -3.14 -m scripts.count_human_bomb_moments --limit-games 5000
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", default=r"C:\workbench\tichu\data\archive.zst")
    p.add_argument("--limit-games", type=int, default=5000)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    from tichu_engine.engine import BombInterrupt
    from tichu_engine.legality import Pass
    from tichu_training.bsw.archive import iter_archive
    from tichu_training.bsw.parser import parse_tch
    from tichu_training.bsw.replay import replay_round

    c: Counter = Counter()
    for n, (stem, text) in enumerate(iter_archive(args.archive, game_ids=None)):
        if n >= args.limit_games:
            break
        try:
            game = parse_tch(text, game_id=stem)
        except Exception:  # noqa: BLE001
            continue
        for rnd in game.rounds:
            rep = replay_round(rnd)
            if rep.final_state is None:
                continue
            c["rounds"] += 1
            prev = None  # (actor, action) of the last engine-stepped play/pass/bomb
            for (parsed, action), pre in zip(rep.decisions, rep.pre_decision_states):
                if pre is None or parsed.kind not in ("play", "pass"):
                    continue
                if isinstance(action, BombInterrupt):
                    c["oot_bombs"] += 1
                    holder = pre.public.trick.leader
                    if pre.public.trick.top_combination is None:
                        kind = "empty_trick"
                    elif prev is not None and isinstance(prev[1], Pass):
                        kind = "after_pass"
                    elif prev is not None and prev[0] == parsed.player:
                        kind = "after_own_play"
                    else:
                        kind = "after_play_covered"
                    c[kind] += 1
                    if holder is not None:
                        same = holder % 2 == parsed.player % 2
                        c[f"{kind}_{'partner' if same else 'opponent'}_top"] += 1
                elif not isinstance(action, Pass) and type(action).__name__.endswith("Bomb"):
                    c["on_turn_bombs"] += 1
                prev = (parsed.player, action)
    rounds = max(1, c["rounds"])
    res = {"games": args.limit_games, "raw": dict(sorted(c.items())),
           "per_1000_rounds": {k: round(1000 * v / rounds, 2) for k, v in sorted(c.items())
                               if k != "rounds"}}
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
