"""Write a human-reviewable decision tape for an Agent's Play Decisions.

Plays N self-play Rounds (the chosen agent in all four seats) over a seeded
Full-strength Starting-Position Pool and dumps, per Play Decision, the public-state
context (scores, hand sizes, callers, wish, trick sequence), the hand, the chosen
play, and the top-k ranked legal alternatives with policy probabilities. Review the
tape to localise concrete tactical weaknesses. (The live `serve --tape-log` tape
additionally carries a `replay:` of each /act request; this offline tape has no
wire request to replay.)

Usage (reuses any eval config's `agents:` block; defaults to the first agent):
  py -3.14 scripts/decision_tape.py --config configs/eval_behavioral_v5.yaml `
    --agent master --rounds 5 --seed 0 --out C:\workbench\tichu\data\tape_master.txt
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import yaml

from tichu_eval.decision_tape import TapeSink, record_round, render_text
from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_ml.registry import build_agent
import tichu_inference.ml_agent  # noqa: F401 — registers the `ml` factory


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--agent", default=None, help="agent name from the config (default: first)")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    specs = {s["name"]: s for s in config["agents"]}
    name = args.agent or config["agents"][0]["name"]
    if name not in specs:
        raise SystemExit(f"agent {name!r} not in config (have {sorted(specs)})")
    spec = specs[name]
    builder = partial(build_agent, spec["factory"], **spec.get("kwargs", {}))

    agents = tuple(builder() for _ in range(4))   # self-play: same agent, 4 seats
    positions = generate_full_position_pool(seed=args.seed, n=args.rounds)
    sink = TapeSink()
    for i, pos in enumerate(positions):
        record_round(agents, pos, round_idx=i, sink=sink, top_k=args.top_k)

    header = (f"decision tape — agent={name}  rounds={args.rounds}  seed={args.seed}\n"
              f"{len(sink.records)} reviewable decisions (>=2 legal actions)\n")
    args.out.write_text(header + render_text(sink), encoding="utf-8")
    print(header + f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
