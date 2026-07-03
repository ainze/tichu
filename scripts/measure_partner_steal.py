r"""Measure the partner-overtake blunder head-to-head across agents.

Profiles `partner_steal_rate` (following a winning partner with a legal beat, the
fraction OVERTOOK instead of ceding) and `partner_steal_caller_rate` (the subset
where the partner is a Tichu/Grand caller — the sharp blunder). Runs each agent
guard-on and guard-off, because the shipped `suppress_partner_trick_bomb` guard
(ml_agent.py) only covers BOMBs — so the guard-on vs guard-off gap isolates the
bomb blunder, and the guard-on `partner_steal_rate` is the *non-bomb* overtake the
guard never touches.

Usage:
  $env:PYTHONPATH = "<worktree>\src"
  python scripts/measure_partner_steal.py `
      --champion C:\workbench\tichu\data\export\<cotrained_champion_dir> `
      --bc       C:\workbench\tichu\data\export\bc_full_corpus_v6_wishfix_memmap `
      --n 300
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _ml_builder(export_dir: str, *, skill_decile: int, guard: bool):
    """Preload one MLAgent from an export dir; the zero-arg builder returns the
    shared instance (act() is seat-agnostic, safe for serial self-play)."""
    from tichu_inference.ml_agent import MLAgent

    d = Path(export_dir)

    def opt(name: str):
        p = d / name
        return str(p) if p.is_file() else None

    agent = MLAgent(
        checkpoint_path=str(d / "policy.pt"),
        skill_decile=skill_decile,
        schupfen_path=opt("schupfen.pt"),
        tichu_call_path=opt("tichu_call.pt"),
        grand_call_path=opt("grand_tichu_call.pt"),
        partner_trick_guard=guard,
    )
    return lambda: agent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--champion", help="export dir of the cotrained champion")
    ap.add_argument("--bc", help="export dir of the BC baseline")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decile", type=int, default=9)
    args = ap.parse_args()

    from tichu_eval.behavioral import run_behavioral_profiles
    from tichu_eval.full_position_pool import generate_full_position_pool
    from tichu_ml.rule_agent import RuleAgent

    positions = generate_full_position_pool(seed=args.seed, n=args.n)

    builders: dict = {"rule": RuleAgent}
    if args.bc:
        builders["bc_guard_off"] = _ml_builder(args.bc, skill_decile=args.decile, guard=False)
        builders["bc_guard_on"] = _ml_builder(args.bc, skill_decile=args.decile, guard=True)
    if args.champion:
        builders["champion_guard_off"] = _ml_builder(args.champion, skill_decile=args.decile, guard=False)
        builders["champion_guard_on"] = _ml_builder(args.champion, skill_decile=args.decile, guard=True)

    profiles = run_behavioral_profiles(builders, positions)

    print(f"\npositions={args.n}  decile={args.decile}\n")
    hdr = (f"{'agent':22s} {'steal_rate':>10} {'steal_opps':>10} "
           f"{'caller_rate':>11} {'caller_opps':>11} {'bomb_legal_rate':>15}")
    print(hdr)
    print("-" * len(hdr))
    for name, p in profiles.items():
        r = p.as_row()
        print(f"{name:22s} {r['partner_steal_rate']:>10.4f} "
              f"{r['partner_steal_opportunities']:>10} "
              f"{r['partner_steal_caller_rate']:>11.4f} "
              f"{r['partner_steal_caller_opportunities']:>11} "
              f"{r['bomb_when_legal_rate']:>15.4f}")
    print("\nRead: guard-on partner_steal_rate = the NON-bomb overtake the shipped")
    print("guard does not cover; (guard-off - guard-on) ~ the bomb blunder it does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
