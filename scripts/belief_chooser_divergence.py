r"""ADR-0041 Q-discriminability pre-gate — how often do the belief-on and
belief-off **Belief-Optimal Chooser**s actually pick different actions?

`Belief EV Value` (`D_on - D_off`) can only be non-zero at Decisions where the
two arms *diverge*. Measuring the divergence rate costs a pilot instead of two
paired tournaments, and it is the pre-check piKL lacked: that build shipped a
coverage gate but never asked whether its `Q` was discriminable, and lost 58
points/round to an argmax that was 32% unstable.

Both arms are scored at the **same** Decisions of the **same** champion self-play
Rounds, with common random numbers, so a divergence is attributable to the
marginals alone. Neither arm's choice is played out — this measures decision
disagreement, not EV.

  py -3.14 scripts/belief_chooser_divergence.py `
    --export-dir C:\workbench\tichu\data\runs\cotrain_v6_pbrs_resid_wish_gated_cpfix\export\iter_03328 `
    --belief C:\workbench\tichu\data\runs\belief_selfplay_v6\belief_h256.bin `
    --rounds 200 --workers 10
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

_WORKER: dict = {}


def _init_worker(export_dir: str, belief_path: str, skill_decile: int) -> None:
    import torch

    torch.set_num_threads(1)
    from tichu_inference.ml_agent import MLAgent
    from tichu_training.belief.selfplay_emit import load_belief_marginals_fn

    export = Path(export_dir)
    agent = MLAgent(
        export / "policy.pt",
        skill_decile=skill_decile,
        schupfen_path=export / "schupfen.pt",
        tichu_call_path=export / "tichu_call.pt",
        grand_call_path=export / "grand_tichu_call.pt",
    )
    _WORKER["agents"] = [agent] * 4
    _WORKER["belief_path"] = belief_path


def _divergence_task(task):
    """Replay one champion Round and, at each Decision of `seat`, ask both arms
    what they would play. Returns `(decisions, divergences, deviations_off,
    deviations_on)`."""
    from tichu_engine.legality import legal_actions_for
    from tichu_training.belief.rich_history import RichHistory
    from tichu_training.belief.selfplay_emit import load_belief_marginals_fn
    from tichu_training.search.belief_chooser import BeliefOptimalChooser
    from tichu_training.search.blunder_miner import record_round

    round_idx, position, seat, worlds, top_k, win, delta = task
    agents = _WORKER["agents"]
    _result, decisions = record_round(agents, position)

    # A Chooser sees only its own turns, so the replay owns the history: the
    # accumulator is advanced over EVERY Decision and read at the seat's own.
    history = RichHistory()
    marginals_fn = load_belief_marginals_fn(
        _WORKER["belief_path"], history_provider=history.block,
    )

    def arm(marginals_fn):
        return BeliefOptimalChooser(
            agents[seat], agents, marginals_fn=marginals_fn, worlds=worlds,
            top_k=top_k, win_threshold=win, delta_threshold=delta,
            seed=round_idx,  # common random numbers across the two arms
        )

    off, on = arm(None), arm(marginals_fn)
    n = diverged = 0
    for d in decisions:
        if d.seat != seat:
            history.update(d.seat, d.chosen, d.state)
            continue
        view = d.state.private_view(d.seat)
        # Count only Decisions the Chooser actually re-decides: a pending
        # Schupfen/Dragon/Wish and a forced single-legal-action move are passed
        # straight through by both arms and would dilute the rate.
        if view.public.pending_decision is not None or len(
            list(legal_actions_for(view))
        ) <= 1:
            history.update(d.seat, d.chosen, d.state)
            continue
        a_off, a_on = off.act(view), on.act(view)
        n += 1
        diverged += int(repr(a_off) != repr(a_on))
        history.update(d.seat, d.chosen, d.state)
    return n, diverged, off.deviations, on.deviations


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ADR-0041 chooser divergence pre-gate")
    p.add_argument("--export-dir", required=True)
    p.add_argument("--belief", required=True, type=Path)
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--pool-seed", type=int, default=909000)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--skill-decile", type=int, default=9)
    p.add_argument("--seat", type=int, default=0)
    p.add_argument("--worlds", type=int, default=24)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--win-threshold", type=float, default=0.70)
    p.add_argument("--delta-threshold", type=float, default=15.0)
    args = p.parse_args(argv)

    from tichu_eval.full_position_pool import generate_full_position_pool

    positions = generate_full_position_pool(seed=args.pool_seed, n=args.rounds)
    tasks = [
        (i, pos, args.seat, args.worlds, args.top_k,
         args.win_threshold, args.delta_threshold)
        for i, pos in enumerate(positions)
    ]

    n = diverged = dev_off = dev_on = 0
    with mp.Pool(
        processes=args.workers, initializer=_init_worker,
        initargs=(str(args.export_dir), str(args.belief), args.skill_decile),
    ) as pool:
        for i, (a, b, c, d) in enumerate(
            pool.imap_unordered(_divergence_task, tasks, chunksize=1), 1
        ):
            n += a
            diverged += b
            dev_off += c
            dev_on += d
            if i % 20 == 0:
                print(f"  {i}/{len(tasks)} rounds — {n} decisions, "
                      f"{diverged} divergent", flush=True)

    print(f"\ndecisions scored:        {n}")
    print(f"belief-off deviations:   {dev_off}  ({dev_off / max(1, n):.4f})")
    print(f"belief-on  deviations:   {dev_on}  ({dev_on / max(1, n):.4f})")
    print(f"ARM DIVERGENCE:          {diverged}  ({diverged / max(1, n):.4f})")
    print("\nA divergence rate near zero means `D_on - D_off` is zero by "
          "construction and the paired tournament cannot resolve anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
