r"""probe_concrete_resolver — EV test for the bomb-preserving Concrete Resolver.

The v1 Action Space is intent-level, so every concrete realisation of an Intent
shares one logit and the pick among them is a tie-break the network cannot
express. `tichu_training.concrete_resolver` supplies that tie-break (keep bombs,
then keep same-suit runs); before it, the winner was whatever order the legal
frozenset happened to iterate in.

This is a cleaner A/B than the forced-action probes in `probe_heuristics`: both
sides load the IDENTICAL checkpoint and differ only in `bomb_preserving_resolver`,
so the delta is purely the tie-break. The resolver never changes WHICH Intent is
played, only which cards realise it.

  * CI clears 0   -> the arbitrary pick was costing EV; keep the resolver on.
  * CI spans 0    -> no measurable effect at this n (read WITH the trigger rate).
  * CI below 0    -> the cost function is picking the wrong variant.

The frequency pass is not optional colour: it reports how often a Play Decision
is even contested (top Intent with >1 realisation) and how often the resolver
actually overrides enumeration order. A null EV result means something quite
different at a 0.1% override rate than at 20%.

    py -m scripts.probe_concrete_resolver --config configs/cotrain_wish_v5.yaml \
        --export-dir C:\workbench\tichu\data\runs\cotrain_v6_pbrs_resid_wish_gated_cpfix\export\iter_03328
    py -m scripts.probe_concrete_resolver --config ... --export-dir ... --freq-only
"""

import argparse
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import yaml

from tichu_engine.legality import legal_actions_for
from tichu_eval.full_position_pool import load_full_position_pool
from tichu_eval.play_full import play_full_round
from tichu_eval.tournament import run_full_tournament

# Importing eval_matrix registers the `ml` factory and exposes the module-level
# `_build_agent` the spawn workers resolve when they unpickle the partials.
from tichu_training.cli import eval_matrix
from tichu_inference.ml_agent import MLAgent, _rank_legal_by_logits
from tichu_training.concrete_resolver import resolution_cost_fn

_EXPORT_FILES = {
    "checkpoint_path": "policy.pt",
    "schupfen_path": "schupfen.pt",
    "tichu_call_path": "tichu_call.pt",
    "grand_call_path": "grand_tichu_call.pt",
}


def _subject_kwargs(config, export_dir: str | None) -> tuple[dict, int, str]:
    skill_decile = int(config.get("ppo", {}).get("skill_decile", 9))
    if export_dir:
        d = Path(export_dir)
        return ({k: str(d / f) for k, f in _EXPORT_FILES.items()},
                skill_decile, f"export-dir {d}")
    kwargs = dict(config["eval"]["master"])
    return kwargs, skill_decile, f"eval.master (baseline) {kwargs.get('checkpoint_path')}"


def _tournament(subject_kwargs, skill_decile, positions, *, workers, bootstrap_iters, seed):
    """Resolver ON vs OFF over the same checkpoint, seat-swapped."""
    builders = {
        "resolver_on": partial(eval_matrix._build_agent, "ml", skill_decile=skill_decile,
                               bomb_preserving_resolver=True, **subject_kwargs),
        "resolver_off": partial(eval_matrix._build_agent, "ml", skill_decile=skill_decile,
                                bomb_preserving_resolver=False, **subject_kwargs),
    }
    result = run_full_tournament(
        builders, positions, bootstrap_iters=bootstrap_iters, seed=seed,
        workers=workers, progress=None,
    )
    lo, hi = (float(x) for x in result.ci("resolver_on", "resolver_off"))
    return {
        "mean": float(result.mean("resolver_on", "resolver_off")),
        "lo": lo, "hi": hi,
        "n": int(result.n("resolver_on", "resolver_off")),
        "win_rate": float(result.win_rate("resolver_on", "resolver_off")),
        "tie_rate": float(result.tie_rate("resolver_on", "resolver_off")),
    }


def _frequency(subject_kwargs, skill_decile, positions) -> Counter:
    """Audit every Play Decision of 4-seat self-play: how many are contested
    (the chosen Intent has more than one realisation), how often the resolver
    overrides enumeration order, and what it saves when it does."""
    agent = MLAgent(skill_decile=skill_decile, **subject_kwargs)
    stats: Counter = Counter()

    def observer(seat, private_state, action):
        if private_state.public.pending_decision is not None:
            return
        stats["play_decisions"] += 1
        legal = list(legal_actions_for(private_state))
        if len(legal) < 2:
            stats["forced"] += 1
            return
        try:
            logits = agent._play_logits(private_state)
        except Exception:  # noqa: BLE001 — audit is best-effort, never fails a round
            return
        cost = resolution_cost_fn(private_state.hand)
        on = _rank_legal_by_logits(legal, logits, cost)
        off = _rank_legal_by_logits(legal, logits, None)
        if not on or not off:
            return
        if on[0] != off[0]:
            stats["overridden"] += 1
            c_on, c_off = cost(on[0]), cost(off[0])
            if c_on[0] < c_off[0]:
                stats["bomb_saved"] += 1
            elif c_on[1] < c_off[1]:
                stats["run_saved"] += 1

    for pos in positions:
        play_full_round(tuple(agent for _ in range(4)), pos.state, pos.grand_prefixes,
                        observer=observer)
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="EV test for the bomb-preserving Concrete Resolver")
    p.add_argument("--config", required=True, help="Co-training config YAML (eval block + pool).")
    p.add_argument("--export-dir", default=None, help="Directory of the subject's exported nets.")
    p.add_argument("--n-deals", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--bootstrap-iters", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--freq-positions", type=int, default=200)
    p.add_argument("--freq-only", action="store_true")
    args = p.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    eval_cfg = config["eval"]
    subject_kwargs, skill_decile, subject_desc = _subject_kwargs(config, args.export_dir)
    workers = args.workers if args.workers is not None else int(eval_cfg.get("workers", 1))
    bootstrap_iters = (args.bootstrap_iters if args.bootstrap_iters is not None
                       else int(eval_cfg.get("bootstrap_iters", 1000)))
    seed = args.seed if args.seed is not None else int(eval_cfg.get("seed", 0))

    positions = load_full_position_pool(Path(eval_cfg["starting_position_pool"]))
    cap = eval_cfg.get("n_deals") if args.n_deals is None else args.n_deals
    if cap is not None:
        positions = positions[: int(cap)]

    print(f"probe-concrete-resolver: ON vs OFF, same checkpoint "
          f"({len(positions)} deals, {workers} workers, seed {seed})", flush=True)
    print(f"  subject: {subject_desc}", flush=True)

    if args.freq_positions:
        s = _frequency(subject_kwargs, skill_decile, positions[: args.freq_positions])
        rounds = args.freq_positions
        contested = s["overridden"]
        print(f"  trigger rate over {rounds} deals (4-seat self-play):", flush=True)
        print(f"    play decisions   {s['play_decisions']}  "
              f"({s['play_decisions'] / rounds:.1f}/round, {s['forced']} forced)", flush=True)
        print(f"    resolver changed the pick  {contested}  "
              f"({contested / max(rounds, 1):.2f}/round, "
              f"{contested / max(s['play_decisions'], 1):.2%} of decisions)", flush=True)
        print(f"      of which bomb saved={s['bomb_saved']}  run saved={s['run_saved']}",
              flush=True)

    if args.freq_only:
        return 0

    r = _tournament(subject_kwargs, skill_decile, positions,
                    workers=workers, bootstrap_iters=bootstrap_iters, seed=seed)
    verdict = ("*** +EV: keep the resolver on ***" if r["lo"] > 0 else
               "neutral (CI spans 0)" if r["hi"] > 0 else "-EV (resolver hurts)")
    loss_rate = 1.0 - r["win_rate"] - r["tie_rate"]
    print(f"  resolver_on vs resolver_off  mean={r['mean']:+.2f}  "
          f"CI=[{r['lo']:+.2f}, {r['hi']:+.2f}]  "
          f"win={r['win_rate']:.1%} vs {loss_rate:.1%} (tie {r['tie_rate']:.1%})  "
          f"n={r['n']}  -> {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
