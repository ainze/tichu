r"""probe_heuristics — EV premise-test for the deep-research Tichu heuristics (2026-06-08).

For each heuristic guardrail probe (see `tichu_training.search.heuristic_probes`), wrap the
SAME subject policy, run a seat-swap Full-strength Tournament of `probe` vs the bare subject
over the Full-strength Pool, and report the per-Round mean/CI/win-rate. Because both sides
load the identical checkpoint, the delta is purely the forced intervention:

  * CI clears 0   -> the subject makes this mistake; the rule is free EV worth a guardrail.
  * CI spans 0    -> the subject already plays the rule (or it is EV-neutral); skip it.
  * CI below 0    -> the heuristic is wrong for this policy; leave it alone.

A separate in-process frequency pass reports how often each trigger fired, so a ~0 delta
is interpretable (handled-but-neutral vs never-fired). It reuses one probe instance across
all four self-play seats — a coarse "does it ever fire" signal, not the tournament's count.

The SUBJECT (the policy being probed) defaults to the config's `eval.master` block — but
that is usually the BC *reference baseline*, not your co-trained best agent. To probe the
actual cotrain snapshot, point `--export-dir` at its exported nets (the directory
`check_cotrain` writes, e.g. `<run_dir>/export/iter_NNNNN/`, holding policy.pt / schupfen.pt
/ tichu_call.pt / grand_tichu_call.pt). Run `check_cotrain` once first if no export exists.

Reuses the co-training config shape (`eval.starting_position_pool` + workers/n_deals/seed):

    # probe the BC reference baseline (eval.master):
    py -m scripts.probe_heuristics --config configs/cotrain_wish_v5.yaml
    # probe the actual cotrain_wish_v5 snapshot:
    py -m scripts.probe_heuristics --config configs/cotrain_wish_v5.yaml \
        --export-dir C:\workbench\tichu\data\runs\cotrain_wish_v5\export\iter_01500
    py -m scripts.probe_heuristics --config configs/cotrain_wish_v5.yaml \
        --probes forced_split_aces forced_follow_low --n-deals 4000 --workers 8
    py -m scripts.probe_heuristics --config configs/cotrain_wish_v5.yaml --freq-only
"""

import argparse
import sys
from functools import partial
from pathlib import Path

import yaml

from tichu_eval.full_position_pool import load_full_position_pool
from tichu_eval.play_full import play_full_round
from tichu_eval.tournament import run_full_tournament

# Importing eval_matrix registers the `ml` factory AND every probe, and exposes the
# module-level `_build_agent` the spawn workers resolve when they unpickle the partials.
from tichu_training.cli import eval_matrix
from tichu_training.search.heuristic_probes import _MasterProbe  # noqa: F401  (registers names)

ALL_PROBES = [
    "forced_split_aces",
    "forced_follow_low",
    "forced_support_tichu",
    "forced_keep_partner_trick",
    "forced_dragon_lastout",
    # Divergence-mined caller-pressure pair (2026-06-08).
    "forced_press_opp_caller",
    "forced_yield_opp_caller",
]


# Exported-net filenames MLAgent expects (mirrors check_cotrain._EXPORT_NAMES).
_EXPORT_FILES = {
    "checkpoint_path": "policy.pt",
    "schupfen_path": "schupfen.pt",
    "tichu_call_path": "tichu_call.pt",
    "grand_call_path": "grand_tichu_call.pt",
}


def _subject_kwargs(config, export_dir: str | None) -> tuple[dict, int, str]:
    """The policy under test. With --export-dir, wrap that snapshot's exported nets
    (the cotrain agent); otherwise fall back to the config's `eval.master` baseline."""
    skill_decile = int(config.get("ppo", {}).get("skill_decile", 9))
    if export_dir:
        d = Path(export_dir)
        kwargs = {key: str(d / fname) for key, fname in _EXPORT_FILES.items()}
        return kwargs, skill_decile, f"export-dir {d}"
    kwargs = dict(config["eval"]["master"])
    return kwargs, skill_decile, f"eval.master (baseline) {kwargs.get('checkpoint_path')}"


def _load_positions(config, n_deals: int | None):
    eval_cfg = config["eval"]
    positions = load_full_position_pool(Path(eval_cfg["starting_position_pool"]))
    cap = eval_cfg.get("n_deals") if n_deals is None else n_deals
    if cap is not None:
        positions = positions[: int(cap)]
    return positions


def _tournament(probe: str, subject_kwargs: dict, skill_decile: int, positions,
                *, workers: int, bootstrap_iters: int, seed: int) -> dict:
    builders = {
        "base": partial(eval_matrix._build_agent, "ml", skill_decile=skill_decile, **subject_kwargs),
        probe: partial(eval_matrix._build_agent, probe, skill_decile=skill_decile, **subject_kwargs),
    }
    result = run_full_tournament(
        builders, positions, bootstrap_iters=bootstrap_iters, seed=seed,
        workers=workers, progress=None,
    )
    lo, hi = (float(x) for x in result.ci(probe, "base"))
    return {
        "mean": float(result.mean(probe, "base")),
        "lo": lo, "hi": hi,
        "n": int(result.n(probe, "base")),
        "win_rate": float(result.win_rate(probe, "base")),
        "tie_rate": float(result.tie_rate(probe, "base")),
        "ship": lo > 0.0,
    }


def _frequency(probe: str, subject_kwargs: dict, skill_decile: int, positions) -> int:
    """Total trigger fires over `positions` in 4-seat self-play (one shared probe
    instance). A coarse opportunity signal, not the tournament's per-Round count."""
    agent = eval_matrix._build_agent(probe, skill_decile=skill_decile, **subject_kwargs)
    agent.interventions = 0
    for pos in positions:
        play_full_round(tuple(agent for _ in range(4)), pos.state, pos.grand_prefixes)
    return agent.interventions


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="EV premise-test for deep-research Tichu heuristics")
    p.add_argument("--config", required=True, help="Co-training config YAML (eval block + pool).")
    p.add_argument("--export-dir", default=None,
                   help="Directory of the subject's exported nets (policy.pt / schupfen.pt / "
                        "tichu_call.pt / grand_tichu_call.pt) — probe the cotrain snapshot "
                        "instead of the eval.master baseline.")
    p.add_argument("--probes", nargs="+", default=ALL_PROBES,
                   help=f"Subset of probes to run (default: all). Choices: {', '.join(ALL_PROBES)}")
    p.add_argument("--n-deals", type=int, default=None, help="Override eval n_deals (pool cap).")
    p.add_argument("--workers", type=int, default=None, help="Override tournament workers.")
    p.add_argument("--bootstrap-iters", type=int, default=None, help="Override bootstrap resamples.")
    p.add_argument("--seed", type=int, default=None, help="Override eval seed.")
    p.add_argument("--freq-positions", type=int, default=200,
                   help="Positions for the in-process frequency pass (default: 200; 0 to skip).")
    p.add_argument("--freq-only", action="store_true", help="Skip the tournament; only count fires.")
    args = p.parse_args(argv)

    unknown = [x for x in args.probes if x not in ALL_PROBES]
    if unknown:
        p.error(f"unknown probe(s): {unknown}. Choices: {ALL_PROBES}")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    eval_cfg = config["eval"]
    subject_kwargs, skill_decile, subject_desc = _subject_kwargs(config, args.export_dir)
    workers = args.workers if args.workers is not None else int(eval_cfg.get("workers", 1))
    bootstrap_iters = args.bootstrap_iters if args.bootstrap_iters is not None else int(
        eval_cfg.get("bootstrap_iters", 1000))
    seed = args.seed if args.seed is not None else int(eval_cfg.get("seed", 0))

    positions = _load_positions(config, args.n_deals)
    freq_positions = positions[: args.freq_positions] if args.freq_positions else []

    print(f"probe-heuristics: {len(args.probes)} probe(s) vs bare subject  "
          f"({len(positions)} deals, {workers} workers, seed {seed})", flush=True)
    print(f"  subject: {subject_desc}", flush=True)

    rows = []
    for probe in args.probes:
        fires = _frequency(probe, subject_kwargs, skill_decile, freq_positions) if freq_positions else None
        if args.freq_only:
            print(f"  {probe:<26s} fires={fires} over {len(freq_positions)} deals (4-seat self-play)",
                  flush=True)
            continue
        r = _tournament(probe, subject_kwargs, skill_decile, positions,
                        workers=workers, bootstrap_iters=bootstrap_iters, seed=seed)
        r["probe"], r["fires"] = probe, fires
        rows.append(r)
        verdict = "*** +EV: add guardrail ***" if r["ship"] else (
            "neutral (CI spans 0)" if r["hi"] > 0 >= r["lo"] else "-EV (rule hurts)")
        loss_rate = 1.0 - r["win_rate"] - r["tie_rate"]
        fires_str = "n/a" if fires is None else str(fires)
        print(
            f"  {probe:<26s} mean={r['mean']:+.2f}  CI=[{r['lo']:+.2f}, {r['hi']:+.2f}]  "
            f"win={r['win_rate']:.1%} vs {loss_rate:.1%} (tie {r['tie_rate']:.1%})  "
            f"fires={fires_str}  n={r['n']}  -> {verdict}", flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
