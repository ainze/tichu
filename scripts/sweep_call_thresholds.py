r"""sweep_call_thresholds — EV of the Tichu/Grand call decision boundary (2026-06-08).

The play policy has no blanket-rule fruit left (7 forced-action probes, all neutral/-EV),
so the remaining inference-time lever is the CALL decision. `MLAgent` exposes
`tichu_threshold` / `grand_threshold` (call iff P(call) >= threshold; default 0.5 == argmax;
LOWER => call more). The grand/schupfen critics under-fit (R^2 0.10/0.25), so the call head
may be miscalibrated — shifting the boundary is free EV with no retraining.

Each variant changes ONE threshold from the 0.5/0.5 baseline and tournaments vs that baseline
over the Full-strength Pool (which runs the Grand->Tichu call phase, so the change manifests).
Both sides wrap the SAME nets, so the delta is purely the boundary move. Reports per-Round
mean/CI/win-rate AND `call_bonus_mean` — the call-bonus component of the delta, so you can see
whether a change made more calls that PAID (mean up, call_bonus up) vs more calls that FAILED
(call_bonus up, mean down). An optional self-play pass reports call rate + success rate.

  * CI clears 0 -> that boundary beats argmax -> adopt it (set the threshold in the ml kwargs).
  * CI spans 0  -> the call head is already ~calibrated at 0.5 on that axis.
  * CI below 0  -> 0.5 is better than that shift.

Subject defaults to the config's eval.master; pass --export-dir to tune the cotrain snapshot.

    py -m scripts.sweep_call_thresholds --config configs/cotrain_wish_v5.yaml ^
        --export-dir C:\workbench\tichu\data\runs\cotrain_wish_v5\export\iter_06225
"""

import argparse
import sys
from functools import partial
from pathlib import Path

# Heavy imports (torch via eval_matrix, the tournament/pool/profiler) are loaded lazily
# inside main()/_builder so the pure helpers below stay unit-testable without torch.

_BASELINE = 0.5


def build_variants(tichu, grand, base_tichu, base_grand):
    """One variant per single-threshold change from (base_tichu, base_grand). Returns
    [(label, tichu_threshold, grand_threshold)], skipping any that equal the baseline."""
    out = []
    for t in tichu:
        if abs(t - base_tichu) > 1e-9:
            out.append((f"tichu{t:.2f}", t, base_grand))
    for g in grand:
        if abs(g - base_grand) > 1e-9:
            out.append((f"grand{g:.2f}", base_tichu, g))
    return out


def _subject_kwargs(config, export_dir):
    skill_decile = int(config.get("ppo", {}).get("skill_decile", 9))
    if export_dir:
        d = Path(export_dir)
        kwargs = {
            "checkpoint_path": str(d / "policy.pt"),
            "schupfen_path": str(d / "schupfen.pt"),
            "tichu_call_path": str(d / "tichu_call.pt"),
            "grand_call_path": str(d / "grand_tichu_call.pt"),
        }
        return kwargs, skill_decile, f"export-dir {d}"
    kwargs = dict(config["eval"]["master"])
    return kwargs, skill_decile, f"eval.master (baseline) {kwargs.get('checkpoint_path')}"


def _builder(subject, sd, t, g):
    from tichu_training.cli import eval_matrix  # registers `ml`; spawn-safe module-level fn

    return partial(eval_matrix._build_agent, "ml", skill_decile=sd,
                   tichu_threshold=t, grand_threshold=g, **subject)


def main(argv=None) -> int:
    import yaml

    from tichu_eval.full_position_pool import load_full_position_pool
    from tichu_eval.tournament import run_full_tournament

    p = argparse.ArgumentParser(description="Sweep MLAgent call thresholds for EV vs argmax(0.5)")
    p.add_argument("--config", required=True, help="Co-training config YAML (eval block + pool).")
    p.add_argument("--export-dir", default=None, help="Cotrain snapshot export dir (else eval.master).")
    p.add_argument("--tichu", type=float, nargs="+", default=[0.3, 0.4, 0.6, 0.7],
                   help="Tichu thresholds to test (grand held at baseline). Default 0.3 0.4 0.6 0.7.")
    p.add_argument("--grand", type=float, nargs="+", default=[0.3, 0.4, 0.6, 0.7],
                   help="Grand thresholds to test (tichu held at baseline).")
    p.add_argument("--base-tichu", type=float, default=_BASELINE)
    p.add_argument("--base-grand", type=float, default=_BASELINE)
    p.add_argument("--n-deals", type=int, default=None, help="Override eval n_deals.")
    p.add_argument("--workers", type=int, default=None, help="Override tournament workers.")
    p.add_argument("--bootstrap-iters", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--rate-positions", type=int, default=200,
                   help="Self-play deals for the call-rate context pass (0 to skip).")
    args = p.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    eval_cfg = config["eval"]
    subject, sd, subject_desc = _subject_kwargs(config, args.export_dir)
    workers = args.workers if args.workers is not None else int(eval_cfg.get("workers", 1))
    boot = args.bootstrap_iters if args.bootstrap_iters is not None else int(
        eval_cfg.get("bootstrap_iters", 1000))
    seed = args.seed if args.seed is not None else int(eval_cfg.get("seed", 0))

    positions = load_full_position_pool(Path(eval_cfg["starting_position_pool"]))
    cap = eval_cfg.get("n_deals") if args.n_deals is None else args.n_deals
    if cap is not None:
        positions = positions[: int(cap)]

    variants = build_variants(args.tichu, args.grand, args.base_tichu, args.base_grand)
    base_label = f"base({args.base_tichu:.2f}/{args.base_grand:.2f})"
    base_builder = _builder(subject, sd, args.base_tichu, args.base_grand)

    print(f"sweep-call-thresholds: {len(variants)} variants vs {base_label}  "
          f"({len(positions)} deals, {workers} workers, seed {seed})", flush=True)
    print(f"  subject: {subject_desc}", flush=True)
    print("  (threshold = call iff P(call) >= threshold; lower => call more)", flush=True)

    # Optional call-rate context (self-play): baseline + every variant in one profiling pass.
    if args.rate_positions:
        from tichu_eval.behavioral import run_behavioral_profiles
        rate_pos = positions[: args.rate_positions]
        builders = {base_label: base_builder}
        for label, t, g in variants:
            builders[label] = _builder(subject, sd, t, g)
        profiles = run_behavioral_profiles(builders, rate_pos)
        print(f"\ncall rates (self-play, {len(rate_pos)} deals):", flush=True)
        print(f"  {'agent':<16} {'tichu_call':>11} {'tichu_succ':>11} {'grand_call':>11} {'grand_succ':>11}",
              flush=True)
        for label in [base_label] + [v[0] for v in variants]:
            pr = profiles[label]
            print(f"  {label:<16} {pr.tichu_call_rate:>11.3f} {pr.tichu_success_rate:>11.3f} "
                  f"{pr.grand_call_rate:>11.3f} {pr.grand_success_rate:>11.3f}", flush=True)

    print(f"\nEV vs {base_label} (mean = variant - base, per round):", flush=True)
    print(f"  {'variant':<14} {'mean':>8} {'CI':>20} {'win%':>7} {'call_bonus':>11}  verdict", flush=True)
    for label, t, g in variants:
        builders = {base_label: base_builder, label: _builder(subject, sd, t, g)}
        result = run_full_tournament(builders, positions, bootstrap_iters=boot,
                                     seed=seed, workers=workers, progress=None)
        mean = float(result.mean(label, base_label))
        lo, hi = (float(x) for x in result.ci(label, base_label))
        win = float(result.win_rate(label, base_label))
        cb = float(result.call_bonus_mean(label, base_label))
        verdict = "*** +EV: adopt ***" if lo > 0 else ("neutral" if hi > 0 else "-EV")
        print(f"  {label:<14} {mean:>+8.2f} [{lo:>+7.2f},{hi:>+7.2f}] {win:>6.1%} {cb:>+11.2f}  {verdict}",
              flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
