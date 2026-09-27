r"""A/B the partner-trick bomb guard: guard OFF vs guard ON, same nets.

The guard rewrote a bomb over the PARTNER's winning trick into a Pass. It rode in
`MLAgent` (serving + the promotion gate; co-train rollouts sample the raw net) from
2026-06-09 until this A/B retired it on 2026-09-27 (v7 iter_15360, 2,000 deals:
+0.04/Round [-0.05, +0.16], 8 deals diverged). It now lives on only as the
`partner_trick_guard` probe (heuristic_probes.py), which is the guard-ON arm here.
It shipped on logic + a live observation (the 2026-06-09
probe was underpowered: 3 fires / 200 deals, CI [-4.8, +4.3]). That live observation
predates the Nuxt codec fix (trick STARTER sent as `trick.leader`), which both fakes
"bombed my partner" and inverts the guard.

Both arms load the identical export, so the seat-swap delta is purely the guard. MLAgent
is greedy (deterministic), so a deal where the guard never fires contributes a pair
mean of exactly 0; the power sits in the few deals that diverge, reported separately.

    delta = guard_off - guard_on        (> 0 means the guard costs EV -> remove it)

A serial frequency pass (guard OFF self-play) counts where the guard WOULD fire and
tags each fire's context (partner called / self called / anyone out yet).

    $env:PYTHONPATH = "<worktree>\src"
    py -3.14 -m scripts.ab_partner_trick_guard `
        --export-dir C:\workbench\tichu\data\runs\cotrain_v7_gated\export\iter_15360 `
        --pool C:\workbench\tichu\data\full_position_pool_s0_n20000.parquet `
        --n-deals 20000 --workers 10 --out C:\workbench\tichu\data\runs\ab_partner_trick_guard_v7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np

from tichu_engine.legality import legal_actions_for
from tichu_eval.full_position_pool import load_full_position_pool
from tichu_eval.play_full import play_full_round
from tichu_eval.tournament import _bootstrap_ci, collect_pair_deltas, pair_cluster

# Registers the `ml` factory + the probes (`partner_trick_guard`); its
# `_build_agent` is what spawn workers unpickle.
from tichu_training.cli import eval_matrix
from tichu_training.search.heuristic_probes import suppress_partner_trick_bomb

_EXPORT_FILES = {
    "checkpoint_path": "policy.pt",
    "schupfen_path": "schupfen.pt",
    "tichu_call_path": "tichu_call.pt",
    "grand_call_path": "grand_tichu_call.pt",
}


def _kwargs(export_dir: str, skill_decile: int) -> dict:
    d = Path(export_dir)
    kw = {k: str(d / f) for k, f in _EXPORT_FILES.items() if (d / f).is_file()}
    kw["skill_decile"] = skill_decile
    return kw


def _ci(x: np.ndarray, iters: int, seed: int) -> tuple[float, float, float]:
    """Mean + percentile-bootstrap CI over already-paired (per-deal) observations."""
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    return _bootstrap_ci(np.asarray(x, dtype=float), iters, np.random.default_rng(seed))


def _frequency(kwargs: dict, positions) -> dict:
    """Guard-OFF self-play; count Play Decisions where the guard would have fired."""
    agent = eval_matrix._build_agent("ml", **kwargs)
    ctx: Counter = Counter()
    deals_with_fire = 0
    fired_here = False

    def observer(seat, pv, action):
        nonlocal fired_here
        if pv.public.pending_decision is not None:
            return
        if not suppress_partner_trick_bomb(pv, action, list(legal_actions_for(pv))):
            return
        fired_here = True
        pub = pv.public
        partner = (seat + 2) % 4
        callers = pub.tichu_callers | pub.grand_tichu_callers
        ctx["fires"] += 1
        ctx["partner_called"] += partner in callers
        ctx["self_called"] += seat in callers
        ctx["opp_called"] += any((c % 2) != (seat % 2) for c in callers)
        ctx["nobody_out"] += len(pub.out_order) == 0
        ctx["partner_top_is_bomb"] += type(pub.trick.top_combination).__name__.endswith("Bomb")

    for pos in positions:
        fired_here = False
        play_full_round(tuple(agent for _ in range(4)), pos.state, pos.grand_prefixes,
                        observer=observer)
        deals_with_fire += fired_here
    return {"deals": len(positions), "deals_with_fire": deals_with_fire, **dict(ctx)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--export-dir", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--n-deals", type=int, default=20000)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--skill-decile", type=int, default=9)
    p.add_argument("--bootstrap-iters", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--freq-deals", type=int, default=2000,
                   help="Deals for the serial trigger-frequency pass (0 = skip).")
    p.add_argument("--out", required=True, help="Output dir (deltas.npz + summary.json).")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kwargs = _kwargs(args.export_dir, args.skill_decile)
    positions = load_full_position_pool(Path(args.pool))[: args.n_deals]
    print(f"ab-partner-trick-guard: {len(positions)} deals x seat-swap, "
          f"{args.workers} workers\n  subject: {args.export_dir}", flush=True)

    summary: dict = {"export_dir": args.export_dir, "pool": args.pool,
                     "n_deals": len(positions), "delta": "guard_off - guard_on"}

    if args.freq_deals:
        t0 = time.time()
        freq = _frequency(kwargs, positions[: args.freq_deals])
        summary["frequency"] = freq
        print(f"  frequency ({time.time() - t0:.0f}s): {json.dumps(freq)}", flush=True)

    t0 = time.time()
    off = partial(eval_matrix._build_agent, "ml", **kwargs)
    on = partial(eval_matrix._build_agent, "partner_trick_guard", **kwargs)
    total, bonus = collect_pair_deltas(off, on, positions, workers=args.workers)
    np.savez(out / "deltas.npz", total=total, bonus=bonus)

    per_deal = pair_cluster(total)
    per_deal_play = pair_cluster(total - bonus)
    diverged = per_deal != 0.0
    it, sd = args.bootstrap_iters, args.seed
    m, lo, hi = _ci(per_deal, it, sd)
    pm, plo, phi = _ci(per_deal_play, it, sd + 1)
    cm, clo, chi = _ci(per_deal[diverged], it, sd + 2)
    summary.update({
        "seconds": round(time.time() - t0),
        "per_round": {"mean": m, "lo": lo, "hi": hi},
        "per_round_play_only": {"mean": pm, "lo": plo, "hi": phi},
        "diverged_deals": int(diverged.sum()),
        "per_diverged_deal": {"mean": cm, "lo": clo, "hi": chi},
        "diverged_off_better": int((per_deal > 0).sum()),
        "diverged_on_better": int((per_deal < 0).sum()),
    })
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"  per Round (off-on): {m:+.3f}  CI=[{lo:+.3f}, {hi:+.3f}]  "
          f"(play-only {pm:+.3f} [{plo:+.3f}, {phi:+.3f}])", flush=True)
    print(f"  diverged deals: {int(diverged.sum())}/{len(per_deal)}  "
          f"per diverged deal: {cm:+.2f} [{clo:+.2f}, {chi:+.2f}]  "
          f"off better {summary['diverged_off_better']} / on better "
          f"{summary['diverged_on_better']}", flush=True)
    print(f"  wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
