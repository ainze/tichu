"""Behavior Sensitivity Probe — the analysis.

Each treatment arm (the champion with a Behavior Bias on one lever, ±δ) is paired
deal-for-deal with the shared reference arm (the unbiased champion) — both
Fixed-Opponent arms against the same opponents over the same Pool — and read
through the Behavioral Drift Benchmark's own statistics: the reference plays the
"bc" role and the treatment the "subject" role, so Δ = treatment − reference.
Per arm: ΔEV (points per Round), the lever rate it actually moved, and their
ratio (the slope). `verdicts` applies the pre-registered decision rules
(docs/notes/2026-09-27-behavior-sensitivity-preregistration.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tichu_eval.drift_arms import DriftLog
from tichu_eval.drift_metrics import metric, summarise

# Lever -> (drift metric, cell) holding the lever's Conditional Rate.
LEVER_METRICS: dict[str, tuple[str, str]] = {
    "pass_vs_opponent": ("pass_despite_beat", "opponent"),
    "phoenix_in_combination": ("phoenix_combination_when_legal", "all"),
    "dog_lead": ("dog_lead", "partner not called"),
    "single_lead": ("lead_type", "single"),
}

MIN_RATE_MOVE = 0.05        # below this the manipulation failed (5 pt)
GLOBAL_NULL_UPPER = 2.0     # every arm's ΔEV upper bound below this => global null


def pair_arms(reference: DriftLog, treatment: DriftLog) -> DriftLog:
    """One log the drift statistics can pair: reference as "bc", treatment as
    "subject"."""
    return DriftLog(
        decisions=pd.concat([reference.decisions.assign(arm="bc"),
                             treatment.decisions.assign(arm="subject")], ignore_index=True),
        rounds=pd.concat([reference.rounds.assign(arm="bc"),
                          treatment.rounds.assign(arm="subject")], ignore_index=True),
    )


def arm_effect(reference: DriftLog, treatment: DriftLog, lever: str, *,
               n_boot: int = 2000, seed: int = 0, floor: int = 200) -> dict:
    """ΔEV and Δ lever rate of one treatment arm against the reference."""
    name, cell = LEVER_METRICS[lever]
    panel = summarise(pair_arms(reference, treatment), [metric("round_points"), metric(name)],
                      n_boot=n_boot, seed=seed, floor=floor)

    def get(metric_name, cell_name, arm):
        row = panel[(panel.metric == metric_name) & (panel.cell == cell_name) & (panel.arm == arm)]
        return row.iloc[0]

    ev = get("round_points", "total", "delta")
    card = get("round_points", "card play", "delta")
    ref_rate, treat_rate = get(name, cell, "bc"), get(name, cell, "subject")
    d_rate = get(name, cell, "delta")
    return {
        "lever": lever,
        "d_ev": float(ev.rate), "d_ev_lo": float(ev.rate_lo), "d_ev_hi": float(ev.rate_hi),
        "p": float(ev.p),
        "d_card_play": float(card.rate),
        "rate_ref": float(ref_rate.rate), "rate_treat": float(treat_rate.rate),
        "d_rate": float(d_rate.rate), "d_rate_lo": float(d_rate.rate_lo),
        "d_rate_hi": float(d_rate.rate_hi),
        "exposure_ref": float(ref_rate.exposure), "exposure_treat": float(treat_rate.exposure),
        "slope": float(ev.rate) / (100.0 * float(d_rate.rate)) if d_rate.rate else float("nan"),
    }


def holm(p_values, alpha: float = 0.05) -> np.ndarray:
    """Holm–Bonferroni step-down: which hypotheses are rejected at family `alpha`.
    NaN p-values are never rejected (and do not count toward m)."""
    p = np.asarray(p_values, dtype=np.float64)
    keep = np.zeros(len(p), dtype=bool)
    valid = np.flatnonzero(~np.isnan(p))
    m = len(valid)
    for k, i in enumerate(valid[np.argsort(p[valid], kind="stable")]):
        if p[i] > alpha / (m - k):
            break
        keep[i] = True
    return keep


@dataclass
class Verdicts:
    levers: dict[str, str]
    failed_manipulation: list[tuple[str, str]] = field(default_factory=list)
    global_null: bool = False


def verdicts(arms: pd.DataFrame, *, alpha: float = 0.05) -> Verdicts:
    """The pre-registered reading of the arm table (columns lever, sign, d_ev,
    d_ev_lo, d_ev_hi, d_rate, p). Holm runs across every arm at once."""
    arms = arms.assign(significant=holm(arms.p.to_numpy(), alpha=alpha))
    gain = arms.significant & (arms.d_ev > 0)
    loss = arms.significant & (arms.d_ev < 0)
    out: dict[str, str] = {}
    for lever, g in arms.groupby("lever", sort=False):
        if gain[g.index].any():
            out[lever] = "lever found"
        elif loss[g.index].all():
            out[lever] = "sharp optimum"
        elif loss[g.index].any():
            out[lever] = "one-sided optimum"
        else:
            out[lever] = "flat optimum"
    failed = [(r.lever, r.sign) for r in arms.itertuples() if abs(r.d_rate) < MIN_RATE_MOVE]
    return Verdicts(levers=out, failed_manipulation=failed,
                    global_null=bool(not gain.any() and (arms.d_ev_hi < GLOBAL_NULL_UPPER).all()))
