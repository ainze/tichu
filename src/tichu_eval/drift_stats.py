"""Behavioral Drift Benchmark — the statistics shared by every metric.

A metric reduces the log to per-deal `(events, opportunities)` counts per arm and
per cell (a distribution metric has one cell per category). From those this
module reports, per cell:

* **Exposure** — opportunities per 100 seat-Rounds (4 per deal: 2 Seat-Swap
  halves x 2 Subject seats), because a policy changes the situations it reaches;
* **Conditional Rate** — events / opportunities as a **ratio of sums** (never a
  mean of per-deal rates, which one-opportunity deals would dominate);
* a **paired Δ** (subject − bc) on the shared deals;
* percentile 95% CIs from a **deal-cluster bootstrap**: a deal is resampled
  whole — every Decision of both halves and both arms — because Decisions within
  a deal are correlated (the unpaired-bootstrap error behind the gate's false
  nulls, see project history). Resampling both arms with the same deal draw is
  what makes Δ paired.

See CONTEXT.md §"Exposure" / §"Conditional Rate".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEAT_ROUNDS_PER_DEAL = 4
ARMS = ("bc", "subject")


def summarise_cells(
    counts: pd.DataFrame, *, deals=None, n_boot: int = 2000, seed: int = 0,
    floor: int = 200,
) -> pd.DataFrame:
    """Tidy summary of one metric's per-deal counts.

    `counts` has columns `deal, arm, cell, events, opportunities` (a missing
    (deal, arm, cell) row means zero). `deals` is every deal of the run — deals
    where a cell never arose still count toward Exposure and are still resampled;
    it defaults to the deals present in `counts`. Returns one row per
    (cell, arm in {bc, subject, delta}); cells where either arm has fewer than
    `floor` opportunities are `suppressed` (rate and its CI NaN)."""
    deal_ids = np.array(sorted(set(deals) if deals is not None else set(counts.deal)))
    n = len(deal_ids)
    rng = np.random.default_rng(seed)
    # One shared resample for every cell and both arms: the pairing.
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)
    seat_rounds = SEAT_ROUNDS_PER_DEAL * n

    out: list[dict] = []
    if counts.empty:
        # The situation never arose in either arm: still report the metric, as
        # a suppressed row, so its absence reads "n too small" and not "no drift".
        nan = np.full(n_boot, np.nan)
        for arm in (*ARMS, "delta"):
            out.append(_row("all", arm, 0, 0, 0.0, np.zeros(n_boot), np.nan, nan, True))
        return pd.DataFrame(out)
    for cell, cell_counts in counts.groupby("cell", sort=False):
        e, o = _per_deal(cell_counts, deal_ids)          # (arms, deals) each
        e_tot, o_tot = e.sum(axis=1), o.sum(axis=1)
        e_bs, o_bs = weights @ e.T, weights @ o.T        # (boot, arms)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = e_tot / o_tot
            rate_bs = e_bs / o_bs
        expo = 100.0 * o_tot / seat_rounds
        expo_bs = 100.0 * o_bs / seat_rounds
        suppressed = bool((o_tot < floor).any())
        for a, arm in enumerate(ARMS):
            out.append(_row(cell, arm, e_tot[a], o_tot[a], expo[a], expo_bs[:, a],
                            rate[a], rate_bs[:, a], suppressed))
        d_rate_bs = rate_bs[:, 1] - rate_bs[:, 0]
        out.append(_row(cell, "delta", e_tot[1] - e_tot[0], o_tot[1] - o_tot[0],
                        expo[1] - expo[0], expo_bs[:, 1] - expo_bs[:, 0],
                        rate[1] - rate[0], d_rate_bs, suppressed,
                        p=_two_sided_p(d_rate_bs)))
    return pd.DataFrame(out)


def benjamini_hochberg(p_values, q: float = 0.05) -> np.ndarray:
    """Which hypotheses survive the Benjamini–Hochberg step-up at FDR `q`.
    NaN p-values (suppressed cells) never survive."""
    p = np.asarray(p_values, dtype=np.float64)
    keep = np.zeros(len(p), dtype=bool)
    valid = np.flatnonzero(~np.isnan(p))
    if not len(valid):
        return keep
    order = valid[np.argsort(p[valid])]
    m = len(order)
    passing = np.flatnonzero(p[order] <= q * np.arange(1, m + 1) / m)
    if len(passing):
        keep[order[: passing[-1] + 1]] = True
    return keep


def _per_deal(cell_counts: pd.DataFrame, deal_ids: np.ndarray):
    """Dense (arm, deal) event / opportunity matrices, zero where absent."""
    idx = {d: i for i, d in enumerate(deal_ids)}
    e = np.zeros((len(ARMS), len(deal_ids)))
    o = np.zeros((len(ARMS), len(deal_ids)))
    for a, arm in enumerate(ARMS):
        rows = cell_counts[cell_counts.arm == arm]
        cols = rows.deal.map(idx).to_numpy()
        np.add.at(e[a], cols, rows.events.to_numpy(dtype=np.float64))
        np.add.at(o[a], cols, rows.opportunities.to_numpy(dtype=np.float64))
    return e, o


def _row(cell, arm, events, opps, expo, expo_bs, rate, rate_bs, suppressed, p=np.nan):
    lo, hi = _ci(rate_bs)
    elo, ehi = _ci(expo_bs)
    if suppressed:
        rate, lo, hi, p = np.nan, np.nan, np.nan, np.nan
    return {
        "cell": cell, "arm": arm, "events": events, "opportunities": opps,
        "exposure": expo, "exposure_lo": elo, "exposure_hi": ehi,
        "rate": rate, "rate_lo": lo, "rate_hi": hi, "p": p, "suppressed": suppressed,
    }


def _ci(samples: np.ndarray) -> tuple[float, float]:
    s = samples[~np.isnan(samples)]
    if not len(s):
        return np.nan, np.nan
    lo, hi = np.percentile(s, [2.5, 97.5])
    return float(lo), float(hi)


def _two_sided_p(delta_bs: np.ndarray) -> float:
    """Bootstrap two-sided p for Δ ≠ 0, floored at 1/B (descriptive, for BH)."""
    s = delta_bs[~np.isnan(delta_bs)]
    if not len(s):
        return np.nan
    tail = min((s <= 0).mean(), (s >= 0).mean())
    return float(min(1.0, max(2.0 * tail, 1.0 / len(s))))


def summarise_reference(
    counts: pd.DataFrame, rounds: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0,
    floor: int = 200,
) -> pd.DataFrame:
    """Tidy summary of one metric's counts for an **unpaired reference** (the
    human games): rate and Exposure with 95% CIs, and no Δ / p / BH — it plays
    different deals against different opponents, so nothing pairs it with an arm.

    `rounds` is the reference log's Round table (`deal, half, game`, one row per
    measured team). The bootstrap resamples whole **Games**, since the same
    players recur across a Game's Rounds; Exposure divides by the seat-Rounds
    actually measured (2 per measured team-Round), which varies by Game when a
    skill filter measures only one team."""
    arm = rounds.arm.iloc[0]
    game_of = rounds.drop_duplicates("deal").set_index("deal").game
    games = pd.Index(sorted(set(rounds.game), key=str))
    seat_rounds = (2 * rounds.groupby("game").size()).reindex(games).to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    g = len(games)
    weights = rng.multinomial(g, np.full(g, 1.0 / g), size=n_boot).astype(np.float64)
    sr_bs = weights @ seat_rounds

    out: list[dict] = []
    if counts.empty:
        return pd.DataFrame([_ref_row("all", arm, 0, 0, np.nan, np.full(n_boot, np.nan),
                                      np.nan, np.full(n_boot, np.nan), True)])
    for cell, cell_counts in counts.groupby("cell", sort=False):
        per_game = (cell_counts.assign(game=cell_counts.deal.map(game_of))
                    .groupby("game")[["events", "opportunities"]].sum().reindex(games, fill_value=0))
        e = per_game.events.to_numpy(dtype=np.float64)
        o = per_game.opportunities.to_numpy(dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate, rate_bs = e.sum() / o.sum(), (weights @ e) / (weights @ o)
        expo, expo_bs = 100.0 * o.sum() / seat_rounds.sum(), 100.0 * (weights @ o) / sr_bs
        out.append(_ref_row(cell, arm, e.sum(), o.sum(), expo, expo_bs, rate, rate_bs,
                            bool(o.sum() < floor)))
    return pd.DataFrame(out)


def _ref_row(cell, arm, events, opps, expo, expo_bs, rate, rate_bs, suppressed) -> dict:
    return {**_row(cell, arm, events, opps, expo, expo_bs, rate, rate_bs, suppressed),
            "bh_survives": False}
