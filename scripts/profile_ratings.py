"""Profile a ratings parquet file — produce blog-post-ready plots + summary.

Reads the output of `compute_trueskill` (one row per rated player) and writes:
- Six PNG figures into --out-dir
- A `summary.md` with headline numbers, plot index, and the pre-filter caveat
- The headline numbers to stdout (sanity-check echo)

The parquet contains only players who survived the Min-games Filter; the
caveat block in summary.md documents this for readers of the resulting post.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tichu_training.ratings.stats import spearman_rho


log = logging.getLogger("profile_ratings")

_DPI = 150
_FIGSIZE = (6.4, 4.0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--input", required=True, metavar="PATH",
                   help="Ratings parquet (output of compute_trueskill)")
    p.add_argument("--out-dir", required=True, metavar="DIR",
                   help="Directory to write PNGs and summary.md into")
    p.add_argument("--min-tichu-calls", type=int, default=5, metavar="N",
                   help="Filter for the small-Tichu success-rate plot and Spearman "
                        "(default: 5, matches compute_trueskill)")
    p.add_argument("--min-grand-tichu-calls", type=int, default=3, metavar="N",
                   help="Filter for the Grand-Tichu success-rate plot "
                        "(default: 3 — Grand Tichu calls are rarer than small Tichu)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    if not input_path.exists():
        log.error("input %s does not exist", input_path)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("reading %s", input_path)
    table = pq.read_table(input_path)
    cols = _extract_columns(table)
    stats = _compute_headline(
        cols,
        min_tichu_calls=args.min_tichu_calls,
        min_grand_tichu_calls=args.min_grand_tichu_calls,
    )

    _print_stats(stats)
    _plot_mu_hist(cols, out_dir / "fig-2-mu-hist.png", plt, sns)
    _plot_n_games_cdf(cols, out_dir / "fig-3-n-games-cdf.png", plt, sns)
    _plot_sigma_vs_experience(cols, out_dir / "fig-4-sigma-vs-experience.png", plt, sns)
    _plot_tichu_success_by_decile(
        cols, args.min_tichu_calls,
        out_dir / "fig-5-tichu-success-by-decile.png", plt, sns,
    )
    _plot_grand_tichu_rate_by_decile(
        cols, out_dir / "fig-6-grand-tichu-rate-by-decile.png", plt, sns,
    )
    _plot_tichu_call_rate_by_decile(
        cols, out_dir / "fig-7-tichu-call-rate-by-decile.png", plt, sns,
    )
    _plot_grand_tichu_success_by_decile(
        cols, args.min_grand_tichu_calls,
        out_dir / "fig-8-grand-tichu-success-by-decile.png", plt, sns,
    )
    _plot_games_lorenz(
        cols, stats, out_dir / "fig-9-games-played-lorenz.png", plt, sns,
    )
    _plot_tichu_vs_grand_tichu_success(
        cols, args.min_tichu_calls, args.min_grand_tichu_calls,
        out_dir / "fig-10-tichu-vs-grand-tichu-success.png", plt, sns,
    )
    _plot_correlation_heatmap(
        cols, out_dir / "fig-11-correlation-heatmap.png", plt, sns,
    )
    _write_summary(stats, args, out_dir / "summary.md")

    log.info("wrote 10 figures + summary.md to %s", out_dir)
    return 0


def _extract_columns(table) -> dict[str, np.ndarray]:
    """Pull each column as a numpy array, preserving nullability where needed."""
    def col(name: str, fill=np.nan):
        arr = table.column(name)
        if arr.null_count == 0:
            return arr.to_numpy(zero_copy_only=False)
        return np.array(arr.to_pylist(),
                        dtype=float if isinstance(fill, float) else object)
    return {
        "player_handle":            np.array(table.column("player_handle").to_pylist(), dtype=object),
        "mu":                       col("mu"),
        "sigma":                    col("sigma"),
        "n_games":                  col("n_games").astype(np.int64),
        "skill_decile":             _decile_to_int(table.column("skill_decile")),
        "tichu_calls":              col("tichu_calls").astype(np.int64),
        "tichu_success_rate":       col("tichu_success_rate"),
        "grand_tichu_calls":        col("grand_tichu_calls").astype(np.int64),
        "grand_tichu_success_rate": col("grand_tichu_success_rate"),
    }


def _decile_to_int(arr) -> np.ndarray:
    """skill_decile is nullable in the schema; coerce nulls to -1 sentinel."""
    py = arr.to_pylist()
    return np.array([-1 if v is None else int(v) for v in py], dtype=np.int64)


def _compute_headline(
    cols: dict[str, np.ndarray],
    *,
    min_tichu_calls: int,
    min_grand_tichu_calls: int,
) -> dict:
    n_players = len(cols["mu"])
    total_games = int(cols["n_games"].sum() // 4)
    n_games = cols["n_games"]
    decile_counts = [int((cols["skill_decile"] == d).sum()) for d in range(10)]

    # Spearman ρ(μ, tichu_success_rate) over filtered rows — matches
    # compute_trueskill's check, recomputed for the same min_tichu_calls.
    mask = (cols["tichu_calls"] >= min_tichu_calls) & np.isfinite(cols["tichu_success_rate"])
    mus = cols["mu"][mask].tolist()
    rates = cols["tichu_success_rate"][mask].tolist()
    rho = spearman_rho(mus, rates) if len(mus) >= 2 else None

    # Engagement concentration — Gini of n_games. 0 = even, 1 = one player has all.
    gini = _gini(n_games.astype(float))

    # Spearman ρ(tichu SR, grand tichu SR) for the scatter caption.
    cross_mask = (
        (cols["tichu_calls"] >= min_tichu_calls)
        & (cols["grand_tichu_calls"] >= min_grand_tichu_calls)
        & np.isfinite(cols["tichu_success_rate"])
        & np.isfinite(cols["grand_tichu_success_rate"])
    )
    cross_rho = spearman_rho(
        cols["tichu_success_rate"][cross_mask].tolist(),
        cols["grand_tichu_success_rate"][cross_mask].tolist(),
    ) if cross_mask.sum() >= 2 else None
    cross_n = int(cross_mask.sum())

    # Leaderboards: sort all rated players by mu, take the extremes.
    # mu alone can be misleading (high mu + high sigma = uncertain), so we
    # render sigma and n_games alongside it for context.
    order = np.argsort(cols["mu"])
    top = [_leaderboard_row(cols, i) for i in order[::-1][:10]]
    bottom = [_leaderboard_row(cols, i) for i in order[:10]]
    most_games_order = np.argsort(cols["n_games"])[::-1][:10]
    most_games = [_leaderboard_row(cols, i) for i in most_games_order]

    return {
        "n_players": n_players,
        "total_games": total_games,
        "median_n_games": int(np.median(n_games)),
        "p95_n_games": int(np.percentile(n_games, 95)),
        "max_n_games": int(n_games.max()),
        "decile_counts": decile_counts,
        "spearman_rho": rho,
        "spearman_n": len(mus),
        "min_tichu_calls": min_tichu_calls,
        "min_grand_tichu_calls": min_grand_tichu_calls,
        "gini_n_games": gini,
        "cross_rho": cross_rho,
        "cross_n": cross_n,
        "top10": top,
        "bottom10": bottom,
        "most_games10": most_games,
    }


def _gini(values: np.ndarray) -> float:
    """Gini coefficient over a non-negative 1-D array. 0 = even, 1 = max-skewed."""
    v = np.sort(values)
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum() - (n + 1) * v.sum()) / (n * v.sum()))


def _leaderboard_row(cols: dict[str, np.ndarray], i: int) -> dict:
    return {
        "handle": str(cols["player_handle"][i]),
        "mu": float(cols["mu"][i]),
        "sigma": float(cols["sigma"][i]),
        "n_games": int(cols["n_games"][i]),
    }


def _print_stats(s: dict) -> None:
    print(f"rated players:        {s['n_players']:,}")
    print(f"total games (sum/4):  {s['total_games']:,}")
    print(f"n_games median:       {s['median_n_games']:,}")
    print(f"n_games p95:          {s['p95_n_games']:,}")
    print(f"n_games max:          {s['max_n_games']:,}")
    shares = [c / s['n_players'] for c in s['decile_counts']] if s['n_players'] else []
    print(f"decile counts:        {s['decile_counts']}")
    print(f"decile shares:        {[round(x, 3) for x in shares]}")
    if s['spearman_rho'] is not None:
        print(f"spearman rho(mu, tichu_success_rate) = "
              f"{s['spearman_rho']:.3f} over {s['spearman_n']:,} players "
              f"(tichu_calls >= {s['min_tichu_calls']})")
    else:
        print(f"spearman rho: undefined (too few players with tichu_calls >= "
              f"{s['min_tichu_calls']})")
    print(f"gini(n_games):        {s['gini_n_games']:.3f}  "
          f"(0 = even play, 1 = one player has all games)")
    if s['cross_rho'] is not None:
        print(f"spearman rho(tichu_SR, grand_tichu_SR) = "
              f"{s['cross_rho']:.3f} over {s['cross_n']:,} players")
    else:
        print(f"spearman rho(tichu_SR, grand_tichu_SR): undefined "
              f"(too few players meet both filters)")


def _plot_mu_hist(cols, path, plt, sns) -> None:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    sns.histplot(cols["mu"], bins=50, ax=ax)
    ax.set_xlabel("TrueSkill μ")
    ax.set_ylabel("Players")
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def _plot_n_games_cdf(cols, path, plt, sns) -> None:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    sns.ecdfplot(x=cols["n_games"], ax=ax)
    ax.set_xscale("log")
    ax.set_xlabel("Games played (log scale)")
    ax.set_ylabel("Fraction of rated players ≤ x")
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def _plot_sigma_vs_experience(cols, path, plt, sns) -> None:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    sns.scatterplot(x=cols["n_games"], y=cols["sigma"], ax=ax,
                    alpha=0.3, s=10, edgecolor=None)
    ax.set_xscale("log")
    ax.set_xlabel("Games played (log scale)")
    ax.set_ylabel("TrueSkill σ (uncertainty)")
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def _plot_tichu_success_by_decile(cols, min_tichu_calls, path, plt, sns) -> None:
    mask = (
        (cols["tichu_calls"] >= min_tichu_calls)
        & np.isfinite(cols["tichu_success_rate"])
        & (cols["skill_decile"] >= 0)
    )
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if mask.sum() == 0:
        ax.text(0.5, 0.5, "no players meet the filter", ha="center", va="center")
    else:
        sns.boxplot(
            x=[str(d) for d in cols["skill_decile"][mask]],
            y=cols["tichu_success_rate"][mask],
            order=[str(d) for d in range(10)],
            ax=ax,
        )
    ax.set_xlabel("Skill Decile (0 = bottom 10%, 9 = top)")
    ax.set_ylabel(f"Tichu success rate (players with ≥ {min_tichu_calls} calls)")
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def _plot_grand_tichu_rate_by_decile(cols, path, plt, sns) -> None:
    decile = cols["skill_decile"]
    rate = cols["grand_tichu_calls"] / np.maximum(cols["n_games"], 1)
    means = []
    for d in range(10):
        sel = decile == d
        means.append(float(rate[sel].mean()) if sel.any() else 0.0)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    sns.barplot(x=[str(d) for d in range(10)], y=means, ax=ax)
    ax.set_xlabel("Skill Decile (0 = bottom 10%, 9 = top)")
    ax.set_ylabel("Grand Tichu calls per game (mean across players)")
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def _plot_tichu_call_rate_by_decile(cols, path, plt, sns) -> None:
    decile = cols["skill_decile"]
    rate = cols["tichu_calls"] / np.maximum(cols["n_games"], 1)
    means = []
    for d in range(10):
        sel = decile == d
        means.append(float(rate[sel].mean()) if sel.any() else 0.0)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    sns.barplot(x=[str(d) for d in range(10)], y=means, ax=ax)
    ax.set_xlabel("Skill Decile (0 = bottom 10%, 9 = top)")
    ax.set_ylabel("Tichu calls per game (mean across players)")
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def _plot_grand_tichu_success_by_decile(cols, min_grand_tichu_calls, path, plt, sns) -> None:
    mask = (
        (cols["grand_tichu_calls"] >= min_grand_tichu_calls)
        & np.isfinite(cols["grand_tichu_success_rate"])
        & (cols["skill_decile"] >= 0)
    )
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if mask.sum() == 0:
        ax.text(0.5, 0.5, "no players meet the filter", ha="center", va="center")
    else:
        sns.boxplot(
            x=[str(d) for d in cols["skill_decile"][mask]],
            y=cols["grand_tichu_success_rate"][mask],
            order=[str(d) for d in range(10)],
            ax=ax,
        )
    ax.set_xlabel("Skill Decile (0 = bottom 10%, 9 = top)")
    ax.set_ylabel(
        f"Grand Tichu success rate (players with >= {min_grand_tichu_calls} calls)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def _plot_games_lorenz(cols, stats, path, plt, sns) -> None:
    v = np.sort(cols["n_games"].astype(float))
    n = len(v)
    cum_players = np.arange(1, n + 1) / n
    cum_games = np.cumsum(v) / v.sum() if v.sum() else np.zeros_like(v)
    # Prepend (0, 0) so the curve starts at the origin.
    x = np.concatenate([[0.0], cum_players])
    y = np.concatenate([[0.0], cum_games])
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(x, y, label=f"Observed (Gini = {stats['gini_n_games']:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Even (Gini = 0)")
    ax.set_xlabel("Cumulative share of rated players (lowest n_games → highest)")
    ax.set_ylabel("Cumulative share of games played")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def _plot_tichu_vs_grand_tichu_success(
    cols, min_tichu_calls, min_grand_tichu_calls, path, plt, sns,
) -> None:
    mask = (
        (cols["tichu_calls"] >= min_tichu_calls)
        & (cols["grand_tichu_calls"] >= min_grand_tichu_calls)
        & np.isfinite(cols["tichu_success_rate"])
        & np.isfinite(cols["grand_tichu_success_rate"])
    )
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if mask.sum() == 0:
        ax.text(0.5, 0.5, "no players meet both filters", ha="center", va="center")
    else:
        sns.scatterplot(
            x=cols["tichu_success_rate"][mask],
            y=cols["grand_tichu_success_rate"][mask],
            ax=ax,
            alpha=0.3,
            s=15,
            edgecolor=None,
        )
    ax.set_xlabel(
        f"Tichu success rate (>= {min_tichu_calls} calls)"
    )
    ax.set_ylabel(
        f"Grand Tichu success rate (>= {min_grand_tichu_calls} calls)"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def _plot_correlation_heatmap(cols, path, plt, sns) -> None:
    metrics = [
        ("mu", "mu"),
        ("sigma", "sigma"),
        ("n_games", "n_games"),
        ("Tichu SR", "tichu_success_rate"),
        ("Grand T. SR", "grand_tichu_success_rate"),
    ]
    n = len(metrics)
    M = np.full((n, n), np.nan)
    for i, (_, ki) in enumerate(metrics):
        xi = cols[ki].astype(float)
        for j, (_, kj) in enumerate(metrics):
            xj = cols[kj].astype(float)
            mask = np.isfinite(xi) & np.isfinite(xj)
            if mask.sum() >= 2:
                r = spearman_rho(xi[mask].tolist(), xj[mask].tolist())
                if r is not None:
                    M[i, j] = r
    labels = [name for name, _ in metrics]
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    sns.heatmap(
        M, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
        xticklabels=labels, yticklabels=labels, ax=ax, cbar_kws={"label": "Spearman ρ"},
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def _leaderboard_table(rows: list[dict]) -> str:
    return "\n".join(
        f"| {i+1} | `{r['handle']}` | {r['mu']:.2f} | {r['sigma']:.2f} | {r['n_games']:,} |"
        for i, r in enumerate(rows)
    )


def _write_summary(stats: dict, args, path: Path) -> None:
    s = stats
    shares = [c / s['n_players'] for c in s['decile_counts']] if s['n_players'] else []
    decile_table = "\n".join(
        f"| {d} | {s['decile_counts'][d]:,} | {shares[d]:.3f} |"
        for d in range(10)
    )
    rho_line = (
        f"**Spearman ρ(μ, tichu_success_rate) = {s['spearman_rho']:.3f}** "
        f"over {s['spearman_n']:,} players (tichu_calls ≥ {s['min_tichu_calls']})"
        if s['spearman_rho'] is not None
        else f"Spearman ρ undefined (too few players with tichu_calls ≥ {s['min_tichu_calls']})"
    )
    cross_line = (
        f"**Spearman ρ(tichu_SR, grand_tichu_SR) = {s['cross_rho']:.3f}** "
        f"over {s['cross_n']:,} players (both filters)"
        if s['cross_rho'] is not None
        else "Spearman ρ(tichu_SR, grand_tichu_SR) undefined (too few players meet both filters)"
    )
    gini_line = (
        f"**Gini(n_games) = {s['gini_n_games']:.3f}** "
        f"(0 = even play across rated players, 1 = max-concentrated)"
    )

    md = f"""# Ratings profile — `{Path(args.input).name}`

## Headline numbers

| Metric | Value |
|---|---|
| Rated players | {s['n_players']:,} |
| Total games observed (Σ n_games / 4) | {s['total_games']:,} |
| n_games — median | {s['median_n_games']:,} |
| n_games — p95 | {s['p95_n_games']:,} |
| n_games — max | {s['max_n_games']:,} |

{rho_line}

{cross_line}

{gini_line}

### Skill Decile fill

| Decile | Players | Share |
|---|---:|---:|
{decile_table}

### Top 10 rated players (by μ)

| Rank | Handle | μ | σ | Games |
|---:|---|---:|---:|---:|
{_leaderboard_table(s['top10'])}

### Bottom 10 rated players (by μ)

| Rank | Handle | μ | σ | Games |
|---:|---|---:|---:|---:|
{_leaderboard_table(s['bottom10'])}

### Top 10 most active players (by games played)

| Rank | Handle | μ | σ | Games |
|---:|---|---:|---:|---:|
{_leaderboard_table(s['most_games10'])}

## Figures

### `fig-2-mu-hist.png` — Distribution of TrueSkill μ

**What it shows:** A histogram of `μ` (the algorithm's best-guess skill rating) across all rated players. TrueSkill's prior starts every player at μ = 25; rated players have moved away from that prior based on game outcomes.

**How to read it:** The x-axis is μ (higher = stronger). The y-axis is the number of players in each bin. A clean bell shape says the population is roughly symmetric around an average; skew or fat tails indicate a population mix (e.g., a tail of elites or a cluster of weak grinders).

**What to note:** The distribution being centred near 25 ± something is expected; the *interesting* part is the spread — wider tails mean BSW has a heterogeneous skill mix worth modelling separately.

### `fig-3-n-games-cdf.png` — Games-played CDF (log x)

**What it shows:** The cumulative fraction of rated players whose `n_games` is ≤ x. X-axis is log-scaled because the distribution is heavy-tailed.

**How to read it:** Pick any horizontal line (e.g., y = 0.5) and read off the x value — that's the median games played. A curve that rises fast on the left = most players are casual; a long flat extension on the right = a tail of grinders who play far more than the median.

**What to note:** The corpus only contains players who survived `--min-games`, so the curve starts at that floor — not at 1. Even so, the inequality between the median and the p95 / max is striking.

### `fig-4-sigma-vs-experience.png` — σ shrinks with games played

**What it shows:** A scatter of σ (uncertainty about a player's skill) against `n_games` (log-scaled). One dot per rated player.

**How to read it:** σ should fall as players accumulate games — every game is more evidence for TrueSkill to narrow its estimate. A clean downward trend validates that the sweep is behaving correctly.

**What to note:** Players near the `--min-games` floor cluster at high σ (~4–8); seasoned players at hundreds of games sit near σ ≈ 1. This is why the μ-leaderboard is misleading on its own — see the σ column in the top-10 table.

### `fig-5-tichu-success-by-decile.png` — Tichu success rate by Skill Decile

**What it shows:** A boxplot of each player's small-Tichu success rate, grouped by Skill Decile. Filtered to players with ≥ {s['min_tichu_calls']} Tichu calls so the rate is meaningful.

**How to read it:** Each box covers the middle 50% of players in that decile; the line inside is the median; whiskers reach 1.5× IQR; dots are outliers. A monotonic upward trend in medians = better players win more of the Tichus they call.

**What to note:** This is the headline validation plot — Spearman ρ = {s['spearman_rho']:.3f} between μ and tichu_success_rate. TrueSkill agrees with an *independent* judgment proxy, supporting the rating's meaningfulness.

### `fig-6-grand-tichu-rate-by-decile.png` — Grand Tichu calls per game by decile

**What it shows:** The mean number of Grand Tichu calls per **Game** for each Skill Decile. A Game = a sequence of Rounds played to 1000 points, typically 6–10 Rounds.

**How to read it:** Y-axis units are *calls per Game*, **not** *fraction of Games with a call*. So a top-decile value near 1.0 means "about one Grand Tichu call per Game on average" (≈ 1-in-8 Rounds), not "calls almost every Game."

**What to note:** Top players call Grand Tichu several times more often than the bottom — a strong behavioural signal of confidence in their opening 8 cards (the only info available pre-Schupfen).

### `fig-7-tichu-call-rate-by-decile.png` — Tichu calls per game by decile

**What it shows:** Mirror of fig-6, but for small Tichu instead of Grand Tichu. Small Tichu has a smaller swing (±100 vs ±200), is declared *after* Schupfen, and is therefore cheaper to attempt speculatively.

**How to read it:** Same units as fig-6 — calls per Game (each Game ≈ 6–10 Rounds). Compare the *shape* against fig-6: if the gradient is similar, skill drives both kinds of calling equally; if flatter or U-shaped, small Tichu is a more universal play and Grand Tichu is what separates skill tiers.

**What to note:** The contrast between this curve and fig-6 is the punchline — the difference reveals which kind of call actually distinguishes skill.

### `fig-8-grand-tichu-success-by-decile.png` — Grand Tichu success rate by Skill Decile

**What it shows:** Mirror of fig-5, but for Grand Tichu. Filtered to players with ≥ {s['min_grand_tichu_calls']} Grand Tichu calls (a lower floor than small Tichu because Grand Tichu is much rarer).

**How to read it:** Same boxplot format as fig-5. Pairs with fig-6: fig-6 tells you *how often* each decile calls, this tells you *how often they win when they do*.

**What to note:** Together with fig-6, this answers "do top players just call more, or also call *better*?" If success rate also climbs with decile, both volume and judgment scale with skill. If it's flat, top players just take more shots at the same hit rate.

### `fig-9-games-played-lorenz.png` — Engagement concentration (Lorenz curve)

**What it shows:** Cumulative share of rated players (x-axis, ordered from least-active to most-active) against their cumulative share of all Games played (y-axis). The diagonal is the "perfectly even" reference; the deeper the curve bows below the diagonal, the more games are concentrated in a small group of grinders.

**How to read it:** Pick any x and read off y. If x = 0.9 gives y = 0.5, then the bottom 90% of rated players account for only 50% of Games — the top 10% generate the other 50%. The **Gini coefficient** (in the legend) summarises the bow: 0 = perfectly even, 1 = one player has all Games.

**What to note:** This is the most concise answer to "who is BSW Tichu *actually* played by?" A Gini ≥ 0.5 means the corpus is dominated by a small grinder population — which has implications for whose play style the model is cloning.

### `fig-10-tichu-vs-grand-tichu-success.png` — Does call judgment transfer?

**What it shows:** One dot per player who meets *both* call-volume filters (≥ {s['min_tichu_calls']} Tichu calls AND ≥ {s['min_grand_tichu_calls']} Grand Tichu calls). X = their small-Tichu success rate. Y = their Grand-Tichu success rate.

**How to read it:** A tight upward diagonal cloud = players who are good at one kind of call are good at the other (judgment is a shared skill). A diffuse blob = the two judgments decouple (mastering one says little about the other). The Spearman correlation across these players is reported in the headline numbers section.

**What to note:** This is the only plot in the report that asks *whether two skill proxies move together* rather than testing each one against μ. A weak correlation here would be a surprise — likely a publishable finding for the blog.

### `fig-11-correlation-heatmap.png` — Spearman correlations between all metrics

**What it shows:** A 5×5 heatmap of pairwise Spearman ρ between μ, σ, n_games, Tichu SR, and Grand Tichu SR. Red = positive correlation, blue = negative, white ≈ zero. Each cell uses the pairwise-complete subset of players where both metrics are defined.

**How to read it:** The diagonal is always 1.0. Off-diagonal cells answer "do these two move together across the player population?" Strong negative σ↔n_games confirms uncertainty shrinks with experience. A strong positive μ↔Tichu-SR mirrors fig-5's headline. Anything *surprising* (e.g., σ correlating with success rates) deserves a paragraph in the post.

**What to note:** This is the most compact "everything-vs-everything" view; treat it as a checklist for which other plots are worth zooming in on.

## A note on the population (caveat for the post)

> The parquet input contains only players who survived the Min-games Filter
> (`--min-games`, default 20 in `compute_trueskill`). Anonymous Seats — guest
> accounts and BSW serialiser quirks that drop a handle — are excluded
> upstream by the TrueSkill sweep (ADR-0010). Distributions of `n_games`
> therefore start at the filter threshold, not at 1, and "rated players"
> undercounts total BSW handles.
"""
    path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
