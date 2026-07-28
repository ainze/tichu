#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
# ]
# ///
"""Plot Full-Stack Co-Training curves from a co-train `ppo_log.csv` (ADR-0034).

Distinct from `plot_ppo_csv.py` (single-policy PPO Refine, ADR-0029): co-train
sharpens FOUR nets together, so `train_cotrain` writes one row per iteration with
per-net columns:

    iter, wall_s, loss, value_loss,
    {net}_policy_loss, {net}_kl, {net}_entropy, {net}_kl_coef   for net in
    {play, schupfen, tichu, grand}  (+ wish when cotrain_wish is on)

The nets are auto-detected from the `*_policy_loss` columns, so a wish-enabled run
plots its extra net with no flag. Six panels in a 3x2 grid, one line per net:

  1. Critic value_loss (log) + total loss — the warm-up / under-fit signal. With a
     deeper/wider Perfect-Info Critic (ADR-0038), this is THE panel: value_loss
     should fall and flatten low. A high plateau ⇒ the critic still under-fits the
     lever decisions (go deeper before wider).
  2. Per-net policy_loss — the clipped surrogate per net.
  3. Per-net KL-to-BC (lever #1) — does each net move off BC? Watch schupfen: its
     target ANNEALS 0.008→0.025 (cotrain_v6), so its KL should rise over iters
     [750, 2750]; pinned-flat-low ⇒ leashed, the lever can't move. --kl-target
     draws a single reference line (the play/calls target); schupfen differs.
  4. Per-net beta_KL (kl_coef, log) — the adaptive controller's response. Pinned
     high ⇒ the anchor can't hold; floored ⇒ the policy isn't moving (loosen).
  5. Per-net entropy — should sit modest/stable; a collapse ⇒ no exploration.
  6. Greedy-gate tournament margin (from the sibling `promotion_gate.csv`): per-
     opponent mean score margin per gate window with its bootstrap 95% CI band,
     promotions marked. Empty panel for non-gated runs.
  7. Gate margin DECOMPOSED into card play vs call bonus (from the sibling
     `promotion_gate_components.csv`). Panel 6 shows only the total, and a total
     near zero can be two large opposite halves: the iter27008 champion read +0.14
     against the served export while being +5.5 on card play and −5.4 on calls — a
     trade invisible for 27k iterations that surfaced only at ship time. Watch for
     the two lines DIVERGING; that is co-training reallocating value between them
     rather than adding any. Empty for runs predating the split.

Behavioral dials + strength vs the SHIPPED master stay OUT OF LOOP: `check_cotrain`
and `eval_matrix --mode behavioral`. Panel 6 shows the in-run gate opponents only.

Usage:
    ./plot_cotrain_csv.py path/to/run-dir                 # auto-finds ppo_log.csv
    ./plot_cotrain_csv.py path/to/run-dir --watch 15      # live-follow
    ./plot_cotrain_csv.py path/to/run-dir --out fig.png --window 20
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Stable per-net colour so a net keeps its colour across panels and redraws.
_NET_COLORS = {"play": "C0", "schupfen": "C1", "tichu": "C2", "grand": "C3", "wish": "C4"}
_SUFFIX = "_policy_loss"


def _resolve_csv(p: Path) -> Path:
    if p.is_file():
        return p
    if p.is_dir():
        cand = p / "ppo_log.csv"
        if cand.is_file():
            return cand
        sys.exit(f"no ppo_log.csv in {p}")
    sys.exit(f"not found: {p}")


def _roll(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=1).mean()


def _nets(df: pd.DataFrame) -> list[str]:
    """Net names in canonical order, derived from the `*_policy_loss` columns."""
    found = {c[: -len(_SUFFIX)] for c in df.columns if c.endswith(_SUFFIX)}
    ordered = [n for n in ("play", "schupfen", "tichu", "grand", "wish") if n in found]
    return ordered + sorted(found - set(ordered))


def _promotion_iters(csv_path: Path) -> list[int]:
    """Iters at which a new champion was promoted, from a sibling `promotion_gate.csv`
    (rounds-as-gate runs). Empty for non-gated runs. The CSV has one row per opponent
    per verdict, so dedupe the iters where `promoted == 1`."""
    gate = csv_path.parent / "promotion_gate.csv"
    if not gate.is_file():
        return []
    try:
        g = pd.read_csv(gate)
    except (pd.errors.EmptyDataError, OSError):
        return []
    if not {"iter", "promoted"} <= set(g.columns):
        return []
    return sorted({int(i) for i in g.loc[g["promoted"] == 1, "iter"]})


def _gate_frame(csv_path: Path) -> pd.DataFrame | None:
    """The sibling `promotion_gate.csv` as a DataFrame, or None when absent/empty.
    One row per opponent per gate window: iter, opponent, n, mean, ci_lo, ci_hi,
    promoted."""
    gate = csv_path.parent / "promotion_gate.csv"
    if not gate.is_file():
        return None
    try:
        g = pd.read_csv(gate)
    except (pd.errors.EmptyDataError, OSError):
        return None
    need = {"iter", "opponent", "mean", "ci_lo", "ci_hi", "promoted"}
    if g.empty or not need <= set(g.columns):
        return None
    return g


# Stable per-opponent colour in the gate panel (extras fall back to the cycle).
_OPP_COLORS = {"bc": "C0", "champion": "C1", "cpfix3328": "C3"}


def _plot_gate(ax, csv_path: Path) -> None:
    """Panel 6: greedy-gate tournament margin per opponent, CI-banded, promotions
    marked. The gate records seat-swapped greedy mini-tournament margins every
    `greedy_every` iters — the deployed-strength trajectory the negative-control /
    plateau reads come from."""
    g = _gate_frame(csv_path)
    ax.set_title("Greedy-gate tournament margin (learner − opponent, per window)")
    ax.set_ylabel("score margin / round")
    ax.grid(True, alpha=0.3, which="both")
    if g is None:
        ax.text(0.5, 0.5, "no promotion_gate.csv (non-gated run)",
                ha="center", va="center", transform=ax.transAxes, alpha=0.6)
        return
    ax.axhline(0.0, color="k", linestyle="--", alpha=0.4, linewidth=1.0)
    for i, (opp, rows) in enumerate(g.groupby("opponent", sort=False)):
        rows = rows.sort_values("iter")
        color = _OPP_COLORS.get(str(opp), f"C{(4 + i) % 10}")
        ax.plot(rows["iter"], rows["mean"], color=color, linewidth=1.5, marker="o",
                markersize=3, label=f"vs {opp} (last {rows['mean'].iloc[-1]:+.2f})")
        ax.fill_between(rows["iter"], rows["ci_lo"], rows["ci_hi"],
                        color=color, alpha=0.15, linewidth=0)
    promo = g.loc[g["promoted"] == 1]
    if not promo.empty:
        for k, it in enumerate(sorted({int(i) for i in promo["iter"]})):
            ax.axvline(it, color="C3", linestyle="--", alpha=0.55, linewidth=1.0,
                       label="champion promoted" if k == 0 else None)
    ax.legend(loc="best", fontsize=8)


def _components_frame(csv_path: Path) -> pd.DataFrame | None:
    """The sibling `promotion_gate_components.csv`, or None when absent/empty.
    One row per opponent per component per gate window: iter, opponent, component,
    n, mean, se, ci_lo, ci_hi, promoted. Written separately from
    `promotion_gate.csv` because that file's 7-column header is already on disk
    mid-run for every live run."""
    comp = csv_path.parent / "promotion_gate_components.csv"
    if not comp.is_file():
        return None
    try:
        c = pd.read_csv(comp)
    except (pd.errors.EmptyDataError, OSError):
        return None
    need = {"iter", "opponent", "component", "mean", "ci_lo", "ci_hi"}
    if c.empty or not need <= set(c.columns):
        return None
    return c


# Colour carries the COMPONENT (the thing being compared), linestyle carries the
# opponent — so a card-play/call-bonus divergence reads at a glance even with three
# opponents on the panel.
_COMPONENT_COLORS = {"card_play": "C2", "call_bonus": "C3"}
_OPP_LINESTYLES = {"champion": "-", "bc": "--", "cpfix3328": ":"}


def _plot_components(ax, csv_path: Path) -> None:
    """Panel 7: the gate margin split into card play and call bonus.

    The total (panel 6) hides a reallocation between the two. Diverging lines mean
    co-training is trading one for the other rather than adding value — measured
    once at iter27008 as +5.5 card play against −5.4 call bonus for a net +0.14,
    i.e. ~87% of the reallocation wasted."""
    c = _components_frame(csv_path)
    ax.set_title("Gate margin decomposed — card play vs call bonus (diverging = a trade, not a gain)")
    ax.set_ylabel("score margin / round")
    ax.grid(True, alpha=0.3, which="both")
    if c is None:
        ax.text(0.5, 0.5, "no promotion_gate_components.csv\n"
                          "(run predates the split, or non-gated)",
                ha="center", va="center", transform=ax.transAxes, alpha=0.6)
        return
    ax.axhline(0.0, color="k", linestyle="--", alpha=0.4, linewidth=1.0)
    for (opp, comp), rows in c.groupby(["opponent", "component"], sort=False):
        rows = rows.sort_values("iter")
        color = _COMPONENT_COLORS.get(str(comp), "C4")
        style = _OPP_LINESTYLES.get(str(opp), "-.")
        ax.plot(rows["iter"], rows["mean"], color=color, linestyle=style,
                linewidth=1.4, marker="o", markersize=2.5,
                label=f"{comp} vs {opp} ({rows['mean'].iloc[-1]:+.2f})")
        ax.fill_between(rows["iter"], rows["ci_lo"], rows["ci_hi"],
                        color=color, alpha=0.10, linewidth=0)
    if "promoted" in c.columns:
        promo = sorted({int(i) for i in c.loc[c["promoted"] == 1, "iter"]})
        for k, it in enumerate(promo):
            ax.axvline(it, color="C3", linestyle="--", alpha=0.45, linewidth=1.0,
                       label="champion promoted" if k == 0 else None)
    ax.legend(loc="best", fontsize=7, ncol=2)


def _plot_critic(ax, df: pd.DataFrame, window: int, promo_iters=()) -> None:
    x, v = df["iter"], df["value_loss"]
    ax.scatter(x, v, s=8, alpha=0.2, color="C0", label="value_loss (per iter)")
    ax.plot(x, _roll(v, window), color="C0", linewidth=1.6, label=f"value_loss (rolling {window})")
    if (v > 0).all():
        ax.set_yscale("log")
    ax.set_ylabel("value_loss (log)", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    promo = f"  —  {len(promo_iters)} champion promotions" if promo_iters else ""
    ax.set_title(f"Critic value loss (deeper-critic / warm-up signal)  —  last: {v.iloc[-1]:,.2f}{promo}")
    ax.grid(True, alpha=0.3, which="both")

    # Champion promotions (rounds-as-gate): a dashed vertical marker each time a new
    # champion was selected, so the value-loss / total-loss panel reads against the
    # ratchet. One legend entry for the set.
    for k, it in enumerate(promo_iters):
        ax.axvline(it, color="C3", linestyle="--", alpha=0.55, linewidth=1.0,
                   label="champion promoted" if k == 0 else None)

    ax2 = ax.twinx()
    loss = df["loss"]
    ax2.plot(x, _roll(loss, window), color="C7", linewidth=1.2, alpha=0.8, label="total loss")
    ax2.set_ylabel("total loss", color="C7")
    ax2.tick_params(axis="y", labelcolor="C7")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)


def _plot_per_net(ax, df: pd.DataFrame, window: int, suffix: str, *, ylabel: str,
                  title: str, logy: bool = False, target: float | None = None) -> None:
    x = df["iter"]
    cols = []
    for net in _nets(df):
        col = f"{net}{suffix}"
        if col not in df.columns:
            continue
        ax.plot(x, _roll(df[col], window), color=_NET_COLORS.get(net, "C7"),
                linewidth=1.5, label=f"{net} (last {df[col].iloc[-1]:.4g})")
        cols.append(col)
    if target is not None:
        ax.axhline(target, color="k", linestyle="--", alpha=0.4, label=f"ref target {target:g}")
    if logy and cols and (df[cols] > 0).all().all():
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8, ncol=2)


def _suptitle(csv_path: Path, df: pd.DataFrame, positions_per_iter: int) -> str:
    wall = df["wall_s"]
    elapsed_h = wall.sum() / 3600.0
    valid = wall > 0
    rph = (positions_per_iter * 3600.0 / wall[valid]).tail(max(1, len(df) // 10)).mean() if valid.any() else float("nan")
    return (f"{csv_path}  [Full-Stack Co-Train]  —  {len(df)} iters, "
            f"{elapsed_h:.2f} h, ~{rph:,.0f} rounds/h (M={positions_per_iter})")


def _draw(fig, axes, csv_path: Path, window: int, kl_target: float, positions_per_iter: int) -> bool:
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return False
    if df.empty:
        return False

    for ax in axes:
        for t in list(getattr(ax, "_twins", [])):
            t.remove()
        ax._twins = []
        ax.clear()

    orig_twinx = {ax: ax.twinx for ax in axes}
    for ax in axes:
        def _tracked(_ax=ax):
            t = orig_twinx[_ax]()
            _ax._twins.append(t)
            return t
        ax.twinx = _tracked

    _plot_critic(axes[0], df, window, _promotion_iters(csv_path))
    _plot_per_net(axes[1], df, window, "_policy_loss", ylabel="policy_loss",
                  title="Per-net clipped policy loss")
    _plot_per_net(axes[2], df, window, "_kl", ylabel="kl_to_bc",
                  title="Per-net KL-to-BC (lever #1; schupfen target anneals 0.008→0.025)",
                  target=kl_target)
    _plot_per_net(axes[3], df, window, "_kl_coef", ylabel="beta_KL (log)",
                  title="Per-net beta_KL (kl_coef) — pinned-high=anchor can't hold, floored=not moving",
                  logy=True)
    _plot_per_net(axes[4], df, window, "_entropy", ylabel="entropy",
                  title="Per-net entropy")
    _plot_gate(axes[5], csv_path)
    _plot_components(axes[6], csv_path)
    # 4x2 grid for 7 panels; the spare slot stays blank rather than stretching the
    # layout. Re-hidden every redraw because `ax.clear()` restores visibility.
    axes[7].set_visible(False)

    for ax in axes:
        ax.twinx = orig_twinx[ax]
        ax.set_xlabel("iteration")

    fig.suptitle(_suptitle(csv_path, df, positions_per_iter), fontsize=10)
    fig.tight_layout()
    return True


def _plot(csv_path: Path, window: int, out: Path | None, kl_target: float,
          positions_per_iter: int, watch: float | None) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(18, 16))
    axes = axes.flatten()
    for ax in axes:
        ax._twins = []

    if watch is not None:
        plt.ion()
        plt.show(block=False)
        print(f"watching {csv_path} (redraw every {watch:g}s; Ctrl-C to stop)")
        try:
            while True:
                if not _draw(fig, axes, csv_path, window, kl_target, positions_per_iter):
                    print("  (no data rows yet)")
                fig.canvas.draw_idle()
                plt.pause(watch)
                if not plt.fignum_exists(fig.number):
                    break
        except KeyboardInterrupt:
            print("\nstopped watching")
        return

    if not _draw(fig, axes, csv_path, window, kl_target, positions_per_iter):
        sys.exit(f"{csv_path} has no data rows yet")

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"wrote {out}")
    else:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="co-train ppo_log.csv file or run directory containing it")
    ap.add_argument("--window", type=int, default=10, help="rolling-mean window in iterations (default 10)")
    ap.add_argument("--out", type=Path, default=None, help="save PNG here instead of opening a window")
    ap.add_argument("--kl-target", type=float, default=0.02,
                    help="reference KL target line (default 0.02 = play/calls target; schupfen anneals)")
    ap.add_argument("--positions-per-iter", type=int, default=512,
                    help="M games per iteration, for the rounds/hour estimate (default 512)")
    ap.add_argument("--watch", type=float, default=None, metavar="SECONDS",
                    help="live-follow: redraw every SECONDS (the CSV is flushed each iteration)")
    args = ap.parse_args()
    _plot(_resolve_csv(args.path), args.window, args.out, args.kl_target,
          args.positions_per_iter, args.watch)


if __name__ == "__main__":
    main()
