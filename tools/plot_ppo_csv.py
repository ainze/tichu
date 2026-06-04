#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
# ]
# ///
"""Plot PPO Refine training curves from a `ppo_log.csv` (ADR-0029).

`train_ppo` writes one row per iteration:
    iter, wall_s, loss, policy_loss, value_loss, entropy, kl_to_bc, kl_coef
This is the in-loop dashboard for *following a live run* — the four panels map
onto the decisions locked in the run-design grilling:

  1. Critic value_loss (log y) — the Q5 "do we need a critic warm-up?" signal.
     A steep early decline that flattens low ⇒ the separate critic is catching
     up on its own; a high plateau / divergence ⇒ add a warm-up.
  2. Policy loss + entropy (twin axis) — entropy is lever #2 (ent_coef). It
     should rise off BC's low base (exploring under-tried bombs) and stabilise;
     a collapse ⇒ no exploration, a runaway ⇒ heading for bomb-spam.
  3. KL-to-BC + beta_KL (kl_coef, log y) — lever #1. kl_to_bc should sit inside
     the adaptive controller's band [target/1.5, target*1.5] (shaded); beta_KL
     is what the controller moves to keep it there. Drift above the band with
     beta_KL pinned high ⇒ the anchor can't hold; stuck below ⇒ the policy isn't
     moving (loosen the leash).
  4. Throughput — wall_s/iter and the implied rounds/hour (needs M, the
     positions_per_iter; pass --positions-per-iter, default 256). This is the
     number that turns the iteration budget into the wall-clock kill-budget.

NOTE: the behavioral dials (caller_bomb_passivity_rate etc.) are NOT in this CSV
— they are computed out-of-loop by exporting a snapshot and running
`eval_matrix --mode behavioral`. This tool follows training health; the kill
rule's headline metric is tracked separately.

Usage:
    ./plot_ppo_csv.py path/to/ppo_log.csv
    ./plot_ppo_csv.py path/to/run-dir                 # auto-finds ppo_log.csv
    ./plot_ppo_csv.py path/to/run-dir --out fig.png
    ./plot_ppo_csv.py path/to/run-dir --watch 10      # live-follow, redraw every 10s
    ./plot_ppo_csv.py path/to/run-dir --window 20 --kl-target 0.02 --positions-per-iter 256
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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


def _plot_value_loss(ax, df: pd.DataFrame, window: int) -> None:
    x, v = df["iter"], df["value_loss"]
    ax.scatter(x, v, s=8, alpha=0.2, color="C0", label="value_loss (per iter)")
    ax.plot(x, _roll(v, window), color="C0", linewidth=1.6, label=f"rolling {window}")
    if (v > 0).all():
        ax.set_yscale("log")
    ax.set_ylabel("value_loss (log)", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    last = v.iloc[-1]
    ax.set_title(f"Critic value loss (Q5 warm-up signal)  —  last: {last:,.1f}, "
                 f"{len(df)} iters")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=8)


def _plot_policy_entropy(ax, df: pd.DataFrame, window: int) -> None:
    x = df["iter"]
    pl = df["policy_loss"]
    ax.scatter(x, pl, s=8, alpha=0.2, color="C0", label="policy_loss (per iter)")
    ax.plot(x, _roll(pl, window), color="C0", linewidth=1.6, label=f"policy_loss (rolling {window})")
    ax.axhline(0.0, color="C0", linestyle=":", alpha=0.4)
    ax.set_ylabel("policy_loss", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.set_title("Clipped policy loss & entropy (lever #2: ent_coef)")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ent = df["entropy"]
    ax2.plot(x, _roll(ent, window), color="C3", linewidth=1.4, alpha=0.9,
             label=f"entropy (rolling {window})")
    ax2.set_ylabel("entropy", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)


def _plot_kl(ax, df: pd.DataFrame, window: int, kl_target: float) -> None:
    x = df["iter"]
    kl = df["kl_to_bc"]
    lo, hi = kl_target / 1.5, kl_target * 1.5
    ax.axhspan(lo, hi, color="C2", alpha=0.12, label=f"controller band [{lo:.4f}, {hi:.4f}]")
    ax.axhline(kl_target, color="C2", linestyle="--", alpha=0.6, label=f"target {kl_target:g}")
    ax.scatter(x, kl, s=8, alpha=0.25, color="C0", label="kl_to_bc (per iter)")
    ax.plot(x, _roll(kl, window), color="C0", linewidth=1.6, label=f"kl_to_bc (rolling {window})")
    ax.set_ylabel("kl_to_bc", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.set_title("KL-to-BC anchor & beta_KL (lever #1: kl.target)")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    coef = df["kl_coef"]
    ax2.plot(x, coef, color="C4", linewidth=1.2, alpha=0.85, drawstyle="steps-post",
             label="beta_KL (kl_coef)")
    if (coef > 0).all():
        ax2.set_yscale("log")
    ax2.set_ylabel("beta_KL (log)", color="C4")
    ax2.tick_params(axis="y", labelcolor="C4")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)


def _plot_throughput(ax, df: pd.DataFrame, window: int, positions_per_iter: int) -> None:
    x = df["iter"]
    wall = df["wall_s"]
    ax.scatter(x, wall, s=8, alpha=0.2, color="C4", label="wall_s / iter")
    ax.plot(x, _roll(wall, window), color="C4", linewidth=1.6, label=f"rolling {window}")
    ax.set_ylabel("seconds / iter", color="C4")
    ax.tick_params(axis="y", labelcolor="C4")

    # rounds/hour = M games per iter / seconds-per-iter * 3600.
    valid = wall > 0
    rph = pd.Series(index=x.index, dtype=float)
    rph[valid] = positions_per_iter * 3600.0 / wall[valid]
    total_h = wall.sum() / 3600.0
    mean_rph = _roll(rph, window).iloc[-1] if valid.any() else float("nan")
    ax.set_title(f"Throughput  —  ~{mean_rph:,.0f} rounds/h (M={positions_per_iter}), "
                 f"elapsed {total_h:.2f} h over {len(df)} iters")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(x, _roll(rph, window), color="C2", linewidth=1.4, alpha=0.9,
             label=f"rounds/hour (rolling {window})")
    ax2.set_ylabel("rounds / hour", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)


def _draw(fig, axes, csv_path: Path, window: int, kl_target: float, positions_per_iter: int) -> bool:
    """Read the CSV and (re)draw all panels onto the given axes. Returns True if
    there was data to plot, False otherwise (e.g. the run hasn't logged a row yet)."""
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return False
    if df.empty:
        return False

    for ax in axes:
        ax.clear()
        for extra in list(getattr(ax, "_twins", [])):
            extra.remove()
        ax._twins = []

    # twinx() makes a new axes each draw; track them so a watch-redraw can clear.
    orig_twinx = {}
    for ax in axes:
        orig_twinx[ax] = ax.twinx
        def _tracked_twinx(_ax=ax):
            t = orig_twinx[_ax]()
            _ax._twins.append(t)
            return t
        ax.twinx = _tracked_twinx

    _plot_value_loss(axes[0], df, window)
    _plot_policy_entropy(axes[1], df, window)
    _plot_kl(axes[2], df, window, kl_target)
    _plot_throughput(axes[3], df, window, positions_per_iter)

    for ax in axes:
        ax.twinx = orig_twinx[ax]
        ax.set_xlabel("iteration")

    fig.suptitle(f"{csv_path}  [PPO Refine]", fontsize=10)
    fig.tight_layout()
    return True


def _plot(csv_path: Path, window: int, out: Path | None, kl_target: float,
          positions_per_iter: int, watch: float | None) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 13))
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
                    break  # window closed
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
        _center_window_on_screen(fig)
        plt.show()


def _center_window_on_screen(fig) -> None:
    """Centre the matplotlib window on screen. Backend-specific; swallow failures
    so the plot still shows on an unrecognised backend (mirrors plot_step_csv)."""
    try:
        manager = fig.canvas.manager
    except AttributeError:
        return
    try:
        fig.canvas.draw()
    except Exception:  # noqa: BLE001
        pass

    window = getattr(manager, "window", None)
    if window is None:
        return

    if hasattr(window, "winfo_screenwidth"):
        try:
            window.update_idletasks()
            w, h = window.winfo_width(), window.winfo_height()
            sw, sh = window.winfo_screenwidth(), window.winfo_screenheight()
            window.geometry(f"+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 2)}")
            return
        except Exception:  # noqa: BLE001
            pass

    if hasattr(window, "screen") and hasattr(window, "move"):
        try:
            geom = window.frameGeometry()
            sg = window.screen().availableGeometry()
            window.move(sg.x() + (sg.width() - geom.width()) // 2,
                        sg.y() + (sg.height() - geom.height()) // 2)
            return
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="ppo_log.csv file or run directory containing it")
    ap.add_argument("--window", type=int, default=10, help="rolling-mean window in iterations (default 10)")
    ap.add_argument("--out", type=Path, default=None, help="save PNG here instead of opening a window")
    ap.add_argument("--kl-target", type=float, default=0.02,
                    help="kl.target the adaptive controller steers toward (default 0.02); shades the band")
    ap.add_argument("--positions-per-iter", type=int, default=256,
                    help="M games per iteration, for the rounds/hour estimate (default 256)")
    ap.add_argument("--watch", type=float, default=None, metavar="SECONDS",
                    help="live-follow: redraw every SECONDS (the CSV is flushed each iteration)")
    args = ap.parse_args()
    _plot(_resolve_csv(args.path), args.window, args.out, args.kl_target,
          args.positions_per_iter, args.watch)


if __name__ == "__main__":
    main()
