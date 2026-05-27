#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
# ]
# ///
"""Plot per-head loss + accuracy curves from a BC/AWR step.csv log.

Each row in step.csv corresponds to one head firing one batch. The three
heads (play / wish / dragon_assignment) have very different baseline
losses, so a raw plot of `loss_total` is meaningless — this script
splits by head and overlays a rolling-mean trend on top of the noisy
per-batch points.

AWR runs (detected by the presence of an `awr_weight_mean` column) get
two extra panels:

  * AWR weight-mean over batches with loss_total on the twin axis —
    diagnoses whether the reweighting is doing anything (mean ≪ 1 with
    high variance ⇒ aggressive sharpening; mean ≈ 1 ⇒ uniform, AWR is
    effectively plain BC).
  * Per-epoch summary from epoch.csv (sibling file): the
    `win_rate_proxy` (the only out-of-sample signal in the loop) and
    the epoch-level `avg_weight`. This is the panel to optimise against.

Usage:
    ./plot_step_csv.py path/to/step.csv
    ./plot_step_csv.py path/to/run-dir            # auto-finds step.csv (+ epoch.csv)
    ./plot_step_csv.py path/to/step.csv --out fig.png
    ./plot_step_csv.py path/to/step.csv --window 200
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HEADS = ["play", "wish", "dragon_assignment"]


def _resolve_csv(p: Path) -> Path:
    if p.is_file():
        return p
    if p.is_dir():
        cand = p / "step.csv"
        if cand.is_file():
            return cand
        sys.exit(f"no step.csv in {p}")
    sys.exit(f"not found: {p}")


def _load_epoch_csv(step_csv_path: Path) -> pd.DataFrame | None:
    """AWR runs write a sibling epoch.csv with the per-epoch summary.
    Return it if present and non-empty, else None."""
    epoch_csv = step_csv_path.parent / "epoch.csv"
    if not epoch_csv.is_file():
        return None
    try:
        edf = pd.read_csv(epoch_csv)
    except pd.errors.EmptyDataError:
        return None
    return edf if not edf.empty else None


def _plot_head(ax, sub: pd.DataFrame, head: str, window: int) -> None:
    x = sub["step"]
    loss_col = f"loss_{head}"
    acc_col = f"acc_{head}"
    ax.scatter(x, sub[loss_col], s=4, alpha=0.15, color="C0", label="loss (per batch)")
    ax.plot(
        x, sub[loss_col].rolling(window, min_periods=1).mean(),
        color="C0", linewidth=1.6, label=f"loss (rolling {window})",
    )
    ax.set_ylabel("loss", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.set_title(f"{head}  —  {len(sub):,} batches")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(
        x, sub[acc_col].rolling(window, min_periods=1).mean(),
        color="C3", linewidth=1.2, alpha=0.85,
        label=f"acc (rolling {window})",
    )
    ax2.set_ylabel("accuracy", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.set_ylim(0, 1)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)


def _plot_awr_weights(ax, df: pd.DataFrame, window: int) -> None:
    """AWR weight diagnostics: per-batch awr_weight_mean + loss_total.

    `awr_weight_mean ≈ 1` across the run ⇒ uniform weights ⇒ AWR is
    doing nothing (check beta, baseline quality, outcome variance).
    `awr_weight_mean ≪ 1` with high variance ⇒ aggressive sharpening,
    most batches dominated by a few high-advantage samples. Loss_total
    here is the actual minimised quantity, useful alongside the weight
    signal because the weights and the loss interact.
    """
    x = df["step"]
    w = df["awr_weight_mean"]

    ax.scatter(x, w, s=4, alpha=0.15, color="C2", label="awr_weight_mean (per batch)")
    ax.plot(
        x, w.rolling(window, min_periods=1).mean(),
        color="C2", linewidth=1.6, label=f"awr_weight_mean (rolling {window})",
    )
    ax.axhline(1.0, color="C2", linestyle=":", alpha=0.4, label="weight = 1 (uniform)")
    ax.set_ylabel("awr_weight_mean", color="C2")
    ax.tick_params(axis="y", labelcolor="C2")
    ax.set_title(f"AWR weights  —  mean over run: {w.mean():.3f}")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(
        x, df["loss_total"].rolling(window, min_periods=1).mean(),
        color="C4", linewidth=1.2, alpha=0.85,
        label=f"loss_total (rolling {window})",
    )
    ax2.set_ylabel("loss_total", color="C4")
    ax2.tick_params(axis="y", labelcolor="C4")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)


def _plot_epoch_summary(ax, edf: pd.DataFrame) -> None:
    """Per-epoch AWR summary — the panel to actually optimise against.

    `win_rate_proxy` is the held-out play-head top-1 accuracy and is the
    only out-of-sample signal in the whole pipeline. `avg_weight` shows
    the cumulative AWR sharpening per epoch.

    When the streaming path is run with eval_every_chunks>0, epoch.csv
    contains BOTH mid-epoch snapshots (chunks_done populated) and
    end-of-epoch rows (chunks_done empty). Render them as one continuous
    line indexed by snapshot order, with end-of-epoch boundaries marked
    by larger filled markers + vertical dashed guides.
    """
    edf = edf.reset_index(drop=True)
    x = edf.index
    wrp = edf.get("win_rate_proxy")
    aw = edf.get("avg_weight")
    chunks_done = edf.get("chunks_done")

    # Identify which rows are end-of-epoch markers.
    if chunks_done is not None:
        end_mask = chunks_done.astype(str).str.strip().isin(["", "nan"])
    else:
        end_mask = pd.Series([True] * len(edf))
    mid_mask = ~end_mask

    plotted = False
    if wrp is not None:
        wrp_clean = pd.to_numeric(wrp, errors="coerce")
        if wrp_clean.notna().any():
            # Single continuous line through all snapshots, distinct
            # markers for mid vs end.
            ax.plot(x, wrp_clean, "-", color="C3", linewidth=1.2, alpha=0.6)
            if mid_mask.any():
                ax.plot(
                    x[mid_mask], wrp_clean[mid_mask],
                    "o", color="C3", markersize=4, alpha=0.7,
                    label="win_rate_proxy (mid-epoch)",
                )
            if end_mask.any():
                ax.plot(
                    x[end_mask], wrp_clean[end_mask],
                    "s", color="C3", markersize=10, markeredgecolor="black",
                    markeredgewidth=0.8,
                    label="win_rate_proxy (end-of-epoch)",
                )
            ax.set_ylim(0, 1)
            plotted = True
            # Vertical guide at each end-of-epoch boundary.
            for xi in x[end_mask]:
                ax.axvline(xi, color="grey", linestyle=":", alpha=0.3)
    ax.set_ylabel("win_rate_proxy", color="C3")
    ax.tick_params(axis="y", labelcolor="C3")
    ax.set_xlabel("snapshot (mid-epoch evals + end-of-epoch markers)")
    ax.set_title("AWR per-epoch summary")
    ax.grid(True, alpha=0.3)

    if aw is not None:
        aw_clean = pd.to_numeric(aw, errors="coerce")
        if aw_clean.notna().any():
            ax2 = ax.twinx()
            ax2.plot(x, aw_clean, "s--", color="C2", linewidth=1.2, alpha=0.6, markersize=3, label="avg_weight")
            ax2.set_ylabel("avg_weight", color="C2")
            ax2.tick_params(axis="y", labelcolor="C2")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
            plotted = True
        else:
            ax.legend(loc="best", fontsize=8)
    else:
        ax.legend(loc="best", fontsize=8)

    if not plotted:
        ax.text(0.5, 0.5, "epoch.csv has no plottable columns",
                ha="center", va="center", transform=ax.transAxes)


def _plot(csv_path: Path, window: int, out: Path | None) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        sys.exit(f"{csv_path} has no data rows yet")

    is_awr = "awr_weight_mean" in df.columns
    epoch_df = _load_epoch_csv(csv_path) if is_awr else None

    panels: list[str] = list(HEADS)
    if is_awr:
        panels.append("__awr__")
    if epoch_df is not None:
        panels.append("__epoch__")

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.2 * len(panels)))
    if len(panels) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        if panel == "__awr__":
            _plot_awr_weights(ax, df, window)
        elif panel == "__epoch__":
            _plot_epoch_summary(ax, epoch_df)
        else:
            head = panel
            loss_col = f"loss_{head}"
            sub = df[df[loss_col] > 0]
            if sub.empty:
                ax.set_title(f"{head} — no batches fired")
                ax.set_ylabel("loss")
                continue
            _plot_head(ax, sub, head, window)

    # All step-indexed panels (heads + AWR weights) share the x-axis label.
    # The epoch panel sets its own.
    for ax, panel in zip(axes, panels):
        if panel not in ("__epoch__",):
            ax.set_xlabel("step (global batch counter)")

    title_suffix = "  [AWR]" if is_awr else ""
    fig.suptitle(f"{csv_path}{title_suffix}", fontsize=10)
    fig.tight_layout()

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"wrote {out}")
    else:
        _center_window_on_screen(fig)
        plt.show()


def _center_window_on_screen(fig) -> None:
    """Move the matplotlib window so it's centred on the user's screen.

    Matplotlib doesn't expose this directly — the API differs per
    backend (Tk, Qt, GTK, Wx). Try each known path, swallow failures
    silently so the plot still shows even on an unrecognised backend.
    """
    try:
        manager = fig.canvas.manager
    except AttributeError:
        return
    # Force the figure to render once so width/height are known.
    try:
        fig.canvas.draw()
    except Exception:  # noqa: BLE001
        pass

    window = getattr(manager, "window", None)
    if window is None:
        return

    # TkAgg — window is a Tk Toplevel.
    if hasattr(window, "winfo_screenwidth"):
        try:
            window.update_idletasks()
            w = window.winfo_width()
            h = window.winfo_height()
            screen_w = window.winfo_screenwidth()
            screen_h = window.winfo_screenheight()
            x = max(0, (screen_w - w) // 2)
            y = max(0, (screen_h - h) // 2)
            window.geometry(f"+{x}+{y}")
            return
        except Exception:  # noqa: BLE001
            pass

    # Qt5Agg / QtAgg — window is a QMainWindow.
    if hasattr(window, "screen") and hasattr(window, "move"):
        try:
            geom = window.frameGeometry()
            screen_geom = window.screen().availableGeometry()
            x = screen_geom.x() + (screen_geom.width() - geom.width()) // 2
            y = screen_geom.y() + (screen_geom.height() - geom.height()) // 2
            window.move(x, y)
            return
        except Exception:  # noqa: BLE001
            pass

    # GTK3Agg — window is a Gtk.Window.
    if hasattr(window, "get_screen") and hasattr(window, "move"):
        try:
            screen = window.get_screen()
            w, h = window.get_size()
            x = (screen.get_width() - w) // 2
            y = (screen.get_height() - h) // 2
            window.move(x, y)
            return
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="step.csv file or run directory containing it")
    ap.add_argument("--window", type=int, default=100, help="rolling-mean window in batches (default 100)")
    ap.add_argument("--out", type=Path, default=None, help="save PNG here instead of opening a window")
    args = ap.parse_args()
    _plot(_resolve_csv(args.path), args.window, args.out)


if __name__ == "__main__":
    main()
