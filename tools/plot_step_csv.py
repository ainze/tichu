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
    """
    ep = edf["epoch"]
    wrp = edf.get("win_rate_proxy")
    aw = edf.get("avg_weight")

    plotted = False
    if wrp is not None:
        # win_rate_proxy may be empty strings when held_out_subset wasn't set.
        wrp_clean = pd.to_numeric(wrp, errors="coerce")
        if wrp_clean.notna().any():
            ax.plot(ep, wrp_clean, "o-", color="C3", linewidth=1.8, label="win_rate_proxy (held-out)")
            ax.set_ylim(0, 1)
            plotted = True
    ax.set_ylabel("win_rate_proxy", color="C3")
    ax.tick_params(axis="y", labelcolor="C3")
    ax.set_xlabel("epoch")
    ax.set_title("AWR per-epoch summary")
    ax.grid(True, alpha=0.3)
    # Integer ticks for epoch counts.
    ax.set_xticks(list(ep))

    if aw is not None:
        aw_clean = pd.to_numeric(aw, errors="coerce")
        if aw_clean.notna().any():
            ax2 = ax.twinx()
            ax2.plot(ep, aw_clean, "s--", color="C2", linewidth=1.2, alpha=0.85, label="avg_weight")
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
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="step.csv file or run directory containing it")
    ap.add_argument("--window", type=int, default=100, help="rolling-mean window in batches (default 100)")
    ap.add_argument("--out", type=Path, default=None, help="save PNG here instead of opening a window")
    args = ap.parse_args()
    _plot(_resolve_csv(args.path), args.window, args.out)


if __name__ == "__main__":
    main()
