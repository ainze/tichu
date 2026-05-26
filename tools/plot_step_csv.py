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

Usage:
    ./plot_step_csv.py path/to/step.csv
    ./plot_step_csv.py path/to/run-dir            # auto-finds step.csv
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


def _plot(csv_path: Path, window: int, out: Path | None) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        sys.exit(f"{csv_path} has no data rows yet")

    fig, axes = plt.subplots(
        len(HEADS), 1,
        figsize=(11, 3.2 * len(HEADS)),
        sharex=True,
    )
    if len(HEADS) == 1:
        axes = [axes]

    for ax, head in zip(axes, HEADS):
        loss_col = f"loss_{head}"
        acc_col = f"acc_{head}"
        # Each row is a single head's batch — the other heads' columns
        # are zero. Filter to the rows where this head actually fired.
        sub = df[df[loss_col] > 0]
        if sub.empty:
            ax.set_title(f"{head} — no batches fired")
            ax.set_ylabel("loss")
            continue

        x = sub["step"]
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

        # One legend per subplot, combining both axes.
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    axes[-1].set_xlabel("step (global batch counter)")
    fig.suptitle(str(csv_path), fontsize=10)
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
