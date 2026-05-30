r"""Compare per-head BC training trajectories between two runs.

Reads `step.csv` from two run-dirs (or two explicit paths) and prints a
side-by-side per-head, per-quartile loss/accuracy table with deltas.
Used to A/B featurizer versions, model architectures, or hyperparam
sweeps at the training-loss level — Move Prediction Eval on a held-out
set is still the canonical quality metric, but the training trajectory
catches no-op or catastrophic deltas immediately, without needing
checkpoints to be eval-ready.

Quartile slicing handles the wish/dragon_assignment heads, which fire
in only ~1.5% of batches each — small enough that first/middle/last
windows of a fixed size would overlap. Each quartile is a contiguous
1/4 slice of the head's fired-batch sequence in run-order.

If both runs were trained with the same seed + dataset, fired counts
per head will match exactly; the script warns when they don't (signals
a non-comparable A/B — different data scale, different head_weights, or
different filter).

Use:
  py -3.14 scripts/compare_bc_runs.py `
    --a C:\workbench\tichu\data\runs\bc_full_100k_v2 `
    --b C:\workbench\tichu\data\runs\bc_full_100k_v4_memmap

Optional `--weights` reads a config YAML's `head_weights` block and
prints a weighted-accuracy net delta (the same combination the loss
uses), to summarise overall A/B direction in one number.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path


log = logging.getLogger("compare_bc_runs")

HEADS: tuple[str, ...] = ("play", "wish", "dragon_assignment")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--a", required=True, metavar="PATH",
                   help="Run-dir A (containing step.csv) or path to step.csv directly")
    p.add_argument("--b", required=True, metavar="PATH",
                   help="Run-dir B (containing step.csv) or path to step.csv directly")
    p.add_argument("--label-a", default="A", metavar="LABEL",
                   help="Short label for run A in the table header (default: A)")
    p.add_argument("--label-b", default="B", metavar="LABEL",
                   help="Short label for run B in the table header (default: B)")
    p.add_argument("--weights", metavar="YAML",
                   help="Optional config YAML whose `head_weights` block is used "
                        "to compute a weighted-accuracy net delta")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    rows_a = load_step_csv(_resolve_step_csv(args.a))
    rows_b = load_step_csv(_resolve_step_csv(args.b))
    log.info("%s batches: %d    %s batches: %d",
             args.label_a, len(rows_a), args.label_b, len(rows_b))
    if len(rows_a) != len(rows_b):
        log.warning(
            "batch counts differ between runs (%d vs %d). Quartile slicing "
            "still works, but the runs cover different totals of training data.",
            len(rows_a), len(rows_b),
        )

    weights = _load_head_weights(args.weights) if args.weights else None

    weighted_acc_delta = 0.0
    for head in HEADS:
        stats_a = head_stats(rows_a, head)
        stats_b = head_stats(rows_b, head)
        if stats_a is None or stats_b is None:
            log.info("%s: head fires zero batches in one of the runs, skipping",
                     head)
            continue
        if stats_a["n"] != stats_b["n"]:
            log.warning(
                "%s fired counts differ (%s n=%d  %s n=%d) — not a clean A/B "
                "for this head",
                head, args.label_a, stats_a["n"], args.label_b, stats_b["n"],
            )
        _print_head_table(head, stats_a, stats_b, args.label_a, args.label_b)

        if weights and head in weights:
            # Final-quartile (Q4) acc delta, weighted as the loss weights it.
            final_delta = stats_b["quartiles"][-1][3] - stats_a["quartiles"][-1][3]
            weighted_acc_delta += final_delta * weights[head]

    if weights:
        print()
        print(
            f"weighted Q4 acc delta ({args.label_b} - {args.label_a}, "
            f"head_weights from {args.weights}): {weighted_acc_delta:+.4f}"
        )

    return 0


def load_step_csv(path: Path) -> list[dict[str, str]]:
    """Read a `step.csv` written by `train_bc` into a list of row dicts.

    No type coercion happens here — head_stats casts the columns it
    consumes, so adding a column to step.csv doesn't break this loader.
    """
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def head_stats(rows: list[dict[str, str]], head: str) -> dict | None:
    """Quartile-slice the batches in which `head` fired.

    A batch is considered to have fired the head iff `loss_<head> > 0`.
    train_bc emits one row per batch and exactly one head per batch, so
    the play/wish/dragon_assignment fired-sets are disjoint and union
    to the whole step.csv.

    Returns a dict with `n` (total fired batches) and `quartiles` (a list
    of `(name, count, mean_loss, mean_acc)`), or None if the head never
    fired in this run.
    """
    fired = [r for r in rows if float(r[f"loss_{head}"]) > 0.0]
    if not fired:
        return None
    n = len(fired)
    q = max(1, n // 4)
    quartiles = [
        ("Q1", fired[:q]),
        ("Q2", fired[q:2 * q]),
        ("Q3", fired[2 * q:3 * q]),
        # Q4 picks up the remainder so floor-division loss lands in the
        # tail bucket — Q4 is also the one you care about most, so any
        # extra observation it gets is a feature.
        ("Q4", fired[3 * q:]),
    ]
    out: dict = {"n": n, "quartiles": []}
    for name, sl in quartiles:
        loss = sum(float(r[f"loss_{head}"]) for r in sl) / len(sl)
        acc = sum(float(r[f"acc_{head}"]) for r in sl) / len(sl)
        out["quartiles"].append((name, len(sl), loss, acc))
    return out


def _resolve_step_csv(path_str: str) -> Path:
    """Accept either a run-dir (containing step.csv) or a step.csv path
    directly. Raises FileNotFoundError if neither shape resolves.
    """
    p = Path(path_str)
    if p.is_dir():
        candidate = p / "step.csv"
        if not candidate.exists():
            raise FileNotFoundError(
                f"{p} is a directory but contains no step.csv"
            )
        return candidate
    if not p.exists():
        raise FileNotFoundError(f"no such file or directory: {p}")
    return p


def _load_head_weights(yaml_path: str) -> dict[str, float]:
    """Pull the `head_weights` block out of a train_bc config YAML.

    Imports yaml lazily so the comparison script works without yaml
    installed when no --weights flag is passed.
    """
    import yaml
    config = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    raw = config.get("head_weights") or {}
    return {k: float(v) for k, v in raw.items()}


def _print_head_table(
    head: str,
    stats_a: dict,
    stats_b: dict,
    label_a: str,
    label_b: str,
) -> None:
    print(f"\n=== {head.upper()} ===")
    print(f"  fired:   {label_a} n={stats_a['n']:>6}     "
          f"{label_b} n={stats_b['n']:>6}")
    print(f"  {'quartile':>8}  "
          f"{label_a + ' loss':>10} {label_a + ' acc':>9}    "
          f"{label_b + ' loss':>10} {label_b + ' acc':>9}    "
          f"{'dloss':>8} {'dacc':>7}")
    for (qn, _, la, aa), (_, _, lb, ab) in zip(
        stats_a["quartiles"], stats_b["quartiles"],
    ):
        print(f"  {qn:>8}  "
              f"{la:>10.4f} {aa:>9.4f}    "
              f"{lb:>10.4f} {ab:>9.4f}    "
              f"{lb - la:>+8.4f} {ab - aa:>+7.4f}")


if __name__ == "__main__":
    sys.exit(main())
