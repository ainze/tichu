"""cross_version_tournament — literal head-to-head between a v6 and a v5 export.

The transitive BC-anchor comparison (cotrain_v6 vs v6-BC, cotrain_wish_v5 vs
v5-BC) only *approximates* "is v6 stronger than the prior champion". This runs the
two agents at the SAME table. It is faithful because v6 only ADDED engine-state
fields (ADR-0038): the v5 net is fed its native 224-dim features by
`featurizer_v5_frozen` running on the v6 `PrivateState`, proven invariant to the
v6-added fields (`tests/training/featurizer/test_featurizer_v5_frozen.py`). The
action space is shared (`v1`), and the v6 commit changed no scoring/legality — so a
v6-process game IS a v5-process game.

    py -m tichu_training.cli.cross_version_tournament \
        --v6-export C:\...\data\runs\cotrain_v6\export\iter_00225 \
        --v5-export C:\...\data\export\cotrain_wish_v5 \
        --pool      C:\...\data\full_position_pool_s0_n20000.parquet \
        --n-deals 8000 --workers 8 --seed 0

Positive mean ⇒ the v6 agent out-scores the v5 champion per round; CI lower bound
> 0 is the ship-grade verdict. Heavy job (process pool) — do NOT run while training
is using the CPU (OOM, ADR-0034).
"""

import argparse
import importlib
import sys
from functools import partial
from pathlib import Path

from tichu_eval.full_position_pool import load_full_position_pool
from tichu_eval.tournament import run_full_tournament
from tichu_ml.registry import build_agent

# Side-effect import: registers the `ml` agent factory (and makes it resolvable in
# spawned tournament workers, mirroring check_cotrain / eval_matrix).
import tichu_inference.ml_agent  # noqa: F401,E402

# Featurizer selected by STRING TAG, never by passing the module object: a module
# cannot be pickled, so a builder carrying one would die at the process-pool
# boundary. The tag is resolved by import INSIDE the worker.
_FEATURIZER_MODULES = {
    "v6": "tichu_training.featurizer",
    "v5_frozen": "tichu_training.featurizer_v5_frozen",
}


def build_versioned_ml_agent(featurizer_tag: str, **kwargs):
    """Spawn-safe `ml`-agent builder: resolve the featurizer by tag and inject it.

    Top-level (so workers can import it) and module-free in its bound args (only the
    string tag + string paths cross the pickle boundary)."""
    fz = importlib.import_module(_FEATURIZER_MODULES[featurizer_tag])
    return build_agent("ml", featurizer=fz, **kwargs)


def _export_paths(export_dir: str) -> dict:
    d = Path(export_dir)
    return {
        "checkpoint_path": str(d / "policy.pt"),
        "schupfen_path": str(d / "schupfen.pt"),
        "tichu_call_path": str(d / "tichu_call.pt"),
        "grand_call_path": str(d / "grand_tichu_call.pt"),
    }


def run(*, v6_export: str, v5_export: str, pool: str, v6_name: str, v5_name: str,
        n_deals: int | None, workers: int, seed: int, skill_decile: int,
        bootstrap_iters: int, out: str | None = None, progress: bool = True) -> dict:
    builders = {
        v6_name: partial(build_versioned_ml_agent, "v6",
                         skill_decile=skill_decile, **_export_paths(v6_export)),
        v5_name: partial(build_versioned_ml_agent, "v5_frozen",
                         skill_decile=skill_decile, **_export_paths(v5_export)),
    }
    positions = load_full_position_pool(Path(pool))
    if n_deals is not None:
        positions = positions[:n_deals]
    if progress:
        print(f"cross-version tournament: {v6_name} (v6) vs {v5_name} (v5-frozen)  "
              f"{len(positions)} deals, {workers} workers, decile {skill_decile}",
              flush=True)

    result = run_full_tournament(
        builders, positions,
        bootstrap_iters=bootstrap_iters, seed=seed, workers=workers, progress=None,
    )
    mean = float(result.mean(v6_name, v5_name))
    lo, hi = (float(x) for x in result.ci(v6_name, v5_name))
    n = int(result.n(v6_name, v5_name))
    win_rate = float(result.win_rate(v6_name, v5_name))
    tie_rate = float(result.tie_rate(v6_name, v5_name))
    ship = lo > 0.0

    if out is not None:
        from tichu_training.cli.eval_matrix import _write_matrix
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        _write_matrix(result, Path(out))

    if progress:
        verdict = ("*** v6 STRONGER: CI clears 0 ***" if ship
                   else "inconclusive (CI spans 0)")
        print(f"{v6_name} vs {v5_name} (seed {seed}):  mean={mean:+.2f}  "
              f"CI=[{lo:+.2f}, {hi:+.2f}]  n={n}  -> {verdict}", flush=True)
        print(f"  round win-rate: {v6_name} {win_rate:.1%}  vs {v5_name} "
              f"{1.0 - win_rate - tie_rate:.1%}  (tie {tie_rate:.1%}, n={n})", flush=True)

    return {"mean": mean, "ci": (lo, hi), "n": n, "win_rate": win_rate,
            "tie_rate": tie_rate, "ship": ship}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Head-to-head tournament between a v6 and a v5 MLAgent export.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--v6-export", required=True, help="dir with the v6 agent's 4 .pt exports.")
    ap.add_argument("--v5-export", required=True, help="dir with the v5 agent's 4 .pt exports.")
    ap.add_argument("--pool", required=True, help="starting-position pool parquet (engine-level, version-agnostic).")
    ap.add_argument("--v6-name", default="cotrain_v6")
    ap.add_argument("--v5-name", default="cotrain_wish_v5")
    ap.add_argument("--n-deals", type=int, default=8000, help="cap on positions (default 8000; omit/<=0 for all).")
    ap.add_argument("--workers", type=int, default=1, help="process-pool size (do NOT overlap training — OOM).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skill-decile", type=int, default=9)
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--out", default=None, help="optional parquet path for the full matrix.")
    args = ap.parse_args(argv)

    run(v6_export=args.v6_export, v5_export=args.v5_export, pool=args.pool,
        v6_name=args.v6_name, v5_name=args.v5_name,
        n_deals=(args.n_deals if args.n_deals and args.n_deals > 0 else None),
        workers=args.workers, seed=args.seed, skill_decile=args.skill_decile,
        bootstrap_iters=args.bootstrap_iters, out=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
