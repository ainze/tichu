r"""Shrink-and-perturb a co-train champion into warm-start Checkpoints.

The deciding experiment of docs/notes/2026-10-10-train-from-random-options-memo.md:
does the v7 co-train plateau come from WHERE training started, or from the
learning signal? Perturb the champion, re-train it with the v7 recipe, and see
whether it climbs past the champion or settles back to the same level.

Reads a rollout-weights `.pt` ({"models": {net: state_dict}, "critic": ...}, the
format of `_champion.pt` and the `*_opponent.pt` files), blends the selected nets
toward a fresh init (`p <- alpha*p + beta*p_fresh`, see
`tichu_training.shrink_perturb`), and writes:

  <out-dir>/{play,schupfen,tichu,grand}.bin   warm_start Checkpoints (all four;
                                              nets not in --nets are copied as-is)
  <out-dir>/source_opponent.pt                frozen copy of the SOURCE weights, for
                                              the gate's observe-only extra opponent
  <out-dir>/manifest.json                     inputs, sha256, per-net stats

`--alpha 1 --beta 0` writes the source unchanged: that is the CONTROL arm, which
isolates the perturbation from everything else a fresh run resets (Adam state,
critic, KL controller, gate pool).

`--run-dir` + `--bc-opponent` pre-seed the arm's `_bc_opponent.pt`. Without this,
the co-train builds its "bc" gate opponent from the warm-start nets — here, the
(perturbed) champion — and the vs-BC stream would no longer measure the plateau.
Copying the v7 run's file keeps that stream comparable to the v7 run's
window-for-window.

`--seed-champion` (needs `--run-dir`) also copies the SOURCE weights in as the arm's
`_champion.pt`. Without it, the co-train seeds the champion from the warm-start, i.e.
the PERTURBED net. In arm P v1 that left a wrecked net as the rollout opponent for
half of all games, because `also_beat_bc` blocked every promotion, while the control
arm trained against the real champion. Seeded, both arms start the ratchet from the
same champion and train against the same opponents.

Usage (worktree root, this worktree's src FIRST on PYTHONPATH):

  python scripts/shrink_perturb_champion.py `
      --weights     C:\workbench\tichu\data\runs\cotrain_v7_gated\_champion.pt `
      --config      configs\cotrain_v7_snp_a090.yaml `
      --alpha 0.9 --beta 0.1 --seed 0 `
      --out-dir     C:\workbench\tichu\data\snp_v7\a090 `
      --run-dir     C:\workbench\tichu\data\runs\cotrain_v7_snp_a090 `
      --bc-opponent C:\workbench\tichu\data\runs\cotrain_v7_gated\_bc_opponent.pt `
      --seed-champion
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import shutil
import sys
from pathlib import Path

import torch
import yaml

from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.training import load_checkpoint
from tichu_training.checkpoint import Checkpoint
from tichu_training.cli.train_cotrain import _build_models
from tichu_training.featurizer import FEATURIZER_VERSION
from tichu_training.shrink_perturb import NET_TYPES as _NET_TYPES
from tichu_training.shrink_perturb import build_perturbed

log = logging.getLogger("shrink_perturb_champion")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_checkpoint(model: torch.nn.Module, dest: Path) -> None:
    buf = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": {}, "step": 0}, buf)
    Checkpoint(featurizer_version=FEATURIZER_VERSION,
               action_space_version=ACTION_SPACE_VERSION,
               payload=buf.getvalue()).save(dest)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="source rollout-weights .pt")
    p.add_argument("--config", required=True, help="cotrain YAML carrying the arch blocks")
    p.add_argument("--alpha", type=float, required=True, help="shrink factor on trained weights")
    p.add_argument("--beta", type=float, required=True, help="scale on the fresh init")
    p.add_argument("--seed", type=int, default=0, help="seed of the fresh init")
    p.add_argument("--nets", nargs="+", default=["play"], choices=_NET_TYPES,
                   help="nets to perturb (default: play; the rest are copied as-is)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-dir", help="arm run dir to pre-seed with --bc-opponent")
    p.add_argument("--bc-opponent", help="_bc_opponent.pt to copy into --run-dir")
    p.add_argument("--seed-champion", action="store_true",
                   help="copy the SOURCE weights into --run-dir as _champion.pt")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if bool(args.run_dir) != bool(args.bc_opponent):
        p.error("--run-dir and --bc-opponent go together")
    if args.seed_champion and not args.run_dir:
        p.error("--seed-champion needs --run-dir")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    weights = Path(args.weights)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models, stats = build_perturbed(weights, config, alpha=args.alpha, beta=args.beta,
                                    seed=args.seed, nets=list(args.nets))
    for net in _NET_TYPES:
        dest = out_dir / f"{net}.bin"
        _save_checkpoint(models[net], dest)
        # Round-trip through the loader warm_start uses (version pins included).
        check = _build_models(config)[net]
        load_checkpoint(dest, check)
        for (k, a), b in zip(models[net].state_dict().items(), check.state_dict().values()):
            if not torch.equal(a, b):
                raise RuntimeError(f"round-trip mismatch in {net}.{k}")
        s = stats.get(net)
        log.info("%-8s -> %s%s", net, dest.name,
                 f"  rel_change={s['rel_change']:.3f} norm {s['norm_before']:.1f}->{s['norm_after']:.1f}"
                 if s else "  (copied)")

    shutil.copyfile(weights, out_dir / "source_opponent.pt")

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        seeds = {"_bc_opponent.pt": Path(args.bc_opponent)}
        if args.seed_champion:
            seeds["_champion.pt"] = weights
        if (run_dir / "train_state.bin").exists():
            log.warning("%s already has a Resume Bundle; leaving %s alone",
                        run_dir, ", ".join(seeds))
        else:
            for name, src in seeds.items():
                shutil.copyfile(src, run_dir / name)
                log.info("seeded %s <- %s", run_dir / name, src)

    manifest = {
        "source_weights": str(weights),
        "source_sha256": _sha256(weights),
        "alpha": args.alpha, "beta": args.beta, "seed": args.seed,
        "nets": list(args.nets),
        "featurizer_version": FEATURIZER_VERSION,
        "stats": stats,
        "bc_opponent": args.bc_opponent,
        "seeded_champion": bool(args.seed_champion),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("wrote %s", out_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
