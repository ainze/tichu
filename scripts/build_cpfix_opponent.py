r"""Assemble the shipped cotrain champion (iter 3328 of
cotrain_v6_pbrs_resid_wish_gated_cpfix) into a rollout-weights .pt so it can be
used as a FIXED gate/rollout opponent (`promotion_gate.extra_opponents`) in a
new cotrain run.

The snapshot is four per-net `.bin` checkpoints; both the rollout worker
(`_league_opponent`) and the greedy gate (`export_opponent`) want a single
`{"models": {net: state_dict}}` file rebuilt with `_build_models(arch_cfg)`. The
critic is unused by an opponent, so it is omitted.

The arch MUST match the run that produced the snapshot (resid 512x4 schupfen +
tichu, grand 256, play 1024x4) — which is also the arch of the new run, so the
same `_arch_cfg` rebuilds both this opponent and the run's own champion.

  python scripts/build_cpfix_opponent.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(r"C:\workbench\tichu\.claude\worktrees\epic-dubinsky-b29c6c")
sys.path.insert(0, str(ROOT / "src"))

from tichu_training.bc.training import load_checkpoint
from tichu_training.cli.train_cotrain import _build_models, _NET_TYPES

# Must match cotrain_v6_pbrs_resid_wish_gated(_cpfix) arch AND the new run's arch.
ARCH = {
    "model":          {"skill_dim": 64, "trunk_hidden": 1024, "trunk_depth": 4, "trunk_out_dim": 512, "head_hidden": 256},
    "schupfen_model": {"skill_dim": 64, "hidden": 512, "depth": 4, "residual": True},
    "call_model":     {"skill_dim": 64, "hidden": 512, "depth": 4, "residual": True},
    "grand_model":    {"skill_dim": 64, "hidden": 256},
}
RUN = Path(r"C:\workbench\tichu\data\runs\cotrain_v6_pbrs_resid_wish_gated_cpfix")
ITER = 3328
OUT = RUN / f"iter{ITER:05d}_opponent.pt"

models = _build_models(ARCH)
for net in _NET_TYPES:
    src = RUN / "snapshots" / f"iter_{ITER:05d}_{net}.bin"
    load_checkpoint(str(src), models[net])
    print(f"loaded {src.name}")

torch.save({"models": {net: models[net].state_dict() for net in _NET_TYPES}}, OUT)
print(f"wrote {OUT}")

# Sanity: re-export through the EXACT gate path to prove the arch matches and the
# state dicts load into a fresh build with no shape errors.
from tichu_training.ppo.greedy_gate import export_opponent  # noqa: E402

kwargs = export_opponent(str(OUT), ARCH, RUN / "_opponent_export_smoke")
print("export_opponent OK:", {k: Path(v).name for k, v in kwargs.items()})
