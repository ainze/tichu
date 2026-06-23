"""[DEBUG-dog] Export the current_player-fixed schupfen checkpoint to TorchScript.

Reuses export_model._export_schupfen (the blessed path) so the .pt carries the
v6 featurizer stamp the eval/MLAgent loader asserts. Arch mirrors
configs/schupfen_full_corpus_v6_resid.yaml (residual 512x4, skill_dim 64).
"""
from pathlib import Path

from tichu_training.cli.export_model import _export_schupfen

CKPT = r"C:\workbench\tichu\data\runs\schupfen_full_corpus_v6_resid_cpfix\checkpoints\schupfen_final.bin"
OUT = Path(r"C:\workbench\tichu\data\runs\schupfen_full_corpus_v6_resid_cpfix\export\schupfen.pt")
ARCH = {"feature_dim": 591, "skill_dim": 64, "hidden": 512, "depth": 4, "residual": True}

OUT.parent.mkdir(parents=True, exist_ok=True)
_export_schupfen(CKPT, ARCH, OUT)
print(f"wrote {OUT}  ({OUT.stat().st_size} bytes)")
