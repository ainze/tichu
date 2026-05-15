"""Smoke test for `export_model` CLI."""

from pathlib import Path

import pytest
import torch

from tichu_export.torchscript import load_exported
from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.training import save_checkpoint as save_bc_checkpoint
from tichu_training.featurizer import FEATURIZER_VERSION


_FEATURE_DIM = 32


def _bc_ckpt(tmp_path: Path) -> tuple[Path, dict]:
    from tichu_training.bc.heads import BCModel
    torch.manual_seed(0)
    model = BCModel(
        feature_dim=_FEATURE_DIM,
        skill_buckets=10, skill_dim=4,
        trunk_hidden=16, trunk_depth=1, trunk_out_dim=16, head_hidden=16,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    out = tmp_path / "bc.bin"
    save_bc_checkpoint(model, opt, step=1, path=out)
    arch = {
        "feature_dim": _FEATURE_DIM,
        "skill_buckets": 10, "skill_dim": 4,
        "trunk_hidden": 16, "trunk_depth": 1, "trunk_out_dim": 16, "head_hidden": 16,
    }
    return out, arch


def _call_ckpt(tmp_path: Path, name: str) -> tuple[Path, dict]:
    from tichu_training.bc.call_model import TichuCallNetwork
    torch.manual_seed(0)
    net = TichuCallNetwork(feature_dim=_FEATURE_DIM, skill_buckets=10, skill_dim=4, hidden=16)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    out = tmp_path / f"{name}.bin"
    save_bc_checkpoint(net, opt, step=1, path=out)
    arch = {"feature_dim": _FEATURE_DIM, "skill_buckets": 10, "skill_dim": 4, "hidden": 16}
    return out, arch


def _write_model_cfg(tmp_path: Path, bc_arch, tichu_arch, grand_arch) -> Path:
    import yaml
    cfg_path = tmp_path / "model_arch.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "policy": bc_arch,
        "tichu_call": tichu_arch,
        "grand_tichu_call": grand_arch,
    }), encoding="utf-8")
    return cfg_path


def test_cli_exports_policy_plus_two_call_networks(tmp_path):
    from tichu_training.cli.export_model import main

    bc_path, bc_arch = _bc_ckpt(tmp_path)
    tichu_path, tichu_arch = _call_ckpt(tmp_path, "tichu_call")
    grand_path, grand_arch = _call_ckpt(tmp_path, "grand_tichu_call")
    cfg = _write_model_cfg(tmp_path, bc_arch, tichu_arch, grand_arch)
    out_dir = tmp_path / "exported"

    rc = main([
        "--checkpoint", str(bc_path),
        "--tichu-checkpoint", str(tichu_path),
        "--grand-checkpoint", str(grand_path),
        "--model-config", str(cfg),
        "--format", "torchscript",
        "--output", str(out_dir),
    ])
    assert rc == 0
    assert (out_dir / "policy.pt").exists()
    assert (out_dir / "tichu_call.pt").exists()
    assert (out_dir / "grand_tichu_call.pt").exists()

    # Each artifact is loadable with the trainer's pinned versions.
    policy = load_exported(
        out_dir / "policy.pt",
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    call = load_exported(
        out_dir / "tichu_call.pt",
        expected_featurizer_version=FEATURIZER_VERSION,
    )
    # Policy output matches eager forward.
    inputs = (torch.randn(1, _FEATURE_DIM), torch.tensor([0], dtype=torch.long))
    with torch.no_grad():
        ts_out = policy(*inputs)
    assert "play" in ts_out


def test_cli_only_policy_exports_without_call_flags(tmp_path):
    from tichu_training.cli.export_model import main
    bc_path, bc_arch = _bc_ckpt(tmp_path)
    cfg = _write_model_cfg(tmp_path, bc_arch, {}, {})
    out_dir = tmp_path / "exported"
    rc = main([
        "--checkpoint", str(bc_path),
        "--model-config", str(cfg),
        "--format", "torchscript",
        "--output", str(out_dir),
    ])
    assert rc == 0
    assert (out_dir / "policy.pt").exists()
    assert not (out_dir / "tichu_call.pt").exists()


def test_cli_rejects_onnx_format(tmp_path):
    from tichu_training.cli.export_model import main
    bc_path, bc_arch = _bc_ckpt(tmp_path)
    cfg = _write_model_cfg(tmp_path, bc_arch, {}, {})
    out_dir = tmp_path / "exported"
    rc = main([
        "--checkpoint", str(bc_path),
        "--model-config", str(cfg),
        "--format", "onnx",
        "--output", str(out_dir),
    ])
    assert rc != 0
