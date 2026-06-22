"""Co-train `_build_models` honours per-head residual arch (PR #64 + grand split).

Regression for the bug where `_build_models` ignored `depth`/`residual` on the
schupfen/call blocks, so warm-starting a residual BC checkpoint crashed on a
strict `load_state_dict` shape mismatch. Also pins the tichu-vs-grand split: a
residual `call_model` (tichu) must NOT force grand residual — grand reads its own
`grand_model` block (falling back to `call_model` only when absent).
"""

import torch

from tichu_training.bc.training import load_checkpoint, save_checkpoint
from tichu_training.cli.train_cotrain import _arch_cfg, _build_models


def _save(net, path):
    save_checkpoint(net, torch.optim.SGD(net.parameters(), lr=0.1), step=0, path=path)


def test_residual_call_does_not_force_grand_residual():
    """call_model residual (tichu) + non-residual grand_model -> grand stays MLP."""
    cfg = {
        "model": dict(skill_dim=8, trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16),
        "schupfen_model": {"skill_dim": 8, "hidden": 32, "depth": 4, "residual": True},
        "call_model": {"skill_dim": 8, "hidden": 32, "depth": 4, "residual": True},
        "grand_model": {"skill_dim": 8, "hidden": 16},
    }
    nets = _build_models(cfg)
    assert nets["schupfen"].residual is True
    assert nets["tichu"].residual is True
    assert nets["grand"].residual is False  # the split: grand ignores call_model


def test_grand_defaults_to_call_model_when_block_absent():
    """Back-compat: no grand_model -> grand inherits the call_model arch."""
    cfg = {
        "model": dict(skill_dim=8, trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16),
        "schupfen_model": {"skill_dim": 8, "hidden": 16},
        "call_model": {"skill_dim": 8, "hidden": 16, "depth": 4, "residual": True},
    }
    nets = _build_models(cfg)
    assert nets["tichu"].residual is True
    assert nets["grand"].residual is True  # inherits call_model (no grand_model)


def test_worker_arch_extraction_preserves_grand_split():
    """The reported crash: the rollout workers (and league opponents) rebuild nets
    from `_arch_cfg(config)`, then `load_state_dict` the main process's live weights.
    If the extraction drops `grand_model`, the worker rebuilds grand as residual and
    the load fails. Reproduce the exact main->extract->worker->load chain."""
    cfg = {
        "model": dict(skill_dim=8, trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16),
        "schupfen_model": {"skill_dim": 8, "hidden": 32, "depth": 4, "residual": True},
        "call_model": {"skill_dim": 8, "hidden": 32, "depth": 4, "residual": True},
        "grand_model": {"skill_dim": 8, "hidden": 16},
    }
    assert "grand_model" in _arch_cfg(cfg)  # the dropped key
    main = _build_models(cfg)                         # main process build (+warm-start)
    worker = _build_models(_arch_cfg(cfg))            # what a rollout worker rebuilds
    assert worker["grand"].residual is False
    for k in ("play", "schupfen", "tichu", "grand"):  # the runtime load_state_dict
        worker[k].load_state_dict(main[k].state_dict())


def test_residual_warm_start_round_trips(tmp_path):
    """The real bug seam: a residual schupfen/tichu checkpoint loads into the
    cotrain-built nets without a shape mismatch, while grand stays non-residual."""
    cfg = {
        "model": dict(skill_dim=8, trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16),
        "schupfen_model": {"skill_dim": 8, "hidden": 32, "depth": 4, "residual": True},
        "call_model": {"skill_dim": 8, "hidden": 32, "depth": 4, "residual": True},
        "grand_model": {"skill_dim": 8, "hidden": 16},
    }
    src = _build_models(cfg)
    paths = {k: tmp_path / f"{k}.bin" for k in ("schupfen", "tichu", "grand")}
    for k, p in paths.items():
        _save(src[k], p)

    dst = _build_models(cfg)
    # strict load: raises RuntimeError on any arch mismatch.
    for k, p in paths.items():
        load_checkpoint(p, dst[k])
    for k in paths:
        for a, b in zip(src[k].parameters(), dst[k].parameters()):
            assert torch.equal(a, b)
