"""scripts/shrink_perturb_champion.py end to end on a tiny stack.

The warm-start files must load through the same version-pinned loader the
co-train uses, only the selected nets may change, and the run-dir seeding must
put the GIVEN bc opponent in place (else the co-train would build its "bc" gate
opponent from the perturbed champion and the vs-BC stream would stop measuring
the plateau).
"""

import json

import pytest
import torch

from scripts import shrink_perturb_champion as spc
from tichu_training.bc.training import load_checkpoint
from tichu_training.cli.train_cotrain import _build_models

_TINY = {
    "model": {"skill_dim": 4, "trunk_hidden": 16, "trunk_depth": 1, "trunk_out_dim": 8,
              "head_hidden": 8, "use_legal_mask": True},
    "schupfen_model": {"skill_dim": 4, "hidden": 8, "depth": 1, "residual": True},
    "call_model": {"skill_dim": 4, "hidden": 8, "depth": 1, "residual": True},
    "grand_model": {"skill_dim": 4, "hidden": 8},
}


@pytest.fixture
def stack(tmp_path):
    import yaml
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(_TINY), encoding="utf-8")
    torch.manual_seed(123)
    models = _build_models(_TINY)
    weights = tmp_path / "champ.pt"
    torch.save({"models": {k: m.state_dict() for k, m in models.items()}, "critic": {}}, weights)
    return cfg, weights, models


def _loaded(path, net):
    m = _build_models(_TINY)[net]
    load_checkpoint(path, m)
    return m.state_dict()


def test_control_arm_reproduces_the_source_exactly(stack, tmp_path):
    cfg, weights, models = stack
    out = tmp_path / "out"
    assert spc.main(["--weights", str(weights), "--config", str(cfg), "--alpha", "1",
                     "--beta", "0", "--out-dir", str(out)]) == 0
    for net, m in models.items():
        for k, v in _loaded(out / f"{net}.bin", net).items():
            assert torch.equal(v, m.state_dict()[k]), (net, k)


def test_only_the_selected_nets_are_perturbed(stack, tmp_path):
    cfg, weights, models = stack
    out = tmp_path / "out"
    spc.main(["--weights", str(weights), "--config", str(cfg), "--alpha", "0.6",
              "--beta", "0.4", "--out-dir", str(out)])
    play = _loaded(out / "play.bin", "play")
    assert not torch.equal(play["trunk.input_proj.weight"],
                           models["play"].state_dict()["trunk.input_proj.weight"])
    for net in ("schupfen", "tichu", "grand"):
        for k, v in _loaded(out / f"{net}.bin", net).items():
            assert torch.equal(v, models[net].state_dict()[k]), (net, k)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["nets"] == ["play"] and manifest["alpha"] == 0.6
    assert 0.0 < manifest["stats"]["play"]["rel_change"] < 1.0
    assert (out / "source_opponent.pt").read_bytes() == weights.read_bytes()


def test_same_seed_gives_the_same_perturbation(stack, tmp_path):
    cfg, weights, _ = stack
    for d in ("a", "b"):
        spc.main(["--weights", str(weights), "--config", str(cfg), "--alpha", "0.6",
                  "--beta", "0.4", "--seed", "7", "--out-dir", str(tmp_path / d)])
    a, b = _loaded(tmp_path / "a" / "play.bin", "play"), _loaded(tmp_path / "b" / "play.bin", "play")
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_run_dir_is_seeded_with_the_given_bc_opponent(stack, tmp_path):
    cfg, weights, _ = stack
    bc = tmp_path / "bc.pt"
    bc.write_bytes(b"the v7 bc opponent")
    run_dir = tmp_path / "run"
    spc.main(["--weights", str(weights), "--config", str(cfg), "--alpha", "0.6",
              "--beta", "0.4", "--out-dir", str(tmp_path / "out"),
              "--run-dir", str(run_dir), "--bc-opponent", str(bc)])
    assert (run_dir / "_bc_opponent.pt").read_bytes() == b"the v7 bc opponent"


def test_run_dir_with_a_resume_bundle_is_left_alone(stack, tmp_path):
    cfg, weights, _ = stack
    bc = tmp_path / "bc.pt"
    bc.write_bytes(b"new")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "train_state.bin").write_bytes(b"bundle")
    (run_dir / "_bc_opponent.pt").write_bytes(b"old")
    spc.main(["--weights", str(weights), "--config", str(cfg), "--alpha", "0.6",
              "--beta", "0.4", "--out-dir", str(tmp_path / "out"),
              "--run-dir", str(run_dir), "--bc-opponent", str(bc)])
    assert (run_dir / "_bc_opponent.pt").read_bytes() == b"old"


def test_seed_champion_puts_the_unperturbed_source_in_the_run_dir(stack, tmp_path):
    cfg, weights, _ = stack
    bc = tmp_path / "bc.pt"
    bc.write_bytes(b"bc")
    run_dir = tmp_path / "run"
    spc.main(["--weights", str(weights), "--config", str(cfg), "--alpha", "0.6",
              "--beta", "0.4", "--out-dir", str(tmp_path / "out"),
              "--run-dir", str(run_dir), "--bc-opponent", str(bc), "--seed-champion"])
    # The SOURCE, not the perturbed net: the arm's ratchet starts from the real champion.
    assert (run_dir / "_champion.pt").read_bytes() == weights.read_bytes()
    assert (run_dir / "_bc_opponent.pt").read_bytes() == b"bc"
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seeded_champion"] is True


def test_without_seed_champion_no_champion_is_written(stack, tmp_path):
    cfg, weights, _ = stack
    bc = tmp_path / "bc.pt"
    bc.write_bytes(b"bc")
    run_dir = tmp_path / "run"
    spc.main(["--weights", str(weights), "--config", str(cfg), "--alpha", "0.6",
              "--beta", "0.4", "--out-dir", str(tmp_path / "out"),
              "--run-dir", str(run_dir), "--bc-opponent", str(bc)])
    assert not (run_dir / "_champion.pt").exists()


def test_seed_champion_leaves_a_resumed_run_alone(stack, tmp_path):
    cfg, weights, _ = stack
    bc = tmp_path / "bc.pt"
    bc.write_bytes(b"bc")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "train_state.bin").write_bytes(b"bundle")
    (run_dir / "_champion.pt").write_bytes(b"promoted mid-run")
    spc.main(["--weights", str(weights), "--config", str(cfg), "--alpha", "0.6",
              "--beta", "0.4", "--out-dir", str(tmp_path / "out"),
              "--run-dir", str(run_dir), "--bc-opponent", str(bc), "--seed-champion"])
    assert (run_dir / "_champion.pt").read_bytes() == b"promoted mid-run"


def test_seed_champion_without_run_dir_is_refused(stack, tmp_path):
    cfg, weights, _ = stack
    with pytest.raises(SystemExit):
        spc.main(["--weights", str(weights), "--config", str(cfg), "--alpha", "0.6",
                  "--beta", "0.4", "--out-dir", str(tmp_path / "out"), "--seed-champion"])
