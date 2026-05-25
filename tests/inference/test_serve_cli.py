"""serve_inference CLI — config wiring + in-process smoke."""

from pathlib import Path

import pytest
import torch
import yaml
from fastapi.testclient import TestClient

from tichu_export.torchscript import export_torchscript
from tichu_inference.cli.serve import build_app_for_config, main


def _dummy_artifact(tmp_path: Path, name: str) -> Path:
    from tichu_training.action_space import ACTION_SPACE_SIZE
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Linear(FEATURIZER_OUTPUT_DIM, ACTION_SPACE_SIZE)
            self.pc = torch.nn.Linear(FEATURIZER_OUTPUT_DIM, 3)
            self.wr = torch.nn.Linear(FEATURIZER_OUTPUT_DIM, 14)
            self.dg = torch.nn.Linear(FEATURIZER_OUTPUT_DIM, 2)

        def forward(self, features, skill):
            return {"play": self.f(features), "schupfen": self.pc(features),
                    "wish": self.wr(features), "dragon_assignment": self.dg(features)}

    torch.manual_seed(0)
    out = tmp_path / f"{name}.pt"
    export_torchscript(
        M(),
        example_inputs=(torch.randn(1, FEATURIZER_OUTPUT_DIM), torch.tensor([0], dtype=torch.long)),
        featurizer_version="v1",
        action_space_version="v1",
        output_path=out,
    )
    return out


def _write_config(tmp_path: Path) -> Path:
    cfg = {
        "agents": {
            "easy": {"factory": "rule"},
            "medium": {"factory": "ml", "checkpoint": str(_dummy_artifact(tmp_path, "medium"))},
            "hard": {"factory": "ml", "checkpoint": str(_dummy_artifact(tmp_path, "hard"))},
            "master": {"factory": "ml", "checkpoint": str(_dummy_artifact(tmp_path, "master"))},
        },
        "port": 8000,
    }
    path = tmp_path / "serve.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_build_app_for_config_returns_working_app(tmp_path):
    cfg_path = _write_config(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    app = build_app_for_config(cfg)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert set(r.json()["agents"]) == {"easy", "medium", "hard", "master"}


def test_missing_config_file_exits_nonzero(tmp_path):
    rc = main(["--config", str(tmp_path / "does_not_exist.yaml")])
    assert rc != 0


def test_main_with_no_serve_does_not_bind_port(tmp_path):
    cfg = _write_config(tmp_path)
    rc = main(["--config", str(cfg), "--no-serve"])
    assert rc == 0
