"""FastAPI app — POST /act, GET /health, GET /metrics."""

import logging
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from tichu_engine.legality import legal_actions_for
from tichu_engine.state import deal_initial_state

from tichu_export.torchscript import export_torchscript
from tichu_inference.app import create_app
from tichu_inference.codec import private_state_to_json


def _dummy_policy_artifact(tmp_path: Path, name: str,
                           featurizer_version: str | None = None,
                           action_space_version: str | None = None) -> Path:
    # Defaults track the live module constants so the fixture stays
    # current across featurizer / action-space version bumps. Tests that
    # want to exercise mismatch behaviour pass an explicit override
    # (e.g. "vBAD").
    from tichu_training.action_space import (
        ACTION_SPACE_SIZE, ACTION_SPACE_VERSION,
    )
    from tichu_training.featurizer import (
        FEATURIZER_OUTPUT_DIM, FEATURIZER_VERSION,
    )
    if featurizer_version is None:
        featurizer_version = FEATURIZER_VERSION
    if action_space_version is None:
        action_space_version = ACTION_SPACE_VERSION

    class _DummyPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(FEATURIZER_OUTPUT_DIM, ACTION_SPACE_SIZE)
            self.pc = torch.nn.Linear(FEATURIZER_OUTPUT_DIM, 3)
            self.wr = torch.nn.Linear(FEATURIZER_OUTPUT_DIM, 14)
            self.dg = torch.nn.Linear(FEATURIZER_OUTPUT_DIM, 2)

        def forward(self, features, skill_decile):
            return {
                "play": self.fc(features), "schupfen": self.pc(features),
                "wish": self.wr(features), "dragon_assignment": self.dg(features),
            }

    torch.manual_seed(0)
    model = _DummyPolicy()
    out = tmp_path / f"{name}.pt"
    export_torchscript(
        model,
        example_inputs=(torch.randn(1, FEATURIZER_OUTPUT_DIM), torch.tensor([0], dtype=torch.long)),
        featurizer_version=featurizer_version,
        action_space_version=action_space_version,
        output_path=out,
    )
    return out


def _make_config(tmp_path: Path) -> dict:
    return {
        "agents": {
            "easy": {"factory": "rule"},
            "medium": {"factory": "ml", "checkpoint": str(_dummy_policy_artifact(tmp_path, "medium"))},
            "hard": {"factory": "ml", "checkpoint": str(_dummy_policy_artifact(tmp_path, "hard"))},
            "master": {"factory": "ml", "checkpoint": str(_dummy_policy_artifact(tmp_path, "master"))},
        }
    }


def _post_act(client, difficulty: str, ps):
    return client.post("/act", json={
        "difficulty": difficulty,
        "private_state": private_state_to_json(ps),
    })


def test_health_returns_ok_with_all_agents_loaded(tmp_path):
    app = create_app(_make_config(tmp_path))
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["agents"]) == {"easy", "medium", "hard", "master"}


def test_act_returns_legal_action_for_every_difficulty(tmp_path):
    app = create_app(_make_config(tmp_path))
    client = TestClient(app)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    for difficulty in ("easy", "medium", "hard", "master"):
        r = _post_act(client, difficulty, ps)
        assert r.status_code == 200, (difficulty, r.text)
        body = r.json()
        assert body["fallback_used"] in (True, False)
        assert "action" in body and "kind" in body["action"]


def test_act_rejects_unknown_difficulty(tmp_path):
    app = create_app(_make_config(tmp_path))
    client = TestClient(app)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    r = client.post("/act", json={
        "difficulty": "godlike",
        "private_state": private_state_to_json(ps),
    })
    assert r.status_code == 400


def test_version_mismatch_at_create_app_raises(tmp_path):
    config = {
        "agents": {
            "easy": {"factory": "rule"},
            "medium": {"factory": "ml",
                       "checkpoint": str(_dummy_policy_artifact(tmp_path, "broken", featurizer_version="vBAD"))},
            "hard": {"factory": "rule"},
            "master": {"factory": "rule"},
        }
    }
    with pytest.raises(Exception):
        create_app(config)


def test_fallback_path_reflected_in_response_and_metrics(tmp_path):
    app = create_app(_make_config(tmp_path))
    # Corrupt one of the ML agents to force a fallback.
    from tichu_inference.app import _registry_for_test  # type: ignore[attr-defined]
    registry = _registry_for_test(app)
    class Boom:
        def __call__(self, *a, **k):
            raise RuntimeError("synthetic")
    registry["hard"]._module = Boom()

    client = TestClient(app)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    r = _post_act(client, "hard", ps)
    assert r.status_code == 200
    body = r.json()
    assert body["fallback_used"] is True

    metrics = client.get("/metrics").text
    assert "tichu_fallback_total" in metrics
    assert 'difficulty="hard"' in metrics


def test_metrics_endpoint_exposes_request_counts_and_latency(tmp_path):
    app = create_app(_make_config(tmp_path))
    client = TestClient(app)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    for _ in range(3):
        _post_act(client, "easy", ps)
    text = client.get("/metrics").text
    assert "tichu_requests_total" in text
    assert 'difficulty="easy"' in text
    assert "tichu_latency_p99_ms" in text


def test_returned_action_is_decodable_and_legal(tmp_path):
    app = create_app(_make_config(tmp_path))
    client = TestClient(app)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    r = _post_act(client, "easy", ps)
    assert r.status_code == 200
    legal = legal_actions_for(ps)
    # The "kind" field of the returned action must match the kind of at least
    # one legal engine action.
    body = r.json()
    legal_kinds = {type(a).__name__ for a in legal}
    assert body["action"]["kind"] in legal_kinds
