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


def _dummy_aux_artifact(tmp_path: Path, name: str, out_dim_or_tuple) -> Path:
    """Dummy Call (2 logits) or Schupfen (3 x 56) export, stamped like the real
    standalone nets: featurizer version set, action_space version empty."""
    from tichu_training.card_slots import CARD_SLOTS
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM, FEATURIZER_VERSION

    class _Call(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(FEATURIZER_OUTPUT_DIM, 2)

        def forward(self, features, skill_decile):
            return self.fc(features)

    class _Schupfen(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Linear(FEATURIZER_OUTPUT_DIM, CARD_SLOTS)
            self.b = torch.nn.Linear(FEATURIZER_OUTPUT_DIM, CARD_SLOTS)
            self.c = torch.nn.Linear(FEATURIZER_OUTPUT_DIM, CARD_SLOTS)

        def forward(self, features, skill_decile):
            return self.a(features), self.b(features), self.c(features)

    torch.manual_seed(0)
    model = _Schupfen() if out_dim_or_tuple == "schupfen" else _Call()
    out = tmp_path / f"{name}.pt"
    export_torchscript(
        model,
        example_inputs=(torch.randn(1, FEATURIZER_OUTPUT_DIM), torch.tensor([0], dtype=torch.long)),
        featurizer_version=FEATURIZER_VERSION,
        action_space_version="",
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


def _make_config_with_calls(tmp_path: Path) -> dict:
    cfg = _make_config(tmp_path)
    cfg["agents"]["hard"].update({
        "tichu_call": str(_dummy_aux_artifact(tmp_path, "tichu", "call")),
        "grand_call": str(_dummy_aux_artifact(tmp_path, "grand", "call")),
        "schupfen": str(_dummy_aux_artifact(tmp_path, "schup", "schupfen")),
    })
    return cfg


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


def _post_call(client, difficulty: str, kind: str, ps):
    return client.post("/call", json={
        "difficulty": difficulty, "kind": kind,
        "private_state": private_state_to_json(ps),
    })


def _shrink_hand_to(ps, n: int):
    """Return a copy of `ps` whose hand (and matching public hand_size) is `n`
    cards — for building the 8-card states grand-tichu calls expect."""
    import dataclasses
    sizes = list(ps.public.hand_sizes)
    sizes[ps.player] = n
    public = dataclasses.replace(ps.public, hand_sizes=tuple(sizes))
    hand = frozenset(list(ps.hand)[:n])
    return dataclasses.replace(ps, hand=hand, public=public)


def test_call_endpoint_returns_bool_for_ml_with_call_nets(tmp_path):
    app = create_app(_make_config_with_calls(tmp_path))
    client = TestClient(app)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    # grand is only valid on an 8-card hand; tichu has no fixed size.
    cases = {"tichu": ps, "grand": _shrink_hand_to(ps, 8)}
    for kind, call_state in cases.items():
        r = _post_call(client, "hard", kind, call_state)
        assert r.status_code == 200, (kind, r.text)
        assert isinstance(r.json()["call"], bool)


def test_call_declines_for_rule_baseline(tmp_path):
    app = create_app(_make_config_with_calls(tmp_path))
    client = TestClient(app)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    # easy = RuleAgent has no should_call -> always declines.
    r = _post_call(client, "easy", "tichu", ps)
    assert r.status_code == 200
    assert r.json()["call"] is False


def test_call_declines_when_ml_has_no_call_net(tmp_path):
    # `medium` is ML but no call net wired -> should_call returns False.
    app = create_app(_make_config_with_calls(tmp_path))
    client = TestClient(app)
    state = deal_initial_state(seed=0)
    ps = _shrink_hand_to(state.private_view(state.public.current_player), 8)
    r = _post_call(client, "medium", "grand", ps)
    assert r.status_code == 200
    assert r.json()["call"] is False


def test_call_grand_rejects_non_eight_card_hand(tmp_path):
    # A freshly dealt hand is 14 cards; grand must be decided on the 8-card hand.
    app = create_app(_make_config_with_calls(tmp_path))
    client = TestClient(app)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)  # 14 cards
    r = _post_call(client, "hard", "grand", ps)
    assert r.status_code == 400
    assert "8-card" in r.json()["detail"]


def test_call_tichu_allows_full_hand(tmp_path):
    # Tichu has no fixed hand size and must stay unguarded by the grand check.
    app = create_app(_make_config_with_calls(tmp_path))
    client = TestClient(app)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)  # 14 cards
    r = _post_call(client, "hard", "tichu", ps)
    assert r.status_code == 200


def test_call_outcome_appears_in_access_log(tmp_path, caplog):
    app = create_app(_make_config_with_calls(tmp_path))
    client = TestClient(app)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    with caplog.at_level(logging.INFO, logger="tichu_inference.access"):
        r = _post_call(client, "hard", "tichu", ps)
    assert r.status_code == 200
    decision = r.json()["call"]
    expected = f"call:tichu={'true' if decision else 'false'}"
    line = next(m for m in caplog.messages if "POST /call" in m)
    # Difficulty leads the suffix; the call outcome still trails the line.
    assert "difficulty=hard" in line, line
    assert line.endswith(expected), line


def test_act_access_log_has_no_call_outcome(tmp_path, caplog):
    # /act must not leak a call= suffix onto its access line.
    app = create_app(_make_config(tmp_path))
    client = TestClient(app)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    with caplog.at_level(logging.INFO, logger="tichu_inference.access"):
        _post_act(client, "easy", ps)
    line = next(m for m in caplog.messages if "POST /act" in m)
    assert "call:" not in line, line


def test_act_access_log_carries_difficulty(tmp_path, caplog):
    app = create_app(_make_config(tmp_path))
    client = TestClient(app)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    with caplog.at_level(logging.INFO, logger="tichu_inference.access"):
        _post_act(client, "easy", ps)
    line = next(m for m in caplog.messages if "POST /act" in m)
    assert "difficulty=easy" in line, line


def test_call_rejects_unknown_kind(tmp_path):
    app = create_app(_make_config_with_calls(tmp_path))
    client = TestClient(app)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    r = _post_call(client, "hard", "super-tichu", ps)
    assert r.status_code == 400


def test_call_rejects_unknown_difficulty(tmp_path):
    app = create_app(_make_config_with_calls(tmp_path))
    client = TestClient(app)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    r = _post_call(client, "godlike", "tichu", ps)
    assert r.status_code == 400


def test_tape_log_writes_decision_blocks_for_ml(tmp_path):
    cfg = _make_config(tmp_path)
    tape = tmp_path / "tape.txt"
    cfg["tape_log"] = str(tape)
    app = create_app(cfg)
    client = TestClient(app)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    r = _post_act(client, "master", ps)
    assert r.status_code == 200
    assert tape.exists()
    text = tape.read_text(encoding="utf-8")
    assert "chose:" in text and "hand:" in text
    assert "difficulty=master" in text and f"top{0}" not in text  # has a topN block


def test_tape_log_skips_baseline_agents(tmp_path):
    cfg = _make_config(tmp_path)
    tape = tmp_path / "tape.txt"
    cfg["tape_log"] = str(tape)
    app = create_app(cfg)
    client = TestClient(app)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    _post_act(client, "easy", ps)   # RuleAgent: no play_action_scores
    text = tape.read_text(encoding="utf-8") if tape.exists() else ""
    assert "chose:" not in text     # header may exist, but no decision block


def test_no_tape_log_by_default_is_harmless(tmp_path):
    app = create_app(_make_config(tmp_path))   # no tape_log key
    client = TestClient(app)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    assert _post_act(client, "master", ps).status_code == 200


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
