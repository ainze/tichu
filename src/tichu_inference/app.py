"""FastAPI app exposing POST /act, GET /health, GET /metrics.

The service is stateless: no per-game state is held between requests.
Agents are constructed at startup from the supplied config and held in
memory for the process lifetime. Version assertions for the ML agents
happen during `MLAgent` construction — if any artifact's version
disagrees with the trainer's pinned versions, `create_app` raises and
the service does not start.

Prometheus-style metrics:
  - `tichu_requests_total{difficulty="..."}`
  - `tichu_fallback_total{difficulty="..."}`
  - `tichu_latency_p50_ms` / `tichu_latency_p95_ms` / `tichu_latency_p99_ms`
    computed over a rolling window of the last `_LATENCY_WINDOW` requests.
"""

import collections
import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from tichu_inference.codec import action_to_json, private_state_from_json
from tichu_inference.ml_agent import MLAgent
from tichu_ml.agent import Agent
from tichu_ml.rule_agent import RuleAgent


log = logging.getLogger(__name__)


_DIFFICULTIES = ("easy", "medium", "hard", "master")
_LATENCY_WINDOW = 200


def create_app(config: dict) -> FastAPI:
    agents = _build_agents(config["agents"])
    missing = [d for d in _DIFFICULTIES if d not in agents]
    if missing:
        raise RuntimeError(f"missing agents for difficulties: {missing}")

    app = FastAPI()
    state = _AppState(agents=agents)
    app.state.agent_registry = agents
    app.state.metrics = state

    @app.get("/health")
    def health():
        return {"status": "ok", "agents": sorted(state.agents.keys())}

    @app.post("/act")
    async def act(request: Request):
        body: dict = await request.json()
        difficulty = body.get("difficulty")
        if difficulty not in _DIFFICULTIES:
            raise HTTPException(status_code=400, detail=f"unknown difficulty: {difficulty!r}")
        ps_blob = body.get("private_state")
        if ps_blob is None:
            raise HTTPException(status_code=400, detail="private_state is required")
        try:
            ps = private_state_from_json(ps_blob)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=f"malformed private_state: {exc}")

        agent = state.agents[difficulty]
        if hasattr(agent, "last_fallback_used"):
            agent.last_fallback_used = False
        t0 = time.perf_counter()
        action = agent.act(ps)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        fallback_used = bool(getattr(agent, "last_fallback_used", False))

        state.record(difficulty, elapsed_ms, fallback_used)

        return {
            "action": action_to_json(action),
            "action_index": None,
            "fallback_used": fallback_used,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        return state.render_prometheus()

    return app


def _build_agents(spec: dict) -> dict[str, Agent]:
    agents: dict[str, Agent] = {}
    for difficulty in _DIFFICULTIES:
        sub = spec.get(difficulty)
        if sub is None:
            continue
        factory = sub["factory"]
        if factory == "rule":
            agents[difficulty] = RuleAgent()
        elif factory == "ml":
            agents[difficulty] = MLAgent(Path(sub["checkpoint"]))
        else:
            raise ValueError(f"unknown agent factory {factory!r} for {difficulty}")
    return agents


class _AppState:
    def __init__(self, *, agents: dict[str, Agent]) -> None:
        self.agents = agents
        self.requests: defaultdict[str, int] = defaultdict(int)
        self.fallbacks: defaultdict[str, int] = defaultdict(int)
        self.latencies: deque[float] = deque(maxlen=_LATENCY_WINDOW)

    def record(self, difficulty: str, elapsed_ms: float, fallback_used: bool) -> None:
        self.requests[difficulty] += 1
        if fallback_used:
            self.fallbacks[difficulty] += 1
        self.latencies.append(elapsed_ms)

    def render_prometheus(self) -> str:
        lines: list[str] = []
        lines.append("# HELP tichu_requests_total Inference requests per difficulty.")
        lines.append("# TYPE tichu_requests_total counter")
        for d in _DIFFICULTIES:
            lines.append(f'tichu_requests_total{{difficulty="{d}"}} {self.requests[d]}')
        lines.append("# HELP tichu_fallback_total Fallback-to-random events per difficulty.")
        lines.append("# TYPE tichu_fallback_total counter")
        for d in _DIFFICULTIES:
            lines.append(f'tichu_fallback_total{{difficulty="{d}"}} {self.fallbacks[d]}')
        arr = np.asarray(self.latencies, dtype=np.float64) if self.latencies else np.zeros(1)
        for label, q in (("p50", 50), ("p95", 95), ("p99", 99)):
            val = float(np.percentile(arr, q)) if self.latencies else 0.0
            lines.append(f"# HELP tichu_latency_{label}_ms Rolling-window inference latency.")
            lines.append(f"# TYPE tichu_latency_{label}_ms gauge")
            lines.append(f"tichu_latency_{label}_ms {val:.3f}")
        return "\n".join(lines) + "\n"


def _registry_for_test(app: FastAPI) -> dict[str, Agent]:
    return app.state.agent_registry  # type: ignore[no-any-return]
