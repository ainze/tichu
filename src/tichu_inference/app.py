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
import json
import logging
import time
from collections import defaultdict, deque
from http import HTTPStatus
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from tichu_inference.codec import action_to_json, private_state_from_json
from tichu_export.torchscript import exported_uses_legal_mask
from tichu_inference.ml_agent import MLAgent, load_policy_module, load_standalone_net
from tichu_ml.agent import Agent
from tichu_ml.rule_agent import RuleAgent
from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.featurizer import FEATURIZER_VERSION


log = logging.getLogger(__name__)
_access_log = logging.getLogger("tichu_inference.access")


_DIFFICULTIES = ("easy", "medium", "hard", "master")
_LATENCY_WINDOW = 200
_GRAND_TICHU_HAND_SIZE = 8  # grand is called on the freshly dealt 8 cards (ADR-0023)


def create_app(config: dict) -> FastAPI:
    agents = _build_agents(config["agents"])
    missing = [d for d in _DIFFICULTIES if d not in agents]
    if missing:
        raise RuntimeError(f"missing agents for difficulties: {missing}")

    # Optional live decision tape: when `tape_log` is set (serve --tape-log),
    # every Play Decision served by an ML agent is appended to that file with its
    # top-k ranked alternatives, so a human playing visually can look up a move
    # they thought was bad. Lazy — costs nothing when the flag is off.
    tape = _TapeLogger(config["tape_log"]) if config.get("tape_log") else None

    # v7 strict mode (ADR-0044). OFF by default: the live client predates the
    # Rich History Block, and the codec's leniency is what keeps it working.
    # That leniency is also the hazard — a missing accumulator decodes to zeros,
    # so an out-of-date client degrades the policy's input silently and no metric
    # moves. Flip this ON (serve --require-v7, or `require_v7: true` in the YAML)
    # the moment the client sends the block, so a half-updated client is an
    # immediate 400 instead of an invisible strength leak.
    require_v7 = bool(config.get("require_v7", False))
    if require_v7:
        log.info("v7 strict mode: /act and /call reject payloads without "
                 "public.rich_history")

    app = FastAPI()
    state = _AppState(agents=agents)
    app.state.agent_registry = agents
    app.state.metrics = state

    @app.middleware("http")
    async def _log_response_time(request: Request, call_next):
        # Emit a uvicorn-style access line WITH the request duration in ms, and
        # surface it as a response header. uvicorn's own access log carries no
        # timing, so serve.py runs it with access_log=False to avoid a duplicate.
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        client = request.client
        addr = f"{client.host}:{client.port}" if client else "-"
        http_version = request.scope.get("http_version", "1.1")
        path = request.url.path
        if request.url.query:
            path = f"{path}?{request.url.query}"
        try:
            phrase = HTTPStatus(response.status_code).phrase
        except ValueError:
            phrase = ""
        # Endpoints may stash a difficulty and/or a short outcome (e.g.
        # "call:tichu=true") on request.state to be appended to the access line;
        # request.state is shared with the route handler via the ASGI scope.
        # Difficulty leads the suffix so a `difficulty=hard` grep works uniformly
        # across /act and /call.
        difficulty = getattr(request.state, "access_difficulty", None)
        outcome = getattr(request.state, "access_outcome", None)
        fields = [f for f in (f"difficulty={difficulty}" if difficulty else None, outcome) if f]
        suffix = f" {' '.join(fields)}" if fields else ""
        _access_log.info(
            '%s - "%s %s HTTP/%s" %d %s %.1fms%s',
            addr, request.method, path, http_version,
            response.status_code, phrase, elapsed_ms, suffix,
        )
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "agents": sorted(state.agents.keys()),
            "featurizer_version": FEATURIZER_VERSION,
            "action_space_version": ACTION_SPACE_VERSION,
            # So a client can negotiate here rather than discover the mismatch
            # as a 400 in the middle of a game.
            "requires_v7": require_v7,
        }

    @app.post("/act")
    async def act(request: Request):
        body: dict = await request.json()
        difficulty = body.get("difficulty")
        if difficulty not in _DIFFICULTIES:
            raise HTTPException(status_code=400, detail=f"unknown difficulty: {difficulty!r}")
        # Stash for the access line once difficulty is known-valid, before any
        # further validation can raise — so malformed-state 400s still show the tier.
        request.state.access_difficulty = difficulty
        ps_blob = body.get("private_state")
        if ps_blob is None:
            raise HTTPException(status_code=400, detail="private_state is required")
        try:
            ps = private_state_from_json(ps_blob, require_v7=require_v7)
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
        # Live tape: every served Decision (Play / Wish / Dragon / Schupfen),
        # dispatched on the pending type by the logger.
        if tape is not None:
            tape.log(agent, ps, action, difficulty, request_body=body)

        return {
            "action": action_to_json(action),
            "action_index": None,
            "fallback_used": fallback_used,
        }

    @app.post("/call")
    async def call(request: Request):
        body: dict = await request.json()
        difficulty = body.get("difficulty")
        if difficulty not in _DIFFICULTIES:
            raise HTTPException(status_code=400, detail=f"unknown difficulty: {difficulty!r}")
        request.state.access_difficulty = difficulty
        kind = body.get("kind")
        if kind not in ("tichu", "grand"):
            raise HTTPException(status_code=400, detail=f"unknown call kind: {kind!r} (expected 'tichu' or 'grand')")
        ps_blob = body.get("private_state")
        if ps_blob is None:
            raise HTTPException(status_code=400, detail="private_state is required")
        try:
            ps = private_state_from_json(ps_blob, require_v7=require_v7)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=f"malformed private_state: {exc}")

        # Grand Tichu is decided on the freshly dealt 8-card hand (ADR-0023), and
        # the grand-call network only ever saw 8-card states in training. Reject
        # any other hand size loudly rather than answer on out-of-distribution
        # input. Tichu has no single fixed hand size, so it is left unguarded.
        if kind == "grand" and len(ps.hand) != _GRAND_TICHU_HAND_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"grand-tichu call requires an {_GRAND_TICHU_HAND_SIZE}-card hand, got {len(ps.hand)}",
            )

        agent = state.agents[difficulty]
        # Baselines (RuleAgent) have no call policy and always decline; only the
        # ML agents expose `should_call`.
        decision = bool(agent.should_call(ps, kind)) if hasattr(agent, "should_call") else False
        request.state.access_outcome = f"call:{kind}={'true' if decision else 'false'}"
        return {"call": decision}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        return state.render_prometheus()

    return app


def _build_agents(spec: dict) -> dict[str, Agent]:
    # Load each unique artifact at most once and share it across every tier that
    # references it. The decile-mapped serving spec (ADR-0027) points all ML
    # tiers at the same policy + standalone nets, differing only by skill_decile,
    # so this keeps one Model in memory instead of one copy per tier.
    agents: dict[str, Agent] = {}
    policy_cache: dict[Path, object] = {}
    net_cache: dict[Path, object] = {}

    def _policy(path: str | Path):
        # v7 (ADR-0044): the mask flag is stamped on the ARTIFACT, and
        # `from_loaded` has no path to read it from. Cache it beside the module
        # so it is read once per artifact and cannot go missing on the tier that
        # happens to be built second.
        p = Path(path)
        if p not in policy_cache:
            policy_cache[p] = (load_policy_module(p), exported_uses_legal_mask(p))
        return policy_cache[p]

    def _net(sub: dict, key: str):
        val = sub.get(key)
        if not val:
            return None
        p = Path(val)
        if p not in net_cache:
            net_cache[p] = load_standalone_net(p)
        return net_cache[p]

    for difficulty in _DIFFICULTIES:
        sub = spec.get(difficulty)
        if sub is None:
            continue
        factory = sub["factory"]
        if factory == "rule":
            agents[difficulty] = RuleAgent()
        elif factory == "ml":
            kwargs: dict[str, Any] = {}
            if "skill_decile" in sub:
                kwargs["skill_decile"] = int(sub["skill_decile"])
            module, uses_mask = _policy(sub["checkpoint"])
            agents[difficulty] = MLAgent.from_loaded(
                module,
                schupfen=_net(sub, "schupfen"),
                tichu_call=_net(sub, "tichu_call"),
                grand_call=_net(sub, "grand_call"),
                uses_legal_mask=uses_mask,
                **kwargs,
            )
        else:
            raise ValueError(f"unknown agent factory {factory!r} for {difficulty}")
    return agents


class _TapeLogger:
    """Appends a human-reviewable decision tape during live serving. One block
    per served Decision (ML agents only): public-state context, the hand, chosen
    action, and top-k ranked alternatives with policy probabilities, plus a
    single-line `replay:` of the verbatim /act request body so a move a human
    flags as a blunder can be re-fed to /act and fixed later. See
    `tichu_eval.decision_tape`. File is opened per-write (append) — simple and
    safe for single-user interactive play; not a high-throughput path."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write("\n# --- live decision tape opened ---\n")

    def log(self, agent: Agent, private_state, action, difficulty: str,
            *, request_body: dict | None = None) -> None:
        # Lazy import keeps the eval dependency off the default serve path.
        from tichu_engine.state import (
            DragonGivePending, MahjongWishPending, SchupfenPending,
        )
        from tichu_eval.decision_tape import (
            build_record, render_dragon, render_record, render_schupfen, render_wish,
        )

        if not hasattr(agent, "play_action_scores"):
            return  # baselines expose no ranked alternatives
        ts = time.strftime("%H:%M:%S")
        header = f"--- #{self._seq + 1}  {ts}  difficulty={difficulty} ---"
        pending = private_state.public.pending_decision
        if pending is None:
            rec = build_record(agent, private_state, action,
                               seat=private_state.public.current_player)
            block = render_record(rec, header=header) if rec is not None else None
        elif isinstance(pending, MahjongWishPending):
            block = render_wish(agent, private_state, action, header=header)
        elif isinstance(pending, DragonGivePending):
            block = render_dragon(agent, private_state, action, header=header)
        elif isinstance(pending, SchupfenPending):
            block = render_schupfen(agent, private_state, action, header=header)
        else:
            block = None
        if block is None:
            return  # forced / no ranked alternatives — don't burn a sequence number
        self._seq += 1
        if request_body is not None:
            # Verbatim, single-line: a flagged blunder is re-feedable to /act as-is,
            # and `grep '^    replay: '` yields a clean JSONL stream of /act bodies.
            block += "\n    replay: " + json.dumps(request_body, separators=(",", ":"))
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(block + "\n\n")
        except Exception:  # noqa: BLE001 — tape logging must never break serving
            log.exception("tape write failed (continuing)")


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
