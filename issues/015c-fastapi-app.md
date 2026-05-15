---
id: "015c"
title: "FastAPI app — POST /act, GET /health, GET /metrics + 4 difficulty agents"
type: AFK
parent: "015"
blocked_by: ["015b"]
---

## What to build

`tichu_inference.app.create_app(config)` — builds a FastAPI app with three endpoints and four difficulty-level agents wired in at startup.

- Difficulty mapping (config-driven, agents constructed at startup):
    - `easy` — `RuleAgent` (no ML).
    - `medium` — `MLAgent` loading a small BC artifact.
    - `hard` — `MLAgent` loading the full BC artifact.
    - `master` — `MLAgent` loading the BC+AWR artifact.
- `POST /act` request body: `{private_state: <codec dict>, difficulty: "easy"|"medium"|"hard"|"master"}`. Response: `{action: <codec dict>, action_index: int | null, fallback_used: bool}`.
- `GET /health` returns 200 with `{status: "ok", agents: [names...]}` only if every configured agent loaded successfully; otherwise 503.
- `GET /metrics` Prometheus-style text body: per-difficulty request counter, fallback counter, latency p50/p95/p99 over a rolling window (simple last-N samples).
- Service is stateless: no in-process game state held between requests.

Integration tests (TestClient, in-process):
- Request/response cycle for all four difficulties using small/dummy exported artifacts plus the rule agent.
- Returned action is legal in the submitted state for each difficulty.
- Force a malformed model and assert `fallback_used: true` plus a legal action in the response and an ERROR log.
- Version mismatch surfaces at `create_app` time (raises) rather than at request time.

## Acceptance criteria

- [ ] All three endpoints respond as documented.
- [ ] Returned action is legal under `legal_actions_for(private_state)` for all four difficulties.
- [ ] Fallback triggers and is reflected in the metrics counter + response field.
- [ ] Version mismatch raises at app construction; service does not start.

## Blocked by

- [#015b MLAgent](015b-ml-agent.md)
