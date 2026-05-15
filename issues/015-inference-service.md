---
id: "015"
title: "Inference service"
type: AFK
blocked_by: ["014"]
stories: [47, 48, 49, 50, 51, 52]
---

## What to build

Implement a stateless HTTP inference service that accepts a serialized `PrivateState` and returns an action, supporting multiple difficulty levels and falling back gracefully on any model failure.

**Single endpoint.** `POST /act` accepts a JSON body containing a serialized `PrivateState` and a `difficulty` field (`easy`, `medium`, `hard`, `master`). Returns a JSON body with the chosen action index and action description. The game server's integration surface is exactly this endpoint — nothing else.

**Difficulty modes.** Four difficulty levels map to distinct loaded checkpoints:
- `easy` — `RuleAgent` (rule-based, no ML, from `#003`)
- `medium` — small BC checkpoint (fewer layers or fewer training steps)
- `hard` — full BC checkpoint
- `master` — BC + AWR refinement checkpoint

All checkpoints are loaded at startup and held in memory for the process lifetime. No runtime reloading.

**Fallback.** If the model produces a malformed output (NaN logits, no legal action with positive probability, exception in the inference path), the service falls back to a uniformly random legal action selected from the `legal_actions_mask` and logs a loud ERROR-level message with the game state and error. The game never stalls.

**Version assertion.** At startup, the service asserts that every loaded checkpoint's `featurizer_version` and `action_space_version` match the running featurizer and action space. Mismatch aborts startup with a clear error message.

**Health and metrics.** `GET /health` returns 200 when all checkpoints are loaded and the service is ready to serve. `GET /metrics` exposes Prometheus-format counters: requests per difficulty level, fallback count, p50/p95/p99 latency.

**Stateless.** No per-game or per-session state is held in the service. Horizontal scaling is achieved by adding replicas behind a load balancer.

**Integration tests.** Use a tiny dummy model (random weights, correct architecture) to test: request/response cycle for all four difficulty levels, action legality (every returned action is legal in the submitted state), fallback path triggered by a deliberately malformed model, and version mismatch surfacing as startup failure rather than silent corruption.

**CLI.** `serve_inference --config <path>` starts the service. Config specifies checkpoint paths per difficulty, port, and worker count.

## Acceptance criteria

- [ ] `POST /act` returns a legal action for all four difficulty levels
- [ ] All checkpoints are loaded at startup; no runtime reloading
- [ ] Fallback to random legal action fires on any model error and logs at ERROR level
- [ ] Version mismatch at startup aborts with a clear error message
- [ ] `GET /health` returns 200 only when all checkpoints are loaded
- [ ] `GET /metrics` exposes per-difficulty request counts, fallback count, and latency percentiles
- [ ] p99 latency < 500ms under single-request sequential benchmark
- [ ] Service is stateless: no in-process game state between requests
- [ ] Integration tests pass: request/response cycle, legality, fallback, version mismatch
- [ ] `serve_inference` CLI starts the service cleanly

## Blocked by

- [#014 Model export (TorchScript / ONNX)](014-model-export.md)
