---
id: "015d"
title: "`serve_inference` CLI — config-driven uvicorn launcher"
type: AFK
parent: "015"
blocked_by: ["015c"]
---

## What to build

- `serve_inference --config <path>` — reads a YAML config (per-difficulty checkpoint paths, port, worker count), builds the FastAPI app via `create_app(config)`, and starts uvicorn.
- The CLI is split as `main(argv)` returning 0 plus a `build_app_for_config(config) -> FastAPI` helper so tests can assert the wiring without binding a port.

The smoke test does NOT start uvicorn — it calls `build_app_for_config` directly with a config pointing at TorchScript artifacts produced in the test, then drives the app with `TestClient`. This matches the test-style chosen for #015c and keeps CI port-collision-free.

## Acceptance criteria

- [ ] `build_app_for_config(config)` returns a FastAPI app with all four difficulty agents loaded.
- [ ] `serve_inference --config <missing>` exits non-zero with a clear error.
- [ ] A smoke test drives the in-process app via `TestClient` and confirms `GET /health` returns 200.
- [ ] No port is bound during the test (uvicorn.run is not invoked).

## Blocked by

- [#015c FastAPI app](015c-fastapi-app.md)
