---
id: "001"
title: "Project scaffolding & CI skeleton"
type: AFK
blocked_by: []
stories: [53, 54, 55]
---

## What to build

Stand up the four-package Python project layout and a working CI pipeline. The goal is a skeleton that every subsequent slice builds inside — no ML code, no game logic, just the load-bearing structure.

Set up the four importable packages with one-way dependencies enforced:

- `tichu_engine` — pure Python, no ML deps
- `tichu_ml` — depends on `tichu_engine` + PyTorch
- `tichu_training` — depends on `tichu_engine` + `tichu_ml`
- `tichu_inference` — depends on `tichu_ml` + a web framework

Provide a single dependency manifest and lockfile (e.g. `pyproject.toml` + `uv.lock` or `requirements.in` + `requirements.txt`) so local dev and CI build identical environments.

Wire up CI (GitHub Actions or equivalent) with three jobs that run on every pull request: rules-engine tests, parser replay-validation on a fixed small subset, and a training smoke test. Jobs should fail fast and report clearly.

Expose a single CLI entry point per task using the names from the PRD: `train_bc`, `train_calls`, `train_belief`, `eval_matrix`, `parse_bsw`, `compute_trueskill`, `serve_inference`. Stubs that raise `NotImplementedError` are fine at this stage — the shape matters.

## Acceptance criteria

- [ ] Four packages importable from a fresh virtual environment built from the lockfile
- [ ] One-way dependency graph enforced (e.g. `tichu_engine` import does not pull in PyTorch)
- [ ] CI runs and all three jobs are green on an empty implementation (stubs pass trivially)
- [ ] Each CLI entry point is invocable (`tichu-train-bc --help` exits 0)
- [ ] `README.md` documents how to set up the environment and run CI locally

## Blocked by

None — can start immediately
