---
id: "014a"
title: "TorchScript export primitive with embedded version metadata"
type: AFK
parent: "014"
blocked_by: []
---

## What to build

A small TorchScript export primitive used by BC policy and the call networks.

- `export_torchscript(module, *, example_features, example_skill_decile, featurizer_version, action_space_version, output_path)` — traces the module with the example inputs and saves via `torch.jit.save` with the version strings stamped into `_extra_files`.
- `load_exported(path, *, expected_featurizer_version, expected_action_space_version) -> torch.jit.ScriptModule` — `torch.jit.load(path)` plus a `VersionMismatchError` raised if the embedded version strings disagree with the loader expectations (same contract as the python-side `Checkpoint`).
- The traced module must produce numerically equal outputs to the eager module on the same input (to single-precision FP tolerance).

`example_skill_decile` is included so the BC policy's `forward(features, skill_decile)` signature traces cleanly. Call-network exports that don't use skill can pass `None` and the primitive skips that input.

## Acceptance criteria

- [ ] BCModel traces, saves, and loads round-trippably with `featurizer_version` / `action_space_version` extra-files.
- [ ] Loaded module called on a fresh input produces outputs equal to the eager model's outputs (`torch.allclose` with `atol=1e-5`).
- [ ] `load_exported` raises `VersionMismatchError` on featurizer or action-space version drift.
- [ ] Empty `action_space_version` is allowed (so this primitive can be reused by belief / call-network exports that don't use the canonical action space).

## Blocked by

- None — can start immediately.
