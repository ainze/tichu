---
id: "014b"
title: "`export_model` CLI — dispatch BC + Tichu + Grand exports from checkpoints"
type: AFK
parent: "014"
blocked_by: ["014a"]
---

## What to build

A CLI that takes one or more training checkpoints and emits TorchScript artifacts.

- `export_model --checkpoint <bc_ckpt> --format torchscript --output <dir>` writes `policy.pt` (the unified BC trunk + 4 heads).
- `--tichu-checkpoint <ckpt>` writes `tichu_call.pt` if provided; `--grand-checkpoint <ckpt>` writes `grand_tichu_call.pt` if provided.
- For each export the CLI:
    1. Loads the python-side `Checkpoint` (verifies featurizer + action-space versions match the trainer's constants).
    2. Rebuilds the model from `--model-config` (a small YAML with the architecture hyperparameters used at training time — needed because the checkpoint stores weights, not architecture).
    3. Calls `export_torchscript` with the right example inputs.
    4. Prints the output path and the embedded version strings.
- `--format` accepts only `torchscript` at this stage; passing `onnx` exits with a clear "ONNX is a follow-up" message.

## Acceptance criteria

- [ ] CLI smoke: train a tiny BC + tichu + grand from the existing smoke configs, run `export_model` on each, verify the three `.pt` artifacts exist.
- [ ] Each artifact is loadable via `load_exported` and produces matching outputs against the in-memory model.
- [ ] Missing/unrecognized `--format` exits with a non-zero return code and a clear message.
- [ ] No training-only deps are imported during `load_exported` — only `torch` (validated by importing only the inference-side module path).

## Blocked by

- [#014a TorchScript export primitive](014a-torchscript-export.md)
