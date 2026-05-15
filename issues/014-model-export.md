---
id: "014"
title: "Model export (TorchScript / ONNX)"
type: AFK
blocked_by: ["009", "010", "012"]
stories: [46]
---

## What to build

Export trained policy checkpoints to TorchScript or ONNX so that the inference service can load them without the full training environment (no PyTorch training dependencies, no training-time data pipeline).

**Export target.** TorchScript is the primary target (simpler for PyTorch-native inference). ONNX is a secondary target if cross-framework serving becomes a requirement. Both formats must be tested.

**What gets exported.** The full inference path: featurize → trunk → all heads (play, pass, wish, dragon) → logit outputs. The Tichu and Grand Tichu call networks are exported separately. The belief model is not exported in phase 1.

**Version pinning in the artifact.** Exported artifacts must embed the `featurizer_version` and `action_space_version` strings so the inference service can assert compatibility at load time without loading the Python checkpoint.

**Latency verification.** After export, run a latency benchmark: 1,000 sequential `act()` calls on a fixed input. Assert p99 latency < 500ms. This is a load-time check, not a load test — horizontal scaling is the solution for throughput, not model optimization.

**Compatibility test.** Export a checkpoint, load it in a clean Python environment (no training deps), call it on a held-out state, and assert the output matches the non-exported model's output on the same input (to floating-point tolerance).

**CLI.** `export_model --checkpoint <path> --format torchscript|onnx --output <path>`.

## Acceptance criteria

- [ ] TorchScript export works for the full policy (trunk + all four heads)
- [ ] Tichu and Grand Tichu call networks are exported separately
- [ ] `featurizer_version` and `action_space_version` are embedded in the exported artifact
- [ ] Compatibility test passes: exported and non-exported models produce the same output on the same input
- [ ] p99 latency < 500ms on a single-device sequential benchmark
- [ ] Export works from a clean environment with only inference deps installed
- [ ] `export_model` CLI is functional

## Blocked by

- [#009 Multi-head BC training pipeline](009-bc-training-pipeline.md)
- [#010 Tichu & Grand Tichu call networks](010-tichu-call-networks.md)
- [#012 Offline RL refinement (AWR)](012-offline-rl-refinement.md)
