# Tichu AI — Phase 1 Issues

Generated from [PRD-python-ai-implementation.md](../documentation/PRD-python-ai-implementation.md).

## Dependency order

```
001  Project scaffolding & CI skeleton
└── 002  Rules engine
    ├── 003  Baseline agents + contract test
    │   └── 011  Evaluation harness              ← can run in parallel with 009/010
    ├── 004  BSW parser + replay validation
    │   ├── 005  TrueSkill ratings
    │   └── (feeds into 007)
    └── 006  Featurizer + action space
        └── (feeds into 007)
            007  Parquet training records         ← needs 004, 005, 006
            └── 008  Trunk architecture (HITL)
                ├── 009  BC training pipeline
                │   └── 012  Offline RL (AWR)
                │       └── 014  Model export
                │           └── 015  Inference service
                └── 010  Tichu call networks
                    └── (feeds into 014)
            └── 013  Belief model                ← independent of 008–012
```

## Issue index

| # | Title | Type | Blocked by |
|---|-------|------|------------|
| [001](001-project-scaffolding-ci.md) | Project scaffolding & CI skeleton | AFK | — |
| [002](002-rules-engine.md) | Rules engine | AFK | 001 |
| [003](003-baseline-agents-contract-test.md) | Baseline agents + agent-interface contract test | AFK | 002 |
| [004](004-bsw-parser-replay-validation.md) | BSW parser + replay validation | AFK | 002 |
| [005](005-trueskill-ratings.md) | TrueSkill rating computation | AFK | 004 |
| [006](006-featurizer-action-space.md) | Featurizer + canonical action space | AFK | 002 |
| [007](007-parquet-training-records.md) | Parquet training records pipeline | AFK | 004, 005, 006 |
| [008](008-trunk-architecture-decision.md) | Trunk architecture decision | **HITL** | 007 |
| [009](009-bc-training-pipeline.md) | Multi-head BC training pipeline | AFK | 007, 008 |
| [010](010-tichu-call-networks.md) | Tichu & Grand Tichu call networks | AFK | 007, 008 |
| [011](011-eval-harness.md) | Evaluation harness | AFK | 003 |
| [012](012-offline-rl-refinement.md) | Offline RL refinement (AWR) | AFK | 009, 011 |
| [013](013-belief-model.md) | Belief model training | AFK | 007 |
| [014](014-model-export.md) | Model export (TorchScript / ONNX) | AFK | 009, 010, 012 |
| [015](015-inference-service.md) | Inference service | AFK | 014 |
