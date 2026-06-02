# ADR-0027: The Docker serving image maps difficulty tiers to skill deciles of a single BC checkpoint

- **Status:** Accepted — amends [ADR-0024](0024-master-tier-conditions-on-top-skill-decile.md) for the containerised serving deployment
- **Date:** 2026-06-02
- **Related:** [ADR-0003](0003-easy-difficulty-is-a-baseline.md), [ADR-0005](0005-inference-time-skill-conditioning.md), [ADR-0024](0024-master-tier-conditions-on-top-skill-decile.md)

## Context

The self-hosted serving Docker image bundles **one** checkpoint and serves all
four difficulties from it. The committed `serve_100k_v5.yaml` instead uses
distinct checkpoints per tier (BC for `medium`/`hard`, AWR-Refined for
`master`). Separately, the serve-side Difficulty Spec builder
(`tichu_inference.app._build_agents`) silently **dropped** `skill_decile`, so
serve-time decile conditioning never took effect despite ADR-0024's claim that
`master` runs at Decile 9.

## Decision

`easy` = `RuleAgent` (unchanged, ADR-0003). `medium / hard / master` = the
**single BC checkpoint** (`bc_full_100k_v5`) conditioned on Skill Deciles
**3 / 6 / 9**. The policy module and the shared standalone nets
(schupfen / tichu_call / grand_call) are **loaded once and shared** across the
three ML tier-views via `MLAgent.from_loaded(...)`, so there is one Model in
memory, not one copy per tier. `skill_decile` is now threaded through
`_build_agents`; eval's path-based `MLAgent(...)` construction is left
unchanged. The decile-mapped spec lives in the committed
`configs/serve_docker.yaml`; container paths are stable (`/app/models/*.pt`) and
the build bakes the selected model dir into `/app/models/`.

## Considered options

- **Hybrid** (BC for `medium`/`hard`, AWR-Refined for `master`@9): keeps
  ADR-0024's measured +10.6 pts/round on `master` but bundles two policies and
  breaks the single-model story. Rejected for single-model simplicity.
- **AWR-only at varying deciles**: AWR refinement likely flattened the model's
  low-decile response, so the lower tiers would not actually play weaker.
  Rejected.

## Consequences

- Container `master` is **BC@9 — measurably weaker** than the AWR-Refined
  `master` (ADR-0024, +10.6 pts/round vs neutral, CI [+3.95, +17.95], n=1000).
  An accepted trade for one Model in the image. The native `serve_100k_v5.yaml`
  deployment is untouched and can still serve the AWR `master`.
- BC was chosen over AWR because BC provably responds across the decile range
  (it was trained on all deciles), giving believable `medium`/`hard`.
- Threading `skill_decile` through `_build_agents` closes a latent gap between
  ADR-0024's stated behaviour and the serve code.
- The bundled model dir must contain `policy.pt` plus the three standalone nets
  (as `bc_full_100k_v5` does). Building from an AWR-only dir, which ships
  `policy.pt` alone, would need the standalone nets supplied separately.
