# ADR-0024: The `master` tier conditions on the top Skill Decile (9), not Neutral

- **Status:** Accepted — supersedes [ADR-0005](0005-inference-time-skill-conditioning.md) for the `master` tier only
- **Date:** 2026-05-31
- **Related:** [ADR-0001](0001-trunk-architecture.md), [ADR-0006](0006-tournament-play-strength-vs-full-strength.md)

## Context

[ADR-0005](0005-inference-time-skill-conditioning.md) fed the **Neutral Skill
Decile** (`10`) to the Skill Embedding at inference for every difficulty, on the
worry that feeding Decile 9 (the strongest BSW players) is a distribution-shift
target — "what a top player would do" in contexts that may not be top-level.
That decision was made without a measurement.

We have since measured it. A play-strength Tournament (homogeneous teams,
seat-swap, the fixed `(seed=0, n=500)` Starting-Position Pool) ran the **same**
exported BC checkpoint conditioned on Decile 9 vs the Neutral Decile:

```
ml_decile9 vs ml_neutral:  mean = +10.62 pts/round   95% CI = [+3.95, +17.95]   n=1000
ml_decile9 vs rule = +132.3   ml_neutral vs rule = +129.1
```

The CI excludes zero: Decile-9 conditioning is a genuine, free improvement on
the identical checkpoint. Critically, Decile 9 won while playing against
Neutral- and rule-level opponents — so ADR-0005's distribution-shift concern
does not bite in the play-strength setting.

## Decision

**The `master` tier conditions on Skill Decile 9.** Encoded as
`skill_decile: 9` on the master agent in `configs/serve_100k_v5.yaml`, threaded
through `MLAgent(skill_decile=...)`. `easy` / `medium` / `hard` keep the Neutral
Decile (ADR-0005 stands for them); `master` is the tier whose explicit job is
maximal strength.

## Consequences

- ADR-0005's calibration rationale may still hold for tiers serving humans of
  *mixed/unknown* skill — hence the change is scoped to `master`, not global.
- The +10.6 figure is at **100k scale on the BC checkpoint**. The Decile-9 slice
  is thin at 100k; re-measure after full-corpus training, where the gain may
  grow (more top-tier data) or shift. The A/B is one command:
  `eval_matrix --config configs/eval_skill_ab_100k_v5.yaml --mode tournament`.
- Measures play strength only; calls/schupfen conditioning is unevaluated.
