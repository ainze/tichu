# 2026-05-29 — AWR game-target full run is flat on play head

## TL;DR

Refining `bc_full_100k_v2/step_e000_b00050000.bin` (84% play-acc reference) with the new `awr.value_target: "game"` pipeline produced **no measurable lift on the play head** over 4 refine epochs. `win_rate_proxy` stayed pinned at ~0.83 across ~290 mid-epoch evals — flat-to-slightly-below the 0.84 BC baseline. Same result as the earlier round-target full-AWR run that motivated this redesign.

This is a negative result on the **hypothesis** that round_outcome's noise was masking real AWR signal. The redesign code is sound; the redesign idea didn't pan out at full scale.

## Context

- Run dir: `C:\workbench\tichu\data\runs\awr_full_100k_v3_game_h512\`
- Config: [configs/awr_full_100k_game_h512.yaml](../../configs/awr_full_100k_game_h512.yaml)
- BC checkpoint refined from: `bc_full_100k_v2/step_e000_b00050000.bin` (~84% play-acc, ~50% wish, ~63% dragon)
- Parquet manifest: `parquet_100k_v3` (93,518 Complete Games out of 99,998 parsed)
- Code: PR #11 (8 commits, branch `claude/objective-volhard-7e07e4`)

The companion h128 run wasn't completed — h512 was launched in parallel as the V-capacity test and produced a decisive enough result that h128 wasn't needed.

## What was tried

Two earlier results set the bar:
- **Round-target full-AWR** (separate run, prior session): flat. No lift over BC baseline.
- **h128 dry run** at smoke scale: V plateaued at running_mse ≈ 0.229 (~8% variance captured of the binary game_won target).

Hypotheses going into this run:
1. **The target was the problem.** `round_outcome` is ±200 with long tails; `game_won` is bounded [0, 1] and lower-variance. A better-conditioned target → cleaner AWR weights → real refine lift.
2. **V might be capacity-bound.** Bumping `baseline_hidden` 128 → 512 to test whether a wider V extracts more game-outcome signal from the BC trunk's 512-dim representation.

Both were testable; both were tested.

## What happened

**V capacity DID help.** h512 fit much better than h128:
- h128 (smoke dry-run): running_mse plateau ~0.229, **~8% variance captured**
- h512 (this run): running_mse plateau ~0.214, **~14% variance captured** (~16% per-chunk)

Verdict: V was capacity-bound at hidden=128 on this 512-dim trunk feature space. The early-stop (patience=20, min_delta=1e-4) latched cleanly around chunk 170 of Phase 2, avoiding the ~8h of wasted iteration the previous run would have done.

**But play-head accuracy didn't move.** Phase 3 results, summarised from `step.csv` plot (4 epochs, 290 mid-epoch evals):

| Head | BC baseline | After AWR refine | Δ |
|---|---|---|---|
| play | ~0.840 | ~0.830 | ~0 (within ~0.5pp noise on 20k held-out) |
| wish | ~0.50 | ~0.55 (loss drifted 0.080 → 0.065) | mild improvement |
| dragon_assignment | ~0.65-0.70 | ~0.65-0.70 | flat |

AWR weights behaved as designed (`avg_weight ≈ 0.044` stable across the run, the expected shape for `beta=1.0` on standardized advantages with `max_weight=20`). The machinery isn't broken — it just isn't pointing at anything that improves the play head.

## What this tells us

1. **The target wasn't the bottleneck.** Going from round (high-variance, biased toward slammy games) → game (low-variance, aligned with the actual objective) + a 2× better-fitting V didn't unlock play-head refinement. The "round_outcome is too noisy" theory was the right diagnosis of *why running_mse was high*, but wrong about *why refine was flat*.

2. **BC at 84% play-acc is at ceiling for this AWR pipeline.** Two different reward signals (round, game), two V capacities (128, 512), zero play-head lift. The constraint is upstream of AWR — either BC's representation can't be improved by reweighting alone, or 84% IS the achievable ceiling on this featurizer + corpus + BC architecture.

3. **Wish head shows the most response to AWR.** Its loss dropped meaningfully (0.080 → 0.065) and accuracy crept up. Probably because wish was further from its ceiling (50% vs 84%) and had room to absorb gradient signal from re-weighted examples. Consistent with "AWR helps where BC hasn't converged," not with "game-target unlocks new signal that round-target couldn't see."

## What this does NOT tell us

- **Whether AWR is the right stage at all.** Maybe AWR-on-this-BC is genuinely the wrong tool. Maybe AWR-on-a-better-BC would work. The 84% ceiling is a strong confounder.
- **Whether a fundamentally different BC arch would unlock AWR.** Wider trunk (2048 × 6) might give AWR a richer representation to nudge.
- **Whether longer BC training closes the gap.** The 50k-batch BC checkpoint might not be fully converged — if it improves on its own, AWR's flatness is a "no headroom" problem, not an AWR-as-method problem.
- **Whether targeted single-head AWR (e.g. wish-only) is worth it.** The wish-head signal here suggests yes, maybe.

## Followups, ranked by expected value

1. **More BC.** Train BC from step 50k → 100k or 150k. Does play-acc still rise? If yes, AWR's flatness is a "no headroom" symptom and the right move is to keep BC going. If no, BC genuinely plateaued at 84% and any further play-head improvement needs a different stage.

2. **Wish-head targeted AWR.** Re-run AWR with `head_weights: {play: 0, wish: 1, dragon_assignment: 0}` to see if isolating wish gives a meaningful boost. Cheap (~12h on this machine post-IPC-fix). Wish was the only head that responded; might be worth pursuing on its own.

3. **Bigger BC architecture.** Train a wider trunk (2048 × 6, head_hidden 512) and see if 87-88% play-acc is reachable. If yes, the 84% ceiling was capacity-bound. If no, it's a corpus/featurizer ceiling and the followup is Belief Module / search.

4. **Park AWR, pivot to Phase 2 directions.** Per [CONTEXT.md](../../CONTEXT.md), Phase 2 was always going to be search + belief. If two AWR experiments at full scale come back flat, that's the signal to start Phase 2 work rather than tune AWR further. AWR's infrastructure stays in the repo for future use; the experiments demonstrate it works correctly even if it didn't pay off on this BC checkpoint.

## What ships in PR #11 anyway

The AWR redesign code is sound and the infrastructure is useful regardless of this experiment's outcome:
- The `game_won` column on the parquet schema is a permanent labelling capability — useful for Belief / search / future RL work, not just AWR.
- The `awr.value_target` config knob is back-compat (defaults to `"round"`) and exposes a real choice to future experimenters.
- Early-stop saves real wall-clock on baseline-fit, regardless of which target the V fits.
- ADR-0013 (parquet schema versioning by directory) is genuinely useful framing.
- The parallel-dataset parity bug caught during dry-run (commit `58b7f77`) was a real latent bug — its fix has nothing to do with AWR target choice.

Merging PR #11 lands all of that as durable repo state. The experiment outcome is recorded here.

## Related artefacts

- PR #11: https://github.com/ainze/tichu/pull/11
- AWR-investigation handoff: `C:\Users\andri\AppData\Local\Temp\handoff-awr-investigation.md` (round-target diagnosis)
- AWR redesign handoff: `C:\Users\andri\AppData\Local\Temp\handoff-awr-game-outcome-redesign.md` (the design that this notes file reports the outcome of)
- IPC bottleneck handoff: `C:\Users\andri\AppData\Local\Temp\handoff-awr-cpu-utilization.md` (orthogonal — the wall-clock optimisation that would make followup experiments much cheaper)
