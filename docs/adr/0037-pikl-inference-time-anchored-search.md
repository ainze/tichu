---
status: closed
---

# piKL: inference-time KL-regularized better-response on the frozen anchor

> **CLOSED 2026-06-14 — built, run, and falsified into the self-inconsistency
> failure family.** The λ=0.1 probe regressed **−58.66/round** (n=1600); a
> per-decision diagnosis (`scripts/diag_pikl.py`) traced it not to noise or a bug
> but to the **τ-continuation bias**: piKL faithfully maximizes a rollout `Q` that
> assumes the actor and partner revert to τ, but the deployed policy continues as
> piKL — the **same self-inconsistency that closed ADR-0031 / 0035 / 0036**. The
> frozen field stopped co-drift but not this; PIC/more-worlds can't fix a *bias*
> (only variance), and λ→large merely converges piKL → BC. The "residual
> root-future = τ bias" accepted as *bounded* in the design grilling was the killer.
> piKL joins the closed ledger; the strength program re-closes on the
> human-measurement loop. Code (`search/pikl.py`, 14 tests; the launcher; the
> `diag_pikl.py` faithfulness gate) is retained and tested — reusable if a
> consistent-continuation estimator is ever built (that path re-enters the nested
> search ADR-0031 already falsified). Full arc in the Result section below.
>
> **Build status (built test-first):** `search/pikl.py` — torch-free core (`anchored_softmax`,
> `standardize_q`, `pikl_q`), the gate primitive (`intent_rank`), and the
> `piKLAgent` + `build_pikl_agent` integration, and the A/B launcher
> (`run_pikl_ab_shard` + `scripts/run_pikl_ab.py` + `run_pikl_probe.ps1`, reusing the
> pMCPA shard pattern); 14 tests in `tests/training/search/test_pikl.py` (incl. a
> real-iter_06225 end-to-end smoke + shard resumability), 90/90 search suite green, no
> regressions. **Coverage pre-gate PASSED** (Decision E "Result": top-8 100%, k=8
> confirmed); **`q_scale` calibrated to 22** (Decision D). **Only the staged
> probe→decisive detached run remains.** Reopens, *as a probe not a program*, the
> strength line that [ADR-0031](0031-search-and-learning-loop.md),
> [ADR-0035](0035-vine-paired-advantage-ppo.md), and
> [ADR-0036](0036-pmcpa-runtime-policy-adaptation.md) closed. piKL is the **untested
> sibling** named in the pMCPA close and the deep-research run (`handoff-pikl.md`,
> 2026-06-11): inference-time search, **no weight change, frozen field**. This ADR
> records the design as walked one-by-one in the grilling pass of 2026-06-13, which
> corrected several premises of the handoff (the handoff predates the pMCPA kill).

piKL ([Jacob et al., ICML 2022](https://arxiv.org/abs/2112.07544); the Cicero/Diplomacy
recipe) re-weights a frozen behaviour-cloned anchor by an advantage, leashed back toward
the anchor:

    π_i(a) ∝ τ_i(a) · exp( Q_i(a) / λ_i )

`λ→∞` = pure anchor (BC); `λ→0` = unregularized better-response. It converges to NE in
2-player-zero-sum and to **CCE in general/many-player** games — the graceful degradation
that lets it fit 4-player partnership Tichu where the provably-sound CFR/ReBeL/SoG and
the **PIMC** lineage are mathematically restricted to 2p0s.

## Why this is not just another closed lever

The three most recent closed levers died of **one** root cause, stated almost
identically across their ADRs: an improvement signal computed against one continuation,
then a policy that abandons that continuation.

- **Re-anchored vine** ([ADR-0035](0035-vine-paired-advantage-ppo.md)): −23.59 via
  **self-play co-drift** — the estimator rewards moves that beat a *weakening self*; the
  fixed-06225 anchor was *load-bearing safety, not a limitation*.
- **pMCPA** ([ADR-0036](0036-pmcpa-runtime-policy-adaptation.md)): killed even by an
  **oracle on the true world** (−68/round) — the within-deal paired advantage assumes a
  θ_o continuation the adapted θ_a abandons.
- **Search + learning** ([ADR-0031](0031-search-and-learning-loop.md)): the same
  imperfect-info family.

piKL is **structurally outside this family**, and that is the entire reason to reopen:

1. **No co-drift.** The field (partner + all opponents, and the root's own future
   decisions) is the **frozen anchor τ** in the rollout — never the improving policy.
   This *is* the frozen-reference / conservative-policy-iteration successor ADR-0035
   left on the table as principled-but-unbuilt, realized **at inference instead of in
   training**.
2. **No θ_a deviation.** No weights are updated. piKL re-weights one decision's action
   distribution and stops — there is no gradient to abandon its own assumption.
3. **No strategy fusion.** piKL emits a *single* info-set policy marginalized over
   **Determinized Worlds**; it does not take a per-world argmax the way **PIMC** does
   (the defect [ADR-0030](0030-phase-2-search-design.md)/0031 hit).

**Residual, accepted, bounded inconsistency:** the 1-ply `Q(a)` assumes the root's *own*
future decisions continue as τ, but at inference the root plays the piKL-sharpened
π ≠ τ. The λ leash caps how far π strays from τ, and — unlike the three deaths — this is
**non-iterated and non-distilled**. It is the standard one-ply-lookahead approximation
(the Cicero/Suphx regime), tested as a bounded approximation, not assumed away.

## Decisions (walked one-by-one, 2026-06-13)

**A. piKL is a different family from PIMC and outside the closed-lever failure family.**
See above. The framing correction the handoff could not make (it predates the
2026-06-13 pMCPA kill): piKL = ADR-0035's frozen-reference successor, at inference.

**B. `Q` = full deterministic rollout to terminal under the frozen anchor — NOT the
Perfect-Info Critic.** The handoff's "evaluate successor with V" leans on the one value
net the program trusts least: the **Perfect-Info Critic** never cleanly validated
([ADR-0033](0033-perfect-info-critic-escape.md) value-ceiling test came back
*structurally confounded*). Leaning v1 on it repeats the program's recurring trap —
"refuted, but it only tested a crude proxy." piKL needs *correct relative ranking of the
k candidates*, and the project already has an estimator **proven** to have it: the
**blunder-miner**'s paired rollout (`A = R_det(a) − mean over branches`, field = frozen
τ on all four seats, rolled to `_finalise_round`) — luck-cancelled, zero-within-world
variance, 51 true blunders at a 0.7% FP control. It is literally the ADR-0030
**Leaf Rollout** / ADR-0035 vine estimator, and its frozen field is the same property
that keeps piKL out of the failure family. A flat-or-positive result is then attributable
to the *mechanism*, not to suspect Q. The PIC / observable **Value Baseline** as a
rollout-free leaf evaluator is the documented variance/speed **escalation**, gated on a
cheap pre-check: *does it rank the 663 mined corrections the way paired rollout does?*

**C. v1 is Play only; Schupfen is the #1 follow-on, not a co-target.** The handoff's
"Schupfen first, top-k combos" is Play-only vocabulary: **Schupfen** is the standalone
**Schupfen Network** with a *factored* (3-cards × 3-destinations) action space — new
`τ(a)`/top-k machinery — and round-start beliefs are *maximally diffuse* (ADR-0036's
danger-(b)) with the *most expensive* Q (full-round playout per candidate per world). It
has the most new machinery, the noisiest Q, and the costliest Q simultaneously — a poor
*first* integration, even though its leashed-to-BC headroom makes it a high-value
*eventual* target. Play is the mechanism the handoff actually described (BC play-head
top-k Intents — zero new machinery), with tighter mid-round beliefs and cheaper
remaining-round rollouts. Doing both in v1 muddies attribution; lead with Play.
**Corrected gating:** piKL-Schupfen's `Q` evaluates *post-Schupfen* (early-**Play**)
states, so its quality is bounded by **Play** value (~0.45), not the schupfen *head's*
critic (~0.25) — the handoff gated on the wrong number, in Schupfen's favour.

**D. `Q` is standardized before the exponent; the paper's λ grid is in the wrong units
raw.** The paper λ ∈ {0.03, 0.1, 0.3, 1.0} assume a *normalized* `Q`; our `round_outcome`
is in raw Tichu points (±100/±200 on calls/slams), so `exp(Q/λ)` overflows immediately.
**Standardize `Q` per decision** (mean-subtract — free, since `π ∝ τ·exp(Q/λ)` is
invariant to an additive constant — and divide by a fixed point-scale ≈ the std of the
candidates' paired `R_det` spreads), so the paper grid transfers honestly and λ≈0.1 keeps
its "matches BC human-accuracy" meaning. **Calibrated (2026-06-13,
`scripts/pikl_calibrate_qscale.py`, 60 served decisions @ k=8/N=10): median per-decision
`std(Q)` = 21.5 (mean 25.2, right-skewed, max 104) → `q_scale = 22`** — tighter than the
30–60 hand-guess, so the default is set from measurement, not assumption. Keep the
raw-points view as a human sanity check. **Mandatory ±20 clamp on `Q/λ` before `exp`** (vine's iter-169 float32
NaN, ADR-0035).

**E. Coverage pre-gate is a hard precondition; k is empirical.** `support ⊆ τ` is sold as
a free lunch (cost) but is also piKL's hard ceiling: it can only **re-rank what BC
proposes**, never discover what BC misses. Before any loop, histogram the BC head's rank
of the *better* Intent across the 663 mined corrections for k ∈ {4, 8, 16, 32}.
Pre-register **≥~60% in-support at the chosen k → proceed**; mostly out-of-support → piKL
is blind to the located signal → **cheap kill, no tournament**. Set k from the curve; do
not inherit k=8.

**Result (2026-06-13, `scripts/pikl_coverage_gate.py` over `blunder_mining_v1/
corrections.parquet`, τ = iter_06225 @ decile 9): PASS, decisively.** Faithfulness check
100% (the agent's own `chosen_idx` ranks 0 on every row — the forward reproduces the
mining-time anchor). Coverage: **top-4 82.7%, top-8 100%**, max alt-rank = 4. Every mined
correction's better Intent sits at anchor-rank 1–4 (just below the chosen argmax), so the
support⊆τ ceiling is a non-issue and **k=8 fully covers the located signal** — k set from
the data, not inherited. Green light to build the agent.

**F. The decisive experiment, bar, and bands.** Two reads (mirroring pMCPA): the
**mechanism A/B** — piKL on iter_06225's nets vs **un-searched iter_06225** (same τ, with
vs without piKL), seat-swapped — and the **ship bar vs shipped iter_06225, 95% CI > 0**
(ADR-0034/0035 lineage). Effect-size target = the blunder-miner deliverable, **+2–3/round**
(not Suphx's win-rate). Bands:
- **Positive** (CI > 0, ≳ +2): real lever → **route into the owner-vs-agent
  human-measurement loop** (a Tournament A/B is self-relative and cannot confirm the
  superhuman goal — the path chosen at the 2026-06-13 close) → then build Schupfen + DiL.
- **Flat** (CI spans 0): does *not* kill. Exhaust escalations **in order**: finer λ →
  raise k (coverage) → more worlds N → PIC/observable-V leaf eval (after its own ranking
  gate). Only **fully-powered + fully-escalated + still-flat** is the clean kill — and a
  high-value one: it confirms ADR-0031's premise refutation from a *new* angle
  (inference search, not training), leaving blunder-miner/paired-world distillation the
  primary bet.
- **Negative** (clearly worse than un-searched τ): coverage-blind (should've been caught
  by E), Q mis-ranks, or λ too small over-correcting on noisy Q → `diagnose`.

**G. Cost and firing.** **Worlds are SHARED/paired across the k candidates** (common
random numbers): the relative `Q(a) − Q(a')` cancels deal-luck exactly, so **N=20 paired
worlds is adequate** where 20 *independent* would be hopelessly noisy (~12-pt std vs a
30–60-pt signal). This is the blunder-miner trick and is free reuse — and is a correctness
requirement, not a tuning knob. **Fire on every >1-legal-action Play decision; no
BC-uncertainty gate** — blunders can be high-confidence-wrong, and uncertainty-gating
would skip exactly those; verify cheaply against the mined-correction entropy histogram.
Offline-only (latency-bearing experiment, not a served tier in v1); parallelize **across
tournament rounds** on the workers pool, **no nested spawn pool**, torch threads pinned to
1 (pMCPA's wiring lesson). Power-stage: probe (N≈10, n≈8k, CI≈±4) → decisive (N=20,
n≈30–40k, CI≈±2). **k=5 (not 8)** is the operating value — the gate's max alt-rank is 4,
so k=5 covers all 663 corrections at 1.6× the speed.

**Measured cost (2026-06-13, 12-core box, k=5/N=10, threads pinned, 11 workers): ~30
s/position** (rollout-to-terminal Q = ~80 full-round playouts/decision with a torch forward
per ply — pMCPA's cost class). So a ±4 probe on **one** λ (~4000 deals) is **~1.4 days**,
the 4-λ sweep ~1 week. **Chosen path (owner, 2026-06-13): commit the faithful single-point
probe at λ=0.1 first** (the paper's "matches-BC-accuracy" sweet spot — if it doesn't move
there, that is itself strong evidence). The **PIC leaf-eval escalation** (Decision B) is the
pre-registered ~50× speedup to a fast full sweep, deferred unless λ=0.1 looks promising.

**H. τ = iter_06225's play head — the shipped *sharpened* policy, not the raw BC
Checkpoint.** "BC anchor" is ambiguous; the honest answer is the strongest anchor that
never left BC's KL ball. iter_06225 (+8.37 over BC master, sharpened inside a 0.01–0.02
KL ball) gives a stronger field (`Q` measures "good against competent play"), the coverage
prior consistent with the mined corrections (E, which were mined from iter_06225), and the
ship-bar A/B consistency (F). It is still firmly human-plausible, so the convention
argument survives. Pure BC stays available as a convention-purity ablation.

## Result (2026-06-14): λ=0.1 probe regressed −58.66/round — the rollout-Q variance wall

The first probe (λ=0.1, k=5, N=10) read **−58.66/round [−68.29, −47.52]** vs the un-searched
export (n=1600 seat-swap obs, win/loss 0.366/0.617, call-bonus −35.88). Decisive **negative
band**. A per-decision diagnosis (`scripts/diag_pikl.py`, no tournament) isolated the cause:

- **Refuted:** the `asked_tichu` phantom-call faithfulness gap (0% pick-flips on call-live —
  the call net declines on depleted mid-round hands, so no phantom calls fire) and the
  partner-trick-guard asymmetry (0% fire). *(The `asked_tichu` gap is still a latent
  correctness issue worth a defensive fix, but it did not drive this regression.)*
- **NOT a variance problem (the powered probe reversed this).** At n=22 it looked like
  noise (signal≈16 ≈ SE 17); at **n=100** the signal is **25.3 pts > noise 17**, and piKL's
  overrides are robustly **positive by the better N=30 estimate** (+12.7 / +19.8 / +25.7 /
  +53.3 pts at λ 0.1 / 0.3 / 1 / 3). So piKL *successfully maximizes* the rollout `Q`.
- **Root cause — the `Q` is a biased proxy: the τ-continuation self-inconsistency.** The
  rollout scores a candidate by playing the rest of the round with the frozen anchor τ on
  **all four seats — including the actor's own future moves and the partner**. The deployed
  piKL continues as piKL, not τ (it overrides again and again, and so does the piKL partner),
  so each override is justified by a continuation that never happens. The bias compounds with
  override rate → +12.7/override by the τ-proxy but **−58/round** in reality at 45% override.
  **This is the same self-inconsistency that closed [ADR-0031](0031-search-and-learning-loop.md)
  / [ADR-0035](0035-vine-paired-advantage-ppo.md) / [ADR-0036](0036-pmcpa-runtime-policy-adaptation.md)**
  ("the advantage assumes a θ_o continuation the policy abandons"). **piKL does NOT escape the
  failure family** — the "residual root-future = τ bias" accepted as *bounded* in the design
  grilling is the killer, relocated from weight-space to action-space. The frozen field stopped
  co-drift but not this.
- **Two consequences.** (1) **PIC won't fix it** — a critic forward is lower-*variance* but
  still a τ-continuation *estimate*, so it shares the bias; Decision B addresses the wrong
  axis. (2) **λ→large just converges piKL → BC** (override rate 45→31→9→4%), so larger λ is
  *flat-at-best*, not a positive lever — fewer overrides = milder bias, approaching the anchor.

**Lesson:** the coverage gate checked *reachability* but the missing pre-gate was *target
faithfulness* — whether the rollout `Q`'s continuation matches the deployed policy.
`scripts/diag_pikl.py` (the per-decision τ-rollout-vs-reality contradiction) is the cheap
instrument that exposes it without a tournament.

## Strategic placement

- **The product is the online agent first, not a training signal.** A positive piKL ships
  as a **piKL Agent** (a Search-Agent-class candidate / **Difficulty** tier). Distilling
  the search back into the net is a separate, later, *non-assumed* question — distilling
  paired corrections is exactly what vine did (flat, then −23).
- **It converges with the human-measurement loop, doesn't compete.** A Tournament win
  *routes into* the owner-vs-agent loop for the superhuman adjudication.
- **It reopens the closed program as a probe, not a program** — the pMCPA framing.

## Risks

1. **Q ranking, not Q calibration, is the load-bearing property** — addressed by the
   rollout estimator (B) and the coverage gate (E); the PIC escalation is separately
   gated.
2. **Support ⊆ τ ceiling** — the whole point of the E pre-gate; piKL cannot fix what BC
   ranks out of top-k.
3. **Round-start / diffuse-belief noise** — deferred by leading with Play (C); the paired
   worlds (G) is the variance control.
4. **Residual root-future = τ inconsistency** — bounded by λ, non-iterated; measured, not
   assumed (A).
5. **Premise genuinely gone** ([ADR-0031](0031-search-and-learning-loop.md)) — the
   fully-escalated flat (F) is the honest, anti-false-negative way to find that out from a
   new angle.

## Reuse map

| Component | Source | Verdict |
|---|---|---|
| Determinized-World sampler (belief-off/on) | `search/determinize.py` `sample_determinized_world` | reuse as-is (the handoff's "blunder-miner sampler" — there is only this one) |
| Paired rollout / candidates / playout | `search/blunder_miner.py`, `ppo/vine.py` (`playout_from`, `candidate_alternatives`) | reuse; candidates = top-k BC Intents, **worlds shared across candidates** |
| Mined corrections (coverage + Q-rank gates) | `data/runs/cotrain_vine_v1/autopsy_fixrate.py` (663) | reuse as the pre-gate fixtures |
| In-memory net → Agent | `MLAgent.from_loaded` | reuse as-is |
| A/B tournament + seat-swap + CI | `cli/eval_matrix`, vine/pMCPA A/B configs | reuse as-is |
| `piKLAgent` (anchored-softmax over top-k, paired-world Q) | — | **new** |
| Perfect-Info Critic leaf eval | `perfect_info.py` `featurize_perfect_info` | escalation only, behind its own ranking gate |
