# Inference-time tuning frontier is exhausted for cotrain_wish_v5 (iter_06225)

Date: 2026-06-09

## TL;DR

Ran **15 inference-time interventions** against `cotrain_wish_v5 @ iter_06225` — 7
forced-action play probes and 8 call-threshold variants. **All neutral or −EV** vs the bare
agent. The agent is well-calibrated across both play *and* calls; there is no free EV left in
tuning the deployed policy at inference time. Remaining levers are training-side. This note
records the method, results, and the reusable tooling so a future snapshot can be re-checked
in minutes rather than re-derived.

## Method

Every probe wraps the **same** policy and forces a rule only in a precisely-detected trigger
state; a seat-swap full-strength tournament vs the bare policy then isolates the EV of that one
intervention (identical nets on both sides → the delta is purely the nudge). Extends the
`forced_press` premise-test pattern (ADR-0031).

Three search strategies, all plugged into the same tournament harness:

1. **Deep-research heuristics** — five widely-agreed Tichu tactics turned into probes.
2. **Divergence mining** — triangulate `master(BC)` vs `cotrain` vs the recorded `human` move
   on held-out BSW play decisions; the RED bucket (BC matched human, cotrain did not) is the
   high-precision "cotrain left consensus" candidate-mistake set. Cluster, then probe the
   fattest, most codifiable cluster.
3. **Call-threshold sweep** — vary `MLAgent.tichu_threshold` / `grand_threshold` vs the
   argmax-0.5 baseline (a one-line change to adopt, no retraining).

Tooling (all with unit tests):
- `scripts/probe_heuristics.py` — EV runner (probe vs bare subject; `--export-dir` to probe a
  cotrain snapshot rather than the BC baseline).
- `scripts/mine_divergences.py` — per-decision play-divergence miner → ranked cluster table +
  per-decision parquet.
- `scripts/sweep_call_thresholds.py` — call-threshold EV sweep + call-rate context pass.
- `src/tichu_training/search/heuristic_probes.py` — the 5 deep-research probes.
- `src/tichu_training/search/caller_pressure_probes.py` — the 2 divergence-mined probes.

## Results

### 1. Deep-research heuristic probes (n=8000, mean = probe − base, per round)

| Probe | mean | CI | verdict |
|---|---|---|---|
| `forced_follow_low` | **−61.6** | [−66, −57], win 37% | rule badly wrong for this agent |
| `forced_support_tichu` | −3.97 | [−8.8, +0.3] | neutral |
| `forced_split_aces` | −1.22 | [−5.9, +3.2] | neutral |
| `forced_dragon_lastout` | −0.29 | [−5.0, +4.0] | neutral |
| `forced_keep_partner_trick` | −0.06 | [−4.8, +4.3], fires=3 | underpowered (rare) |

`forced_follow_low` is the informative one: the probe only *lowers the height* of a beat the
agent already chose to make, and that costs ~62 pts/round. So the agent's following height is
deliberately calibrated to **secure** the trick — the naive "always follow low" rule is wrong,
and the agent has learned the sophisticated version ("follow with the lowest card that still
wins"). `forced_keep_partner_trick` (don't bomb your own team's winning trick) fires too rarely
(3/200 deals) for the win-rate test to adjudicate; it is logically near-always correct and was
observed live (bomb-over-partner's-bomb), so it is justifiable as a cheap guardrail on
logic+observation, with the carve-out "unless you are a caller going out on that bomb".

### 2. Divergence-mined caller-pressure probes

Mining over 200 held-out games: 111,282 play decisions, cotrain changed 7.2%. RED (3299) ≈
GREEN (2834) — cotrain diverges from BC+human consensus in *both* directions about equally,
the signature of EV refinement, not regression. The top RED signature was opposing-caller
handling: cotrain both cedes where humans contest a caller (`cotrain_cedes/opp_caller`, 142)
and contests where humans pass (`cotrain_contests/opp_caller`, 268).

Built a symmetric pair, both triggered on "following an opponent's trick while an opponent has
called", both vs the bare agent:

| Probe | mean | CI | verdict |
|---|---|---|---|
| `forced_press_opp_caller` (cede→press) | −10.2 | [−14.9, −5.4] | −EV |
| `forced_yield_opp_caller` (contest→pass) | **−108.9** | [−114, −104], win 36% | −EV |

**Both directions lose** → cotrain sits at a local optimum: it beats *both* the human/intuition
"pressure the caller" school and the literature "let them go out, conserve" school by choosing
state-by-state. The 10× asymmetry (over-ceding ≫ over-pressing costlier) means the agent's
aggressive bias is the safe error direction — consistent with `forced_follow_low`. The human
divergences in both directions were the *humans* being suboptimal.

### 3. Call-threshold sweep (n=8000)

All 8 variants neutral (every CI spans 0; every `call_bonus` delta ≤ 0). Call rates move
monotonically and sanely with the threshold (lower → call more, lower success). The call heads
are well-calibrated at argmax. Reconfirms `2026-06-03-call-threshold-no-win.md` on the cotrain
agent.

One observation: baseline **grand success = 0.558, right at the ±200 break-even** (tichu 0.797,
comfortably above). The grand decision operates at its natural margin because the outcome is
intrinsically hard to predict from the decision state — the same irreducible-variance story as
the grand critic R²=0.10. Notably, the under-fit grand critic did **not** produce a fixable
grand-call boundary error.

## On expanding the critic (deeper/wider trunk) — assessed, not pursued

`ValueBaseline` is a 2-layer MLP, a single shared scalar `V(state)` on `round_outcome`, with
`perfect_info: true` (it already sees all four hands). The R² ordering **play 0.45 > schupfen
0.25 > grand 0.10** tracks how much the *state* determines the ±-score outcome — a
state-determinism gradient, not a capacity gradient. Since perfect info is already on and grand
is still 0.10, the deficit is most likely irreducible outcome variance (an 8-card pre-deal hand
weakly determines a ±200 round result), which more parameters cannot fix.

If pursued, the decisive cheap test is an **offline per-decision-type capacity ablation** on the
existing value corpus (`materialised_full_v5/bc`, 1.25B rows): fit small vs deep/wide, report
held-out R² split by decision type — no RL loop. More targeted than raw depth would be
per-decision-type value heads / type-conditioning / loss-reweighting the rare rows. Caveat: a
prior finding already showed value quality was not the strength lever.

## Conclusion

Stop hunting inference-time guardrails on this snapshot — two independent search strategies
(literature + the agent's own divergences from strong humans) plus a direct calibration sweep
all converge on "already calibrated". The tooling here is reusable: re-run it against any future
snapshot to re-check in minutes. The remaining EV is training-side (critic quality — likely
irreducibility-limited; longer/different co-training; more data / league diversity), not in
tuning the deployed policy.
