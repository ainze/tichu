# 2026-06-11 — Blunder-mining via counterfactual replay: the agent's true mistakes are real, rare, and rule-less

## TL;DR

Built a two-tier **blunder-miner** (`tichu_training/search/blunder_miner.py` +
`scripts/mine_blunders.py`) and ran it over 3,000 self-play rounds of the shipped agent
(`cotrain_wish_v5` iter_06225 + partner-trick guard): ~77k play decisions, 308,893
counterfactual alternatives. Findings:

1. **Hindsight gains are almost entirely luck.** 22% of alternatives beat the chosen
   action in the true world; across determinized worlds consistent with the agent's
   observation, 96–99% of those evaporate. The robust criterion's false-positive rate
   is **0.7%** (measured with an EV-neutral control batch).
2. **True blunders exist: ~1 per 50–60 rounds, ~+60 points each, no shared trigger.**
   55 robust blunders surfaced from ~1,500 verified candidates; **51/55 (93%) held up
   in an independent 96-world re-verify** (mean win-rate 0.91, mean delta +62). The
   largest cluster across context keys is n=2. They are scattered judgment misses,
   not a codifiable pattern.
3. **No guardrail to ship.** Even in the most enriched candidate classes
   (follow-with-Single where Pass was better: 6.7% robust; lead-Single where a
   multi-card combo was better: 4.0%), ~95% of trigger states are the agent being
   RIGHT — a blanket rule would re-run the `forced_yield_opp_caller` disaster (−109).
4. **The method itself is the discovery.** Paired evaluation across resampled worlds
   pierces the outcome-variance noise floor that capped PPO (the +8.37 plateau,
   ADR-0034 closure): an effect worth +60 in <1% of states is invisible to
   `R − V(s)` advantages but plainly visible to within-world paired comparison.
   Extrapolating the measured rates: order **+10–15 pts/round** of theoretically
   recoverable EV sits below PPO's noise floor (upper bound; winner's-curse caveats
   below). This is direct evidence for **duplicate-deal / paired-world variance
   reduction as a training signal** — the one mechanism this codebase has never tried.

## Method

Self-play knows the true deal, and the served agents are deterministic (argmax), so a
counterfactual ("what if it had played X here?") is one exact playout, not an estimate.

- **Recorder**: `play_full_round`'s `state_observer` hook captures every Play Decision
  (full `GameState`, chosen action, tichu-ask bookkeeping).
- **Mid-round resume** (`playout_from`): replays from any recorded decision with a
  forced action — mirrors the trusted runner (ADR-0018 tichu-ask reconstruction,
  round-start score differencing) and is **parity-tested from every recorded decision**
  of a round, with and without a mid-round caller.
- **Tier 1 (hindsight)**: force each of the agent's top-4 alternatives in the TRUE
  world; `delta = alt − chosen`, team-relative to the actor. The chosen branch is the
  original round (determinism), so tier 1 costs k playouts per decision.
- **Tier 2 (robustness)**: for candidates, resample K=24 **Determinized Worlds**
  consistent with the actor's observation (`sample_determinized_world`, belief-off)
  and replay chosen-vs-alternative in each. Survives iff the alternative wins ≥70% of
  worlds and the world-mean delta ≥ +15 — "the agent should have known better without
  seeing hidden cards". This kills hindsight bias / strategy fusion.
- **Control**: 150 candidates with |hindsight delta| ≤ 5 ran through the same
  criterion → 1/150 passed = **0.7% false-positive rate**.
- **Re-verify**: every robust survivor re-measured at 96 fresh worlds (new rng salt),
  removing select-on-measurement inflation.

## Results

| Batch (24-world criterion) | n verified | robust | rate |
|---|---|---|---|
| control (|δ| ≤ 5) | 150 | 1 | **0.7%** |
| top-300 by hindsight δ (all +400..+1100 lotteries) | 300 | 6 | 2.0% |
| stratified δ ∈ [15,50) | 150 | 11 | 7.3% |
| stratified δ ∈ [50,150) | 150 | 4 | 2.7% |
| stratified δ ∈ [150,400) | 150 | 2 | 1.3% |
| targeted: follow Single where Pass better | 300 | 20 | **6.7%** |
| targeted: lead Single where combo better | 300 | 12 | **4.0%** |

96-world re-verify of all 55 unique survivors: **51 held** (win96 ≥ 60% & delta96 ≥
+10), mean win-rate **0.91**, mean delta **+62**. Top confirmed cases reach +120…+160
per round with 95–100% world-win-rates. Clusters over (role, phase, caller-context,
chosen-kind, alt-kind): max n=2 — no codifiable trigger.

Headline rates: ~5.3 decisions/round have a hindsight-positive (≥+15) alternative;
~3–7% of those are robust → **~0.2–0.3 true blunders per round**, each worth ~+60 when
it occurs.

## Caveats

- **Policy-relative oracle**: playouts continue with the current policy on all seats.
  Blunders requiring a follow-up the policy would never produce stay invisible; deltas
  are "EV given everyone keeps playing like this".
- **Winner's curse on magnitudes**: candidates were selected on hindsight delta, so
  population-extrapolated EV (+10–15/round) is an upper bound; the 96-world re-verify
  de-inflates per-case magnitudes but not the selection into candidacy.
- **Determinizer observation leak**: resampled worlds ignore what the actor passed at
  Schupfen (3 known cards) — the same simplification PIMC accepted (belief-off).

## Bearing on the plan

- **No inference-time fix follows** — consistent with the exhausted-frontier note
  (2026-06-09): the agent's mistakes are not rule-shaped.
- **The strength-program closure (ADR-0034 addendum) stands** for PPO-shaped training:
  these blunders are exactly what `R − V(s)` cannot see.
- **The one evidence-backed reopening bet is now sharper**: a paired-world /
  duplicate-deal training signal (or a mine→correct→distill loop feeding these
  counterfactual corrections back into the policy at scale). The miner demonstrates
  the variance-reduction mechanism works for *evaluation*; whether it transfers to a
  *training* signal is the open question — and unlike critic depth, trunk size, leash
  tuning, belief inputs, search, and the league, it has never been tested here.

## Artifacts

`data/runs/blunder_mining_v1/`: `tier1_hindsight.parquet` (308,893 rows),
`tier2_verified.parquet`, `tier2_stratified.parquet`, `tier2_targeted.parquet`
(includes the control), `robust_reverified_96.parquet` (the 51 confirmed blunders),
`clusters.csv`, console logs, and the three stage-driver scripts. Rerun:
`py -m scripts.mine_blunders --export-dir <export> --rounds N --out-dir <dir>`.
