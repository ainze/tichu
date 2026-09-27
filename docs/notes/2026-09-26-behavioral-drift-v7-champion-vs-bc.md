# 2026-09-26 — Behavioral drift: the v7 champion vs its BC (and vs humans)

**Report:** [report.html](2026-09-26-behavioral-drift-v7-champion-vs-bc/report.html) ·
**Data:** [summary.csv](2026-09-26-behavioral-drift-v7-champion-vs-bc/summary.csv) (tidy panel) ·
[aa_summary.csv](2026-09-26-behavioral-drift-v7-champion-vs-bc/aa_summary.csv) (A/A gate)

## TL;DR

A descriptive map of **how** the shipped champion (`cotrain_v7_gated` iter-15360)
plays differently from the BC it was trained from — not a strength measure and not
a blunder hunt. 28 metrics over the 6 Decisions, 20,000 deals, with human BSW play
as an unpaired reference. The one-line story: **the BC is more passive than
humans; co-training fixed that and overshot — the champion contests Tricks more
than even top-decile players**, while quietly trading Grand-Tichu calls for Tichu
calls at no gain.

## Setup

**Behavioral Drift Benchmark** (CONTEXT.md), Fixed-Opponent design: the Subject team
(Champion+Champion) and the BC team (BC+BC) each play the **same** BC opponents over
the **same** 20,000 Pool deals with Seat-Swap; only the Subject team's seats are
measured, so a Δ is the Subject's own behavior. "The BC" is the champion's own
warm-start stack (`export_rollout_weights --warm-start`), not `_bc_opponent.pt`.
Both greedy, Skill Decile 9, full v7 features. Every metric reports **Exposure**
(how often the situation arises) and **Conditional Rate** (what the agent does in
it), a paired Δ from a deal-cluster bootstrap, and a Benjamini–Hochberg flag across
the whole panel.

Human columns: 12,000 Rounds of all BSW players and 8,000 of top-decile teams (both
partners decile 9), replayed through the engine and logged by the same code as the
agents, sampled across the whole archive. Unpaired: different deals, human
opponents, no Δ, no BH.

Rerun: `py -3.14 -m tichu_training.cli.drift_benchmark --config configs/drift_v7_iter15360.yaml`
(`--aa` for the gate, `--from-log` to re-summarise without replaying).

## Validation — why the numbers can be trusted

- **A/A Null Run passed:** BC vs BC on disjoint deals → 0 / 138 Δs survive BH, 97.1%
  of CIs cover 0, Synthetic-Game win rate 49%.
- **Matches the gate:** champion margin **+14.06/Round [12.54, 15.67]** vs the
  promotion gate's +14.665 against the same BC.
- **The BC reproduces human Round shapes:** Slam in 22.0% of Rounds (humans 23.4%),
  Dog → partner 14.0% (14.3%), Tichu calls per seat-Round 13.1% (13.8%), Rounds per
  Game 9.21 (9.49). The 22% Slam rate looked suspicious; it is human-normal.
- **No engine bug found.**

## Findings

| metric | humans (all) | humans (top 10%) | BC | champion |
|---|---|---|---|---|
| Pass despite a beat, opponent holds Trick | 30.6% | 30.7% | 36.0% | **25.8%** |
| Pass despite a beat, partner holds Trick | 70.1% | 71.7% | 76.3% | 72.9% |
| Caller passes vs an opponent | 17.4% | 18.7% | 19.2% | 16.7% |
| Partner Overtake, partner called | 21.8% | 23.3% | 18.9% | 22.5% |
| Lead a single | 43.2% | 46.8% | 43.7% | **51.2%** |
| Phoenix inside a combination | 61.7% | 53.4% | 58.5% | **40.6%** |
| Phoenix as a single over an Ace | 27.3% | 34.1% | 25.2% | 33.1% |
| Lead the Dog (partner not called) | 41.4% | 41.9% | 42.7% | **54.8%** |
| Grand-Tichu calls | 7.8% | 13.3% | 10.9% | 8.3% |
| Grand with power 2 (of Dragon/Phoenix/Aces in 8) | 22.5% | 40.5% | 46.9% | 32.8% |
| Tichu calls (when asked) | 15.0% | 15.9% | 14.7% | 17.5% |
| Dog → partner in Schupfen | 14.8% | 18.7% | 14.0% | 11.9% |
| Decline to wish | 15.6% | 10.4% | 10.9% | 9.1% |
| Wish a 2 | 22.2% | 21.5% | 32.4% | 29.3% |
| Slam for, per Round | 11.8% | 14.0% | 11.0% | 13.2% |

1. **Contesting Tricks is the largest drift.** Pass rate with a legal beat on an
   opponent's Trick falls 36.0% → 25.8% (−10.2pt, same on Tricks worth ≥10 pts);
   caller passivity 19.2% → 16.7%; Bombs on an opponent's Trick 13.7% → 15.6% when
   legal; a held Bomb goes unplayed less (8.4% → 6.7%). Trick share 54.8%. The BC was
   *more passive than humans*; the champion is now *more aggressive than top humans*.
2. **Partner Overtake rises ~3.5pt** (called partner 18.9% → 22.5%, top humans 23.3%),
   and the situation arises more (Exposure 42.7 → 55.0 per 100 seat-Rounds) because
   the champion's team holds Tricks more. Consistent with the v7 finding that
   overtaking is +EV (forced-yield −11.7/Round).
3. **The Phoenix became a high single:** inside a combination 58.5% → 40.6%, single
   lead 10.4% → 18.0%, single over an Ace 25.2% → 33.1%. Top humans lean the same way
   (53.4% / 34.1%); the champion overshoots them.
4. **Singles and the Dog lead more:** single leads 43.7% → 51.2% (pairs, straights, full
   houses all down); leading the held Dog 42.7% → 54.8% — no human group does this.
5. **Grand traded for Tichu, at no gain.** Grand 10.9% → 8.3%, almost all at power 2
   (46.9% → 32.8%), Grand success unchanged (65.7% vs 65.9%); Tichu up in every power
   stratum (14.7% → 17.5%) with success down (74.3% → 72.1%). Net call bonus **−1.55
   per Round [−2.51, −0.58]**; the whole edge is card play (**+12.77 [11.96, 13.57]**).
   Consistent with calls sitting at a flat EV optimum
   ([2026-07-29](2026-07-29-calls-are-not-a-lever.md)). On Grand the champion moved
   *away* from top humans (13.3%) toward the all-human rate.
6. **Small elsewhere.** Schupfen: Dog to an opponent 27.7% → 30.6%, to partner 14.0% →
   11.9%; both agents never give a King to an opponent and keep the human odd/even
   convention. Wish: fewer declines, fewer 2s (the BC's 32% is far above humans' 22%).
   Dragon: more often to the opponent holding more cards (70.9% → 73.1%).
7. **Outcomes:** Slams for 11.0% → 13.2%, against 11.0% → 9.8%; finishes last 30.5% →
   27.5%; Synthetic-Game win rate **57.9% [55.5, 59.8]**; Rounds per Game unchanged
   (9.21 → 9.29).

All of this is against BC opponents and descriptive, not causal — e.g. "overtaking a
calling partner makes the call fail" cannot be read off the round-level 2×2.

## Bugs found and fixed along the way

- **`BehavioralProfile.slam_rate` read 0 on every run before today.**
  `_finalise_round` resets `out_order` in the step that ends the Round, so telemetry
  dropped the last finisher. Known since
  [2026-06-03](2026-06-03-master-matches-decile9-human-behavior.md), never fixed.
  Training, the promotion gate and the Tournament never read it.
- **Synthetic Games chained both Seat-Swap halves of a deal** — near-mirror images
  (corr −1.0 in the BC arm) — so games ran 1.2 Rounds long and produced a false
  "champion shortens games by 0.77 Rounds". Now one Round per deal.
- **Human replay leaked future Tichu calls:** the replay pre-populates every caller
  from the first card; callers are now revealed in log order.
- **The first human sample read ~0.6% of BSW history** (a stride cut off early); the
  stride is now sized to the Games actually sampled.

## Caveats

- Human columns are unpaired and face human opponents; top-decile teams mostly face
  weaker tables (their +30/Round margin is not comparable to the champion's +14).
  Human rows exclude out-of-turn Bomb interrupts and have no Trick share; humans see
  the game score.
- Synthetic Games are exact only because every current net is score-blind.
- `wish_fulfilled` (~99%) is near-tautological — almost every card of a wished rank is
  played by Round end. Redefine (e.g. "the wish forced a Play") before reading it.
