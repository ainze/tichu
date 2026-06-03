# 2026-06-03 — Diagnosis closed: play-policy undervalues bombs / strong combos → PPO

## TL;DR

The play-strength gap (a strong human beats `master`) has a concrete, quantified,
mechanistically-understood cause: **the play policy systematically undervalues
bombs and strong made-combos.** It manifests as (a) **caller passivity** — a
committed Tichu/GT caller ceding winnable tricks rather than pressing its lead —
and (b) **combo-breaking** — shedding a card that destroys its own bomb. Three
hand-verified examples + a quantified self-play rate confirm it. No offline method
can fix it (bombs are rare in the corpus → BC has no signal to learn their value;
AWR can't reweight signal that isn't there). **Online self-play (PPO) is the right
tool** — it generates bomb/caller situations and learns their value from the
call-bonus-inclusive reward. This closes the diagnostic arc and mandates PPO.

## Evidence

**Hand-flagged (owner review of live decision tapes):**
1. GT caller, opponent leads a single Q; holds Ace + four-3s bomb + phoenix → **PASS**
   (p=0.86; bomb 0.07). Cedes the lead, jeopardising the GT.
2. *Same round*, opponent leads a pair of Aces; same hand (frozen at 12 cards — it
   passed all round) → **PASS** (p=0.77; bomb 0.23). Passing guarantees −200.
3. Leading with four-3s bomb + jade-4 in hand → leads **Single[3]** (p=0.70),
   *destroying its own bomb*, when leading the spare 4 preserves it.

**Quantified (`eval_matrix --mode behavioral`, 2000 self-play seat-rounds):**

| | caller_passivity (any beat) | caller_bomb_passivity (bomb legal) |
|---|---|---|
| **master** | **0.170** | **0.323** |
| neutral | 0.126 | 0.265 |

A committed caller cedes 17% of winnable tricks — **32% when the winning play is a
bomb** (higher = bomb-specific aversion). Decile-9 master is *worse* than neutral on
both: skill-conditioning adds calls (grand 10.6% vs 0.1%) without the aggression to
back them — more calls + more passivity = more self-inflicted −200s.

## Mechanism

Bombs are rare in the **BSW Corpus**, so BC sees almost no examples of how to *value*
them: it treats bomb-component cards as ordinary low cards (sheds/breaks them) and
won't deploy the bomb at the decisive moment. The decile-9 conditioning amplifies the
mismatch. This is the Bayes-ceiling argument made concrete — the modal human line
dominates; the rare high-value exception is averaged away. Imitation can't learn it;
**online self-play can** (it produces the situations and feels their reward).

## Tooling built to reach this (all tests green)

- **Decision tape** — `eval_matrix` offline + **live serve `--tape-log`** capturing
  Play / Wish / Schupfen / Dragon with ranked alternatives + call context. The live
  tape is what let the owner localise these in real games.
- **Caller-passivity probe** — `caller_passivity_rate` + `caller_bomb_passivity_rate`
  in the behavioral telemetry. The built-in dials to watch PPO drive down.
- Tunable call threshold on MLAgent (separate experiment: argmax is already optimal).

## Decision & design implications for PPO

- **Commit to PPO** (warm-started from BC, KL-anchored). Diagnosis-driven design:
  - **Reward = `round_outcome`** (includes the ±100/±200 call bonus) — caller
    passivity is only "wrong" because passing forfeits the bonus, so the reward must
    carry it.
  - **Opponent structure must surface bomb/caller situations** (self-play does, far
    more than the corpus) — frozen-BC start evolving to a small league so opponents
    eventually respect bombs.
  - **Progress metrics**: caller_passivity ↓, caller_bomb_passivity ↓, call-success ↑,
    Tournament Matrix ↑ — verifiable without guessing.
- All cheaper levers are exhausted (BC scale, AWR, belief, call threshold). PPO is the
  surviving lever and now has a concrete target, not "by elimination."
