# 2026-11-14 — Closing the gap to Sean's bot: value net, search and policy fine-tune tests

## Status: all tests NEUTRAL or NEGATIVE. Nothing here moves the champion.

Sean McCarthy's bot (tichuengine.com) beats our served agent (v7 gated iter 15360) by
**+43.9 ± 5.7 a Round**, with out-of-turn bombs allowed. His greedy policy alone (no
search) beats us by **+13.4 ± 6.0**, so his play-time search is worth about +30.

This note records every test we ran to copy the parts of his design we lack, what each
one showed, and what not to try again. All code is test code under
`src/tichu_training/search/` and `scripts/`; nothing is wired into training or serving.

## What Sean does (his site plus what he told us)

- Three nets from self-play only (over 9M games): a belief net, a policy net, and a
  value net that sees all four hands.
- At play time: about 20 sampled worlds, a shallow tree search with sequential halving
  (Gumbel AlphaZero), value net at the leaves, no play-outs.
- **Value net is trained on pure policy games. Policy net is trained on (more
  expensive) search games.**
- **Depth and world count stay low.** No more than 20 worlds is needed, fewer still
  works; depth can be quite low with Gumbel AlphaZero.
- By default his bot plays without out-of-turn bombs; he has separate nets for the
  mode with them.

## Results

Unless noted: Agent under test vs the plain champion iter 15360, seat-swap, paired CI,
Position Pool `full_position_pool_s0_n20000.parquet`.

### 1. Value net that sees all four hands

Trained on the champion's own greedy self-play games (target = Round outcome).

| Training deals | R² (held out) |
|---|---|
| 18k | 0.49 |
| 70k | 0.54 |
| 400k | 0.61 (opening 0.45, mid-Round 0.65) |

- A net that sees only what one seat sees reaches 0.56, so seeing all hands adds little.
- **It cannot rank moves.** Where it says one move is at least 10 points better than
  another, the true difference is −22. Good at "who is winning", useless at "which of
  these two close moves is better".

### 2. Search that overrides the policy with the value net

20 sampled worlds, override when the value net says a move is ≥10 better with paired t ≥ 2.5.

| Variant | Result a Round |
|---|---|
| Depth 0, 70k-deal net (1000 deals) | −116.0 [−123.3, −108.8] |
| Depth 4, 70k-deal net (300 deals) | −70.0 [−81.6, −58.5] |
| Depth 0, 400k-deal net | about −69 |
| **Exact play-out in the true deal instead of the value net** (220 deals) | **+6.7 [−0.1, +13.4]**, overrides 2.5% of Decisions |

- Trusting the value net over the policy is a disaster, with more data too.
- Even a search that cheats (plays each move to the end in the real deal) gains only
  about +7. One-step look-ahead on top of our policy has a low ceiling.

### 3. Value net trained on move differences ("siblings")

At a Decision, play the chosen move and its top alternatives to the end, train the net
on the *difference*. 11k deals: correlation with the true difference 0.295 → 0.32. No
real gain. Stopped after Sean's hint that his value net is trained on pure policy games.

### 4. Gumbel-style rule: policy preference plus value

Pick `argmax log π(a) + meanQ(a) / τ` instead of a hard override. 300 deals each.

| Setting | Result a Round |
|---|---|
| τ = 150, depth 0 | +0.25 [−4.0, +4.4] |
| τ = 50, depth 0 | +0.38 [−6.1, +6.7] |
| τ = 20, depth 0 | +1.43 [−6.6, +9.0] |
| τ = 50, depth 3 | −1.64 [−7.7, +4.3] |

Mixing in the policy's own preference removes the damage. It adds nothing.

### 5. Policy fine-tune on play-to-the-end results ("search distillation")

At 6 non-forced Play Decisions per deal, the chosen move and the top alternatives are
each played to the end in the real deal. The policy is nudged toward the better-scoring
move (loss linear in the scores) with a KL hold to the original. Tested with no search.

| Run | Held-out gain per Decision | vs champion |
|---|---|---|
| Small trial, 32.6k Decisions, lr 3e-4, kl 0.2 | +1.2 on the tuning set, **+0.30 ± 0.22 on fresh data** | **−4.23 [−6.65, −1.78]** (3000 deals) |
| Same, fine-tuned net used only for on-turn Play Decisions | | −4.10 [−6.49, −1.68] |
| Full run, 593,568 Decisions / 100k deals, lr 1e-4, kl 0.2 | +0.03 ± 0.09 | +1.02 [−0.56, +2.62] (5000 deals) |
| Full run, kl 1.0 | −0.02 ± 0.06 | +0.83 [−0.36, +2.00] (5000 deals) |

- The small-trial loss was noise fitting; with 18× the data it is gone.
- **There is nothing to learn from this signal.** On unseen Decisions the fine-tuned
  policy does no better than the original. It changes its top move in 1–3% of
  Decisions, never to a move that was not scored.
- Not tested: a much longer fine-tune (the full run was 3 epochs). The held-out gain
  was flat across epochs.

### 6. Issue #110: the bomb moments the Bomb Window contract leaves out

One team may bomb in the extra moments, the other always declines. Same champion on
both sides, 5000 deals each.

| Moment | Result a Round | Bombs taken / 1000 Rounds | Deals changed |
|---|---|---|---|
| After a Pass | −0.87 [−1.52, −0.22] | 63 | 3.6% |
| Empty Trick | −0.18 [−0.50, +0.13] | 14 | 0.8% |

Human (BSW) games, 4000 games / 40,142 Rounds: 428 on-turn bombs and **19 out-of-turn
bombs per 1000 Rounds**. The split by moment is not reliable (BSW writes Passes
loosely), the total is.

**Reading:** #110 is not a real part of the gap. Recommend closing it as a deliberate
choice in ADR-0045. Not decided yet.

### 7. Rule bugs fixed along the way (PR #111, merged)

- #108 Round left unscored when the third player goes out with a Mahjong play:
  3.75 per 1000 champion Rounds, about 198 points wrong each time (under 1 a Round).
- #109 Out-of-turn bomb did not clear a fulfilled wish: 0 in 4000 Rounds.

## Lessons

1. **An outcome-trained value net does not rank moves, with more data or all four
   hands.** R² went 0.49 → 0.61 and move ranking stayed wrong-signed. Do not build a
   search that lets such a net overrule the policy.
2. **One-step improvement on top of the champion is worth close to zero.** Cheating
   exact search: +7. Fine-tune on 600k exact comparisons: +1, inside noise. The
   champion's near-top moves are close in value under its own continuation. This is the
   same wall as piKL / pMCPA / vine, now measured without their bias.
3. **Offline gain on the set you tuned on is inflated.** +1.2 became +0.3 on fresh
   Decisions and −4 in play. Always score on data the settings never saw, and always
   finish with a seat-swap test.
4. **Small-data policy fine-tunes do real damage** (−4 from 32k Decisions). The loss
   was in the trained Decisions themselves, not a side effect on Wish / Dragon / windows.
5. **We guessed Sean's recipe wrong twice.** "Value net learns from tried-out moves"
   and "it works because it looks far ahead" were both refuted by his hints. Ask before
   building on a guess about someone else's system.
6. **What we have not copied** is the loop itself: search targets from ~20 sampled
   worlds with a value net, fed back into the policy over millions of games from the
   start, with a belief net. Polishing a finished policy with one play-out per move is
   not that.
7. **Out-of-turn bombs are a small thing**: about 2% of human Rounds, and the uncovered
   moments change about 4% of deals.
8. **`run_export_ab.py`'s `identical_deals` field is wrong** (it tests both deltas for
   zero; identical play gives deltas that cancel, not zeros). Ignore it;
   `probe_bomb_moments.py` reports the correct `unchanged_deals`.

## Open

- The Interrupt Cotrain run (`cotrain_v7_interrupt`, from champion iter 34304) ran one
  night (1300 iterations) and sat at −2.4 [−3.4, −1.4] vs its own starting champion,
  −0.9 [−2.0, +0.3] vs iter 15360. Its first gate window was already −4.6, which a copy
  should not be. Check that before continuing it.
- All tests above used iter 15360 (the served agent that played Sean), not the newer
  iter 34304.
- Sean's actual loop (point 6) is untested here and is a large build.

## Where things are

Code (test builds):

- `src/tichu_training/search/full_state_value.py` — resume a Round from any state,
  emit value rows, the full-state value net.
- `src/tichu_training/search/value_search.py` — the override and Gumbel-style search agents.
- `src/tichu_training/search/search_distill.py` — play-to-the-end move comparisons,
  on-turn-only hybrid agent.
- `src/tichu_training/search/bomb_moments.py` — the #110 probe.
- `scripts/emit_full_state_value.py`, `train_full_state_value.py`,
  `emit_value_siblings.py`, `train_value_siblings.py`, `run_value_search_ab.py`,
  `emit_search_distill.py`, `train_search_distill.py`, `run_export_ab.py`,
  `probe_bomb_moments.py`, `count_human_bomb_moments.py`.
- `tests/training/search/test_value_search.py` — 9 tests.

Data (not in git), under `C:\workbench\tichu\data\runs\`: `full_state_value_v1`,
`value_siblings_v1`, `value_search_ab`, `search_distill_v1` (400 shards, `full_kl0.2`,
`full_kl1.0`), `bomb_moments_probe`. Emit seeds: value 10,000,000+, siblings
20,000,000+, distill 30,000,000+ (all disjoint from the eval pool).
