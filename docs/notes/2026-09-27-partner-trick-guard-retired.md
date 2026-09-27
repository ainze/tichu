# 2026-09-27 — The partner-trick bomb guard is retired

## TL;DR

`MLAgent` no longer overrides a bomb over the partner's winning Trick with a Pass.
On the v7 champion the guard is worth **+0.04/Round [−0.05, +0.16]** (guard off −
guard on): the net almost never wants that bomb, so the guard changed **8 of 2,000
deals**. It was invisible to every strength measure we have, and it split what we
train (the raw net) from what the gate promotes and we serve (net + guard). Serving,
the greedy gate and every `MLAgent` user now play the net's own choice.

## What the guard was

`suppress_partner_trick_bomb` (shipped 2026-06-09, PR #55, ADR-0034): when the
agent's top-ranked action was a Bomb, the partner held the Trick and a Pass was
legal, return Pass — unless the agent was a (Grand-)Tichu caller going out on that
Bomb. Only `MLAgent` applied it; co-train rollouts always sampled the raw net.

It shipped on logic plus one live observation, not on EV: the 2026-06-09 probe
(`forced_keep_partner_trick`) fired 3 times in 200 deals, −0.06 [−4.8, +4.3]
([note](2026-06-09-inference-tuning-frontier-exhausted.md)). That live observation
predates the Nuxt codec fix (trick *starter* sent as `trick.leader`, 47% of live
follow decisions mislabelled), which makes the bot bomb a partner it thinks is an
opponent — and inverts the guard.

## The A/B

`scripts/ab_partner_trick_guard.py`, `cotrain_v7_gated` iter_15360, Skill Decile 9,
first 2,000 deals of `full_position_pool_s0_n20000` with Seat-Swap, paired per deal.
Both arms load the same nets and `MLAgent` is deterministic, so a deal where the
guard never fires scores exactly 0 (smoke-checked).

| | |
|---|---|
| Trigger (guard-off self-play, all 4 seats) | 15 fires in 2,000 deals; partner called 6, self called 4, opponent called 8, nobody out 4; partner's top never a Bomb |
| Deals that diverged | 8 / 2,000 (4 better guard off, 4 better guard on) |
| Per Round, guard off − on | **+0.04 [−0.05, +0.16]** (play-only −0.01 [−0.13, +0.12]) |
| Per diverged deal | +10.6 [−14.4, +35.0] |

Even at the per-diverged-deal CI bound, the population effect is ~±0.15/Round —
below anything the promotion gate or a ship A/B can resolve. A 20,000-deal run would
only narrow the conditional estimate, so it was not run.

Data: `data/runs/ab_partner_trick_guard_v7_pilot/` (`deltas.npz`, `summary.json`).

## What changed

- `MLAgent`: the guard and its `partner_trick_guard` kwarg are gone (also from the
  pikl / pMCPA / vine plumbing that forwarded it).
- The guard lives on only as the `partner_trick_guard` probe in
  `tichu_training.search.heuristic_probes` (with `suppress_partner_trick_bomb`), so
  this A/B and the old vine / piKL diagnostics still reproduce.
- Any gate or tournament number from before this change was measured guard-on; the
  difference is inside noise (above).
