---
status: accepted
---

# Concrete Resolver — a bomb-preserving tie-break for intent-level actions

The v1 Action Space is **intent-level** by design ([ADR-0011](0011-bc-training-replay-on-the-fly.md),
`action_space.py`): concrete card-suit picks are excluded to keep the play head at
1,809 slots instead of ~10k. `PlayPair(rank=5, with_phoenix=False)` is one slot
however many concrete pairs of 5s the Hand can build.

The consequence was never designed, only inherited: every concrete realisation of
an Intent shares **one logit**, so the network cannot express a preference among
them, and something else has to choose. Until now that something was
`sorted(..., key=(-logit, i))` in `_rank_legal_by_logits`, where `i` is the
position in `list(legal_actions_for(...))` — and `legal_actions_for` returns a
**frozenset**. The realisation was picked by set-iteration order, which encodes
nothing about the Hand.

[CONTEXT.md](../../CONTEXT.md) §Resolver already flagged this: *"The picks are
heuristic — that heuristic is itself a design surface."* There was no heuristic.
This ADR fills that surface in.

## The defect, concretely

Hand = jade 3-4-5-6-7 + sword 3-4-5-6-7 (two straight-flush bombs). The engine
enumerates 59 concrete legal combinations, collapsing to 28 Intent slots. The
worst collision:

```
slot 1294 = PlayStraight(start_rank=3, length=5, phoenix_position=None)
            -> 32 concrete variants, one shared logit
```

Thirty of the 32 shred both bombs; two spend only their own. The policy has no
way to say which, and the tie-break did not look.

**Two resolvers had the defect, not one.** Serving resolves in
`_rank_legal_by_logits`; the co-train rollout resolves independently in
`ppo/policy.py` via `by_index.setdefault(idx, action)` — also first-one-wins over
a frozenset. Fixing only serving would have trained the policy against a world
that plays the learner's own Hand worse than the exported agent does — the same
train/infer skew shape as
[`team_scores`](../../CONTEXT.md) and the schupfen `current_player` bug.

## Decision

A single pure helper, `tichu_training/concrete_resolver.py`:

```python
resolution_cost_fn(hand) -> cost(action) -> (bombs_lost, flush_edges_lost)
```

Lexicographic, lower is better, computed as before-minus-after on `hand` vs
`hand − cards(action)`. Both resolvers consume it: serving as the middle sort key
`(-logit, cost, i)`, the rollout as a `min(candidates, key=cost)`. Gated by
`MLAgent(bomb_preserving_resolver=True)` so the A/B below is runnable.

It lives in `tichu_training/` rather than `tichu_engine/` because it is a policy
heuristic, not a rule, and `tichu_inference` already imports from
`tichu_training`. This **supersedes CONTEXT.md §Resolver's** claim that the
forward direction lives only in `tichu_inference/ml_agent.py`.

### Two properties that fell out of the design, both load-bearing

**1. The fragment term has to be suit-aware, or it is dead weight.** Variants of
one Intent spend the *same multiset of ranks* and differ only in suits. So any
rank-based structure metric — longest straight remaining, hand fragmentation by
rank — is **constant across the tie group** and can never discriminate. The term
is therefore same-suit adjacency edges lost: a 5-run holds 4 edges, trimming an
end costs 1, splitting the middle costs 2. That also generalises straight-flush
preservation instead of duplicating it.

**2. A four-of-a-kind bomb is not protectable this way.** It occupies all four
suits of a rank; every variant that touches that rank breaks it equally. The
`bombs_lost` term discriminates **only** for straight-flush bombs. It still earns
its place as the lexicographic head — it is a threshold the edge count cannot
express (losing one edge off a 6-run leaves a bomb; losing one off a 5-run does
not) — but the name oversells it.

### Invariant

The resolver **never changes which Intent is played**. It orders strictly within
a tie group; a higher logit always wins first. This is what keeps it free of
train/infer skew in the other direction: BC labels collapse to Intent, so there
is no training distribution over realisations to skew away from.

## Measurement

Subject: the served champion, `cotrain_v6_pbrs_resid_wish_gated_cpfix/export/iter_03328`.
Harness: `scripts/probe_concrete_resolver.py` — resolver ON vs OFF over the
**identical checkpoint**, seat-swapped. A cleaner A/B than the forced-action
probes in `probe_heuristics.py`: the two arms differ in nothing but the
tie-break.

**Trigger rate** (300 deals, 4-seat self-play):

```
play decisions            17,547  (58.5/round; 7,078 = 40.3% forced)
resolver changed the pick     76  (0.25/round = 0.43% of decisions)
  of which bomb saved          1
           run saved          75
```

The headline finding is in that last column: **the bomb term fires once per 300
rounds.** The fragment term does 75 of 76 overrides. The lever is
suit-run preservation, not bomb preservation — the reverse of the framing that
motivated the work.

**EV result** (4,000 Positions, seat-swapped, n=8,000):

```
resolver_on vs resolver_off  mean=+0.73  CI=[-4.27, +5.34]
                             win=49.2% vs 49.1% (tie 1.6%)   -> NEUTRAL
```

**The probe is underpowered by construction, and this was computed before the
run, not after.** CI half-width ±4.8 at 0.25 overrides/round means the eval can
only resolve an override worth **≥19 points**. An override preserves a suit-run
fragment; nothing in the blunder-mining record
([2026-06-11 note](../notes/2026-06-11-blunder-mining-counterfactual-replay.md):
true blunders ≈0.25/round at ~+60 each) suggests a fragment is remotely in that
class. Required n by assumed per-override value:

```
worth 5 pts -> effect 1.25/round -> n ~   57k deals
worth 2 pts -> effect 0.50/round -> n ~  353k deals
worth 1 pt  -> effect 0.25/round -> n ~ 1.4M deals
```

So the null is **a statement about eval resolution, not about the mechanism**,
and a 20k-deal rerun was deliberately not spent: it moves the floor to ~9
pts/override, still an order of magnitude above the plausible effect. The
informative outcome would have been a CI *below* zero — that would have meant the
cost function prefers the wrong variant. It did not occur.

**Default stays ON**, on the same bar `suppress_partner_trick_bomb` shipped on
(logic + no measurable harm, trigger rate below what a tournament CI can
adjudicate), plus one thing that guard did not have: it removes a genuine
train/infer inconsistency, since both resolvers now agree by construction.

## Accepted on correctness, not on EV (2026-08-01)

**This was judged as a defect fix, not as a strength lever, and the null EV result
is therefore not a reason to withhold it.** The pre-change behaviour was not a
considered trade-off that measurement could vindicate or refute — it was
`sorted(key=(-logit, i))` over a frozenset, i.e. *no decision at all*. Where a
free alternative existed, the agent sometimes spent a card out of a straight
flush for nothing. An unforced, avoidable error with a strictly-better
alternative available does not need to clear an EV bar to be worth removing; it
needs only to not make things worse, which the probe establishes as far as it
can resolve.

The EV harness is the right instrument for *"is this a lever?"* and the wrong one
for *"is this arbitrary?"*. The answer to the second question was already yes,
from reading the code.

**Precision on the word "blunder"**: this is an unforced error in the ordinary
sense. It is **not** a Blunder in this project's technical sense — the
counterfactual-replay miner defines one by a Delta Band (≥15 points on paired
replay), and the 76 measured overrides have **never been scored against that
estimator**. Do not cite this ADR as evidence of a blunder-rate reduction.
Establishing that is exactly the decision-level estimator named under
Falsification, and it remains unbuilt.

## What this does NOT claim

- Not a strength lever, and not claimed as one — see the acceptance basis above.
  0.25 overrides/round at a fragment-preservation effect size is nowhere near the
  ~±5 the 4k-deal eval resolves; see
  [ADR-0042](0042-preference-correction.md) for the same arithmetic done on a
  larger pool.
- Not a fix for the intent-level abstraction. Widening the Action Space to carry
  suits (~10k slots) remains rejected — it invalidates every checkpoint and the
  v1 stamp for a defect this cheap to patch downstream.
- `play_action_scores` is **not** resolver-aware, so the MCTS prior in
  `search/engine_world.py` still spreads mass across all realisations of an
  Intent instead of concentrating on the good one. Same defect, third site, left
  open deliberately — nothing currently on the critical path consumes it.

## Rejected alternatives

- **Widen the Action Space to carry suits.** The ~10k blow-up v1 explicitly
  avoided; breaks every checkpoint.
- **Inference-only fix.** Smallest diff, but leaves the rollout resolving
  arbitrarily — the train/infer skew this project has already paid for twice.
- **Put it in `tichu_engine`.** Reusable by RuleAgent, but the engine is
  rules-only; a policy heuristic does not belong there.
- **Broader hand-evaluation score** (weighted fragmentation, playability). More
  upside, but tunable weights and an unmeasured regression surface for a
  tie-break that fires 0.43% of the time.

## Falsification / when to revisit

- **Resolved 2026-08-01, negative**: the probe did not return CI below 0, so the
  cost function is not actively preferring the wrong variant. Had it, the suspect
  was the edge metric mis-valuing end-trims, and the response was to flip the
  default off — not delete.
- **Do not re-run this A/B at larger n hoping for a verdict.** The arithmetic
  above is pre-registered: at 0.25 overrides/round nothing short of ~350k deals
  can adjudicate a 2-pt override. If someone wants this resolved, the route is a
  *decision-level* estimator (paired counterfactual replay on the 76 override
  states, in the style of the blunder miner), not more tournament deals.
- Revisit if the Action Space is ever re-versioned for another reason, at which
  point carrying suits on the high-collision shapes (straights, pair-steps)
  becomes cheap to fold in.
- The resolver is measurement infrastructure as much as a change: the
  `_frequency` audit in the probe script is the reusable part, and it is the
  thing that turned "bomb preservation" from the premise into a refuted premise.
