# ADR-0045: Interrupt Cotrain — Bomb Windows always on, on the v7 trunk

- **Status:** Accepted (design). Not yet built.
- **Date:** 2026-11-09
- **Lifts the descope in** [ADR-0029](0029-ppo-refine-design.md) §"Out-of-turn bombs — on-turn only for v1" and [ADR-0030](0030-phase-2-search-design.md) §"On-turn only". Builds on b821066 (PR #95: `/act` answers honest off-turn polls).
- **Related:** [ADR-0044](0044-featurizer-v7-rich-history-legal-mask-forced-row-drop.md) (the v7 trunk this keeps), [ADR-0040](0040-frozen-reference-vine-pooled-ratchet.md) (read the ship check, not the gate stream), [ADR-0018](0018-tichu-call-featurises-at-first-non-pass-play.md) (Tichu call at first card play); `tichu-nuxt` arena ADR-0011 + `docs/arena-log.md` (the Bomb Window contract adopted here).

## Context

Nothing in PPO, the gate, eval or the website ever let a seat bomb out of turn, although the engine supports it (`legal_bomb_interrupts`, `BombInterrupt`) and BC learnt it from human off-turn rows. A 300-round RuleAgent probe found a legal mid-trick window in 23.3% of rounds (4.2% of steps) and zero bombs played. The website polls bots only while waiting on a human, in fixed seat order, with a spoofed `current_player`; the arena branch of `tichu-nuxt` meanwhile published a precise **Bomb Window** contract. The handoff called the fix "V8".

## Decision

1. **Not a featurizer version.** "V8" is the **Interrupt Cotrain** lineage on the unchanged v7 Feature Vector and trunk — no re-materialise. A Bomb Window shows the trunk a {Pass + bombs} legal mask, and BC already trained the play head on off-turn rows. A "polled off-turn" bit is only worth adding behind a cheap pre-check, never up front.
2. **Protocol = the arena's Bomb Window, verbatim** (see GLOSSARY.md). After every Play — never after a Pass or the Dog, never with a Wish/Dragon pending, never into an empty Trick — each seat that is not out, not the player, not next to act, and holds a beating bomb is asked, in turn order from the player; first bomb wins, windows re-open on the bomb. Training, gate, eval and website play this one game. We first proposed polling after Passes too, with no exclusion of the player; we dropped it because the arena contract already existed and a post-Pass window re-asks a question the seat just declined.
3. **Always on, no switch.** Rollout, gate, `play_round` / tournament all open Bomb Windows unconditionally. There is no interrupts-off setting and no off-vs-on comparison: the old game is retired, not benchmarked.
4. **Trajectories.** A learner-team Bomb Window decision is an ordinary **play** step, in chronological order in `trajs[seat].steps`, same head, same frozen-BC anchor, off-turn legal mask; a per-step window flag exists for logging only. Opponent-team windows are acted on, not recorded. A window bomb that is a seat's first card play triggers its Tichu-call solicitation. Forced windows cannot occur (only seats with a beating bomb are asked, so ≥2 legal actions) and seats without one get no forward pass.
5. **Throughput.** One decision per game per tick, as today: after a Play the game queues its window seats; each tick pops the next into that model's `_PlayGroup`; a bomb clears the queue. No speculative forwards. `rollout_parallel.py` gets the same change with a serial-vs-parallel parity test (deterministic agents).
6. **Warm start and gate.** Warm-start from the v7 gated champion with its frozen-BC anchor and `extra_opponents` (iter-15360, iter-26880, cpfix3328) under the paired/pooled gate. Every opponent answers Bomb Windows from its own play head (cpfix3328 behind the v6-View Prefix) — none is special-cased never to bomb.
7. **Smoke checks before the detached launch (iteration 0):** window rate matches the probe (~23% of rounds / ~4% of steps, within noise); bomb rate on windows > 0 for learner and anchor; window-0 gate margin vs champion has a CI containing 0 (A/A); zero illegal actions, round scores still consistent. Per-iteration dials, logging only: windows per round, bomb rate on windows, partner-trick bomb rate, bomb-over-bomb count.
8. **Ship bar.** 40k-deal paired seat-swap ship check, Bomb Windows on, CI_lo > 0 against **both** the v7 gated champion and the served export; and the website must serve honest Bomb Window polls (separate `tichu-nuxt` task) before the model is served.

## Consequences

- **Old results stop reproducing.** Re-running any pre-0045 config (drift benchmark, `cross_version_tournament`, resuming `cotrain_v7_gated`) on new code plays a different game. Pre-0045 numbers stay valid as history; compare across the boundary only by re-running both sides.
- **λ < 1 would bring back credit dilution.** The recipe runs γ=1, λ=1, so advantage = R − V(s) and extra window steps cost nothing. A run that drops λ must reconsider point 4.
- **Every caller of `legal_actions_for` for a non-current seat** (RuleAgent, RandomAgent, `ppo/policy.py`, `check_cotrain.py`, `correction/corpus.py`, `pmcpa.py`) now sees Pass + bombs; audit them before they are asked in a Bomb Window.
- `eval_matrix.py` and `drift_human.py` currently drop interrupt decisions; they must count them.
