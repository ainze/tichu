---
name: explore-options
description: Theoretical analysis loop for open design problems. Frame the real objective → generate distinct options → stress each against its load-bearing assumption → converge to a recommendation with the cheapest test that de-risks it. Use when user says "explore options" / "what are my options" / "how should I approach X", is choosing between designs before building, or wants a theoretical (not empirical) analysis of solution approaches.
---

# Explore Options

The theoretical sibling of `/diagnose`. Diagnose finds the cause of a *known* failure by building a feedback loop. This finds the best *approach* to an open problem by reasoning about options before any code is written. Skip phases only when explicitly justified.

When exploring, use the project's domain glossary, read the ADRs in the relevant area, and check memory — many options have already been tried and killed. **Re-proposing a killed idea without addressing why it died is the cardinal sin of this skill.**

## Phase 1 — Frame the real objective

**This is the skill's "wrong bug = wrong fix".** Most bad option-explorations solve the stated problem instead of the real one.

- Restate the problem in one sentence. Separate the **symptom** the user named from the **objective** they actually want.
- Name the **success criterion**: how would we know an option worked? If it isn't measurable or falsifiable, sharpen it until it is.
- List the **hard constraints** (cost, time, compat, irreversibility) and the **non-goals**. An option that violates a hard constraint is dead on arrival — say so early.

Checkpoint the framing with the user before generating options. Cheap, and it reframes half the time.

## Phase 2 — Generate the option space (diverge)

Generate **3–5 genuinely distinct options before analyzing any of them.** Single-option generation anchors on the first plausible idea — the same failure diagnose guards against with ranked hypotheses.

- Make them **orthogonal**, not variations of one idea. If two options share the same load-bearing assumption, they're one option — find a real alternative.
- Always include the **baseline / do-nothing** option and, where relevant, the **cheap-but-ugly** option. They set the bar everything else must beat.
- Pull from prior art: has the codebase, an ADR, or memory already explored this? A killed option still belongs on the list — labelled as killed, with the reason, so it's not silently re-invented.

## Phase 3 — Stress each option (theoretical analysis)

For each option, write down:

1. **Mechanism** — *why* it would work, in one or two sentences. If you can't state the mechanism, it's a vibe, not an option.
2. **Load-bearing assumption** — the single claim the option depends on. This is the analog of a hypothesis's prediction: state the one thing that, if false, sinks it.
3. **Cheapest falsifier** — the smallest experiment, calculation, or lookup that could *kill* the option. Prefer the one you can run in minutes over the one that needs a full build. This is where theoretical analysis earns its keep: cheap kills before expensive commitments.
4. **Failure modes & cost** — how it breaks, what it costs to build/run/maintain, and how reversible it is.

Be adversarial. Try to refute each option, not defend it. The strongest recommendation survives a real attack.

## Phase 4 — Find the deciding crux (converge)

Look across the analyses for the **one question that collapses the space** — the assumption that several options share, or the single unknown that re-ranks everything once resolved. Surface it explicitly.

Often the right move isn't "pick an option" but "run the cheapest experiment that answers the crux, then the choice is obvious." Say that when it's true.

Show the ranked options and the crux to the user before committing — they frequently have domain knowledge that re-ranks instantly. Don't block on it if they're AFK; proceed with your ranking.

## Phase 5 — Recommend

Converge to a **single recommendation** (or a clearly-stated "answer the crux first"):

- The pick, and the crux or trade-off that decides it over the runner-up.
- The **first concrete step** to de-risk it — usually the Phase 3 cheapest falsifier for the chosen option.
- What would change the recommendation (the condition under which the runner-up wins).

Don't hedge into a neutral survey. The user asked for a recommendation; commit to one and own the reasoning.

## Phase 6 — Persist (decide at the end)

Now that the analysis exists, decide whether it's worth keeping:

- If it reached a real decision or killed an option for a stated reason → offer to write it up: an **options memo** (table of options × mechanism/assumption/cheapest-falsifier/verdict + the recommendation) or a **draft ADR** in the project's `docs/adr/` style. Write only on the user's go-ahead.
- If it was a quick exploration that resolved in conversation → don't manufacture an artifact. Say it's done.

When an option was killed here, capture *why* (the assumption that failed and the evidence) so it isn't re-proposed later — that's exactly what the ADRs and memory in this repo are for.
