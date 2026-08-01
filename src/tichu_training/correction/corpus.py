"""Correction Corpus assembly ([ADR-0042](docs/adr/0042-preference-correction.md)).

Tier 2 already computes a verdict for every candidate it screens. The 2026-06-11
build kept only the survivors, so the distiller saw 663 rows saying "prefer the
alternative here" and nothing saying "and in these 8,445 states that trip the same
trigger, what you already do is correct". Positives-only supervision can express
only a blanket rule — which is what the −18.2/round measured.

So: every screened candidate becomes a row, labelled by its verdict.
"""

import random

from tichu_engine.legality import legal_actions_for
from tichu_training.featurizer import _combination_to_action_index, featurize
from tichu_training.search.blunder_miner import (
    candidate_seed,
    record_round,
    verify_candidate,
)

ROBUST_WIN_RATE = 0.7
ROBUST_MEAN_DELTA = 15.0

#: The three Delta Bands. The `|delta| <= 5` Control Band is a false-positive
#: instrument for the rate estimator, not corpus material, so it is not here.
BANDS = ((15, 50), (50, 150), (150, 400))


def band_candidates(tier1, *, bands=BANDS):
    """Every tier-1 Decision falling in a Delta Band, at its best alternative.

    Exhaustive by design — `mine_blunders_stratified` samples 150 per band because
    it is estimating a *rate*; the corpus wants every row it can verify. Tier 1
    emits one row per alternative, so the dedup to the best alternative per
    `(round_idx, turn)` is what makes band membership a property of the Decision
    (ADR-0042 / CONTEXT.md §Delta Band) rather than of a branch.
    """
    best = tier1.sort_values("delta", ascending=False).drop_duplicates(
        subset=["round_idx", "turn"], keep="first").copy()
    lo, hi = bands[0][0], bands[-1][1]
    out = best[(best["delta"] >= lo) & (best["delta"] < hi)].copy()
    out["band"] = [
        next(f"[{a},{b})" for a, b in bands if a <= d < b) for d in out["delta"]
    ]
    return out


def is_correction(verdict: dict, *, win: float = ROBUST_WIN_RATE,
                  delta: float = ROBUST_MEAN_DELTA) -> bool:
    """The tier-2 criterion: the alternative beat the chosen action across most
    Determinized Worlds by a material margin — "the agent should have known
    better without seeing the hidden cards"."""
    return bool(verdict["alt_win_rate"] >= win and verdict["mean_delta"] >= delta)


def progress_stats(rows) -> dict[str, int]:
    """The four numbers a corpus build is watched by.

    `reverified` compares the two readings on the row rather than testing against
    the caller's `--reverify-worlds`, which would count every row as re-verified
    whenever the screen and re-verify world counts happen to coincide.
    """
    reverified = [r for r in rows if r["worlds"] != r["screen_worlds"]]
    return {
        "verified": len(rows),
        "corrections": sum(bool(r["is_correction"]) for r in rows),
        "reverified": len(reverified),
        "regressed": sum(not r["is_correction"] for r in reverified),
    }


def needs_reverify(verdict: dict) -> bool:
    """Whether this verdict's label could actually change under more worlds.

    Measured on the iter_27008 screen: **65% of survivors show an identical delta
    in all 24 worlds** (69% at win-rate exactly 1.0) — the alternative wins by the
    same margin however the hidden cards are partitioned. That is a
    near-deterministic claim, not an estimate, and no number of extra worlds can
    move it. The other ~35% carry spread up to 500 (median SE at 24 worlds ≈ 11
    against a +15 bar) and are where the measured 11% regression at 96 worlds
    lives — those are worth re-measuring.

    Non-survivors are excluded on purpose. This protocol buys down false
    POSITIVES, which teach the policy a worse action. A mislabelled negative is
    inert: it is satisfied at initialisation, contributes zero gradient, and only
    pushes back if the corrections drag that state — so it is not worth a
    re-verify over ~15,100 rows.
    """
    return is_correction(verdict) and verdict["max_delta"] > verdict["min_delta"]


def _featurized(decision) -> dict | None:
    """The Feature Vector plus the legal Intent set at this Decision. `None` when
    any legal Combination falls outside the Action Space (the row is unusable)."""
    pv = decision.state.private_view(decision.seat)
    legal = list(legal_actions_for(pv))
    legal_idx = [_combination_to_action_index(a) for a in legal]
    if any(i is None for i in legal_idx):
        return None
    return {
        "features": featurize(pv).astype("float32").tolist(),
        "legal_idx": legal_idx,
        "chosen_idx": _combination_to_action_index(decision.chosen),
        "seat": decision.seat,
        "turn": decision.turn,
    }


def drift_rows(decisions, *, round_idx: int, n: int, rng=None) -> list[dict]:
    """A drift-guard sample from Decisions where the policy actually had a choice.

    The 2026-06-11 guard sampled uniformly over every Play Decision, so 39.1% of
    its rows were forced (`n_legal <= 1`) and 52.3% had at most two legal Intents
    — it spent most of its budget anchoring no-ops, while the states the blanket
    rule damaged (median turn 46, 5 legal Intents) went unrepresented. A forced
    Decision cannot drift, so it is not evidence.
    """
    choices = [d for d in decisions
               if len(list(legal_actions_for(d.state.private_view(d.seat)))) > 1]
    if n < len(choices):
        choices = (rng or random.Random(round_idx)).sample(choices, n)
    rows = []
    for d in choices:
        row = _featurized(d)
        if row is not None:
            rows.append({**row, "round_idx": round_idx})
    return rows


def verify_and_featurize(agents, position, *, round_idx: int, candidates,
                         worlds: int, reverify_worlds: int | None = None) -> list[dict]:
    """Replay `position`, verify each candidate across `worlds` Determinized
    Worlds, and emit one featurized row per candidate — **both** verdict classes.

    Rows carry `is_correction` (the label), `mean_delta` (the verified magnitude
    the pairwise margin scales by) and the raw verdict fields for auditing.

    With `reverify_worlds`, any candidate the screen marks as movable
    (`needs_reverify`) is measured again on **fresh** worlds and relabelled from
    that second reading; the screen's numbers are kept as `screen_*`. Both
    readings stay on the row because the label is a claim about the second one
    and the first is the selection that produced it — the June run reported only
    the survivors' post-selection numbers, which is how a 24-world delta ended up
    ~11% inflated.
    """
    _, decisions = record_round(agents, position)
    by_turn = {d.turn: d for d in decisions}
    rows: list[dict] = []
    for cand in candidates:
        d = by_turn.get(int(cand["turn"]))
        if d is None or repr(d.chosen) != cand["chosen"]:
            continue  # replay drifted — impossible with deterministic agents
        alt = next((a for a in legal_actions_for(d.state.private_view(d.seat))
                    if repr(a) == cand["alt"]), None)
        alt_idx = _combination_to_action_index(alt) if alt is not None else None
        row = _featurized(d) if alt is not None else None
        if row is None or alt_idx is None:
            continue
        rng = random.Random(candidate_seed(round_idx, d.turn, cand["alt"]))
        screen = verify_candidate(agents, d, alt, worlds=worlds, rng=rng)
        verdict = screen
        if reverify_worlds and needs_reverify(screen):
            # A DIFFERENT salt, so the second reading draws fresh worlds rather
            # than replaying the first `worlds` of the same sequence — otherwise
            # it is not an independent measurement and cannot de-inflate anything.
            rng2 = random.Random(
                candidate_seed(round_idx, d.turn, cand["alt"] + "|reverify"))
            verdict = verify_candidate(agents, d, alt,
                                       worlds=reverify_worlds, rng=rng2)
        rows.append({
            **row, "round_idx": round_idx, "alt_idx": alt_idx,
            "band": cand.get("band"), "hindsight_delta": float(cand.get("delta", 0.0)),
            "screen_worlds": screen["worlds"],
            "screen_win_rate": screen["alt_win_rate"],
            "screen_mean_delta": screen["mean_delta"],
            "screen_min_delta": screen["min_delta"],
            "screen_max_delta": screen["max_delta"],
            "is_correction": is_correction(verdict), **verdict,
        })
    return rows
