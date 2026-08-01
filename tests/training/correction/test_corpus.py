"""Correction Corpus assembly (ADR-0042).

The June 2026 corpus screened 9,108 candidates, kept the 663 that passed, and
threw the other 8,445 verified negatives away — so the distiller trained on
positives only and could express nothing but a blanket rule. The corpus builder's
first job is therefore to emit BOTH verdict classes, and its second is a drift set
drawn from decisions the policy actually chooses between.

Deterministic RuleAgents throughout — no torch, no checkpoints.
"""

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_engine.legality import legal_actions_for
from tichu_ml.rule_agent import RuleAgent
from tichu_training.correction.corpus import (
    is_correction,
    needs_reverify,
    verify_and_featurize,
)
from tichu_training.search.blunder_miner import record_round


def _agents():
    return [RuleAgent() for _ in range(4)]


def _position(seed=11):
    return generate_full_position_pool(seed=seed, n=1)[0]


def _candidates(agents, position, *, n=3):
    """Tier-1-shaped candidate dicts for the first `n` decisions that have a
    legal alternative to the chosen action."""
    _, decisions = record_round(agents, position)
    out = []
    for d in decisions:
        legal = list(legal_actions_for(d.state.private_view(d.seat)))
        alt = next((a for a in legal if a != d.chosen), None)
        if alt is None:
            continue
        out.append({"turn": d.turn, "chosen": repr(d.chosen), "alt": repr(alt),
                    "delta": 20.0, "band": "[15,50)"})
        if len(out) == n:
            break
    return out


def test_every_screened_candidate_yields_a_row_labelled_by_its_verdict():
    """The defect that killed the 2026-06-11 corpus. A candidate that fails the
    robustness criterion is a Verified Non-Correction — evidence that the policy
    is right in a state that trips the same trigger — not a row to discard."""
    agents, pos = _agents(), _position()
    cands = _candidates(agents, pos, n=3)
    rows = verify_and_featurize(agents, pos, round_idx=5, candidates=cands, worlds=2)
    assert len(rows) == len(cands)
    assert all(isinstance(r["is_correction"], bool) for r in rows)


def test_drift_rows_skip_decisions_the_policy_never_chose():
    """The other half of the 2026-06-11 defect. Its drift guard was one uniform
    decision per Round, and 39.1% of them were FORCED (n_legal <= 1) — so ~40% of
    the KL budget anchored no-ops while the damage happened on states with real
    alternatives. A drift row is only evidence if the policy had a choice."""
    from tichu_training.correction.corpus import drift_rows

    agents, pos = _agents(), _position()
    _, decisions = record_round(agents, pos)
    forced = [d for d in decisions
              if len(list(legal_actions_for(d.state.private_view(d.seat)))) <= 1]
    assert forced, "fixture must contain forced decisions or it proves nothing"

    rows = drift_rows(decisions, round_idx=5, n=len(decisions))
    assert rows
    assert all(len(r["legal_idx"]) > 1 for r in rows)
    assert {(r["turn"]) for r in rows}.isdisjoint({d.turn for d in forced})


def test_band_candidates_takes_every_decision_in_band_at_its_best_alternative():
    """The corpus wants EXHAUSTIVE band coverage, unlike the stratified rate
    estimator which samples 150/band. Tier 1 emits one row per alternative, so
    band membership must be resolved per Decision at its best alternative —
    otherwise one Decision enters two bands and its verdict is double-counted."""
    import pandas as pd

    from tichu_training.correction.corpus import band_candidates

    tier1 = pd.DataFrame([
        # one decision, two alternatives straddling a band boundary
        dict(round_idx=0, turn=3, delta=180.0, alt="A"),
        dict(round_idx=0, turn=3, delta=40.0, alt="B"),
        dict(round_idx=0, turn=9, delta=60.0, alt="C"),
        dict(round_idx=1, turn=2, delta=8.0, alt="D"),    # below the bands
        dict(round_idx=1, turn=5, delta=900.0, alt="E"),  # above the bands
    ])
    out = band_candidates(tier1)
    assert set(zip(out["round_idx"], out["turn"])) == {(0, 3), (0, 9)}
    row = out[out["turn"] == 3].iloc[0]
    assert row["alt"] == "A" and row["band"] == "[150,400)"
    assert out[out["turn"] == 9].iloc[0]["band"] == "[50,150)"


def test_corpus_rows_feed_the_preference_loss_directly():
    """The corpus and the loss have to agree on the contract, or the mechanism
    test measures a plumbing bug. `chosen_idx` and `alt_idx` must be real Action
    Indices inside the row's legal Intent set, and the batch must load."""
    import torch

    from tichu_training.action_space import ACTION_SPACE_SIZE
    from tichu_training.correction.loss import preference_loss

    agents, pos = _agents(), _position()
    rows = verify_and_featurize(agents, pos, round_idx=5,
                                candidates=_candidates(agents, pos, n=3), worlds=2)
    for r in rows:
        assert r["chosen_idx"] in r["legal_idx"]
        assert r["alt_idx"] in r["legal_idx"]
        assert r["alt_idx"] != r["chosen_idx"]
        assert len(r["features"]) > 0

    loss = preference_loss(
        torch.zeros(len(rows), ACTION_SPACE_SIZE),
        chosen_idx=torch.tensor([r["chosen_idx"] for r in rows]),
        alt_idx=torch.tensor([r["alt_idx"] for r in rows]),
        is_correction=torch.tensor([r["is_correction"] for r in rows]),
        mean_delta=torch.tensor([float(r["mean_delta"]) for r in rows]),
        delta_scale=50.0,
    )
    assert torch.isfinite(loss)


def test_only_variable_survivors_need_a_reverify():
    """65% of survivors show an IDENTICAL delta in all 24 worlds — the alternative
    wins by the same margin however the hidden cards are partitioned, which is a
    near-deterministic claim, not a statistical one. Re-verifying those buys
    literally nothing. Only the ~35% with cross-world spread can move, and they
    are where the 11% 96-world regression lives."""
    invariant = dict(alt_win_rate=1.0, mean_delta=60.0, min_delta=60.0, max_delta=60.0)
    variable = dict(alt_win_rate=0.75, mean_delta=48.0, min_delta=-90.0, max_delta=210.0)
    neutral = dict(alt_win_rate=0.30, mean_delta=-10.0, min_delta=-140.0, max_delta=120.0)

    assert needs_reverify(invariant) is False
    assert needs_reverify(variable) is True
    # non-survivors are not re-verified: this protocol buys down false POSITIVES.
    # A mislabelled negative is inert — satisfied at init, zero gradient — so it
    # is not worth 15,100 extra playout batches.
    assert needs_reverify(neutral) is False


def test_reverify_records_both_measurements_and_relabels_from_the_second():
    """The screen and the re-verify are separate measurements and both stay on the
    row. The label comes from the SECOND one, so a survivor that regresses under
    fresh worlds becomes a Verified Non-Correction — it moves class rather than
    disappearing. That is only possible because the corpus keeps both classes; in
    the 2026-06 positives-only design a regressing survivor simply vanished."""
    agents, pos = _agents(), _position()
    cands = _candidates(agents, pos, n=3)
    rows = verify_and_featurize(agents, pos, round_idx=5, candidates=cands,
                               worlds=2, reverify_worlds=8)
    assert len(rows) == len(cands)
    for r in rows:
        assert r["screen_worlds"] == 2
        assert "screen_win_rate" in r and "screen_mean_delta" in r
        # re-verified iff the screen said it could move; label follows the last word
        moved = r["worlds"] == 8
        assert moved == needs_reverify({
            "alt_win_rate": r["screen_win_rate"], "mean_delta": r["screen_mean_delta"],
            "min_delta": r["screen_min_delta"], "max_delta": r["screen_max_delta"],
        })
        assert r["is_correction"] == is_correction(r)


def test_reverify_is_off_by_default():
    """The screen alone must remain a valid, cheaper protocol — 65% of survivors
    are invariant, so re-verification is a refinement, not a correctness fix."""
    agents, pos = _agents(), _position()
    rows = verify_and_featurize(agents, pos, round_idx=5,
                               candidates=_candidates(agents, pos, n=2), worlds=2)
    assert all(r["worlds"] == 2 for r in rows)
    assert all(r["screen_worlds"] == 2 for r in rows)


def test_progress_stats_counts_reverified_by_comparing_the_two_readings():
    """These four numbers are what a 12-hour run is watched by, so they must not
    depend on the caller remembering its own flags. Counting re-verified rows as
    `worlds == reverify_worlds` breaks the moment the two world counts coincide —
    every row then reads as re-verified. Compare the readings instead."""
    from tichu_training.correction.corpus import progress_stats

    rows = [
        # screened only, not a correction
        dict(worlds=24, screen_worlds=24, is_correction=False),
        # screened only, invariant correction (no re-verify needed)
        dict(worlds=24, screen_worlds=24, is_correction=True),
        # re-verified and held
        dict(worlds=96, screen_worlds=24, is_correction=True),
        # re-verified and regressed -> changed class
        dict(worlds=96, screen_worlds=24, is_correction=False),
    ]
    assert progress_stats(rows) == {
        "verified": 4, "corrections": 2, "reverified": 2, "regressed": 1,
    }

    # the degenerate config: --reverify-worlds equal to --worlds
    same = [dict(worlds=24, screen_worlds=24, is_correction=True)]
    assert progress_stats(same)["reverified"] == 0


def test_candidates_whose_legal_set_escapes_the_action_space_are_dropped():
    """Not every legal Combination has an Action Index — `Single(phoenix,
    as_rank=1.5)` does not — and a row whose legal set cannot be encoded is
    unusable, so the Decision is skipped entirely. Rows are therefore NOT 1:1 with
    candidates. Pinned because two things depend on it: a progress bar driven by
    rows would never reach 100%, and a caller that assumes 1:1 silently under-counts
    its own corpus. Seed 11 round 1 turn 1 is a real instance."""
    agents = _agents()
    pos = generate_full_position_pool(seed=11, n=4)[1]
    _, decisions = record_round(agents, pos)
    d = next(x for x in decisions if x.turn == 1)
    legal = list(legal_actions_for(d.state.private_view(d.seat)))
    alt = next(a for a in legal if a != d.chosen)
    cands = [{"turn": d.turn, "chosen": repr(d.chosen), "alt": repr(alt),
              "delta": 20.0, "band": "[15,50)"}]

    rows = verify_and_featurize(agents, pos, round_idx=1, candidates=cands, worlds=2)
    assert rows == [], "a legal set outside the Action Space must yield no row"
