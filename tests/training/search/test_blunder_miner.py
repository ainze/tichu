"""Blunder-miner mechanics: the mid-round resume must be a faithful continuation of
the trusted runner (parity from EVERY recorded decision), candidates must come from
the legal set, and tier-2 worlds must keep the actor's observation fixed. All with
deterministic RuleAgents — no torch, no checkpoints.
"""

import os
import random
import subprocess
import sys

from tichu_engine.legality import Pass, legal_actions_for
from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_eval.play_full import play_full_round
from tichu_ml.rule_agent import RuleAgent
from tichu_training.search.blunder_miner import (
    candidate_alternatives,
    candidate_seed,
    decision_context,
    mine_round,
    playout_from,
    record_round,
    team_relative,
    verify_candidate,
)


def _seed_in_subprocess(hash_seed: str, key=(7, 13, "Pass()")) -> int:
    """Compute candidate_seed in a FRESH interpreter under an explicit
    PYTHONHASHSEED. Same-process equality proves nothing here — builtin hash() of a
    str is salted per interpreter, so a salted seed looks perfectly deterministic
    until it crosses a process boundary (and mp 'spawn' crosses one per worker)."""
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    src = str((__file__ + "/../../../../src").replace("\\", "/"))
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c",
         "from tichu_training.search.blunder_miner import candidate_seed;"
         f"print(candidate_seed(*{key!r}))"],
        capture_output=True, text=True, env=env, check=True,
    )
    return int(out.stdout.strip())


class _TichuCallingRule(RuleAgent):
    """RuleAgent that always calls tichu (never grand) — exercises the caller
    bookkeeping that the mid-round resume must reproduce."""

    def should_call(self, private_state, kind: str) -> bool:
        return kind == "tichu"


def _agents(calling_seat=None):
    return [
        _TichuCallingRule() if s == calling_seat else RuleAgent() for s in range(4)
    ]


def _position(seed=11):
    return generate_full_position_pool(seed=seed, n=1)[0]


def test_record_round_captures_play_decisions_deterministically():
    pos = _position()
    result_a, decisions_a = record_round(_agents(), pos)
    result_b, decisions_b = record_round(_agents(), pos)
    assert result_a.total == result_b.total
    assert len(decisions_a) == len(decisions_b) > 0
    for d in decisions_a:
        assert d.state.public.pending_decision is None
        assert d.chosen in legal_actions_for(d.state.private_view(d.seat))
    assert [d.turn for d in decisions_a] == list(range(len(decisions_a)))


def test_playout_from_reproduces_the_original_outcome_from_every_decision():
    # The hard invariant: with deterministic agents, resuming at ANY recorded
    # decision with the chosen action forced must land on the original totals —
    # including the tichu-ask bookkeeping (a seat that calls mid-round).
    for calling_seat in (None, 2):
        agents = _agents(calling_seat)
        pos = _position()
        result, decisions = record_round(agents, pos)
        for d in decisions:
            totals = playout_from(
                agents, d.state, forced_action=d.chosen, asked_tichu=d.asked_tichu,
                initial_scores=d.initial_scores,
            )
            assert totals == result.total, (
                f"parity broke at turn {d.turn} (caller={calling_seat})"
            )


def test_playout_from_with_an_alternative_completes():
    agents = _agents()
    pos = _position()
    _, decisions = record_round(agents, pos)
    d = next(dec for dec in decisions
             if len(legal_actions_for(dec.state.private_view(dec.seat))) > 1)
    alt = next(a for a in legal_actions_for(d.state.private_view(d.seat))
               if a != d.chosen)
    totals = playout_from(agents, d.state, forced_action=alt,
                          asked_tichu=d.asked_tichu, initial_scores=d.initial_scores)
    assert isinstance(totals, tuple) and len(totals) == 2
    assert all(isinstance(t, int) for t in totals)


def test_team_relative_is_actor_team_minus_opponents():
    assert team_relative((30, -10), seat=0) == 40
    assert team_relative((30, -10), seat=1) == -40
    assert team_relative((30, -10), seat=2) == 40


def test_candidate_alternatives_excludes_chosen_and_caps():
    agents = _agents()
    pos = _position()
    _, decisions = record_round(agents, pos)
    d = next(dec for dec in decisions
             if len(legal_actions_for(dec.state.private_view(dec.seat))) > 3)
    legal = legal_actions_for(d.state.private_view(d.seat))
    alts = candidate_alternatives(agents[d.seat], d.state.private_view(d.seat),
                                  d.chosen, top_k=2)
    assert d.chosen not in alts
    assert len(alts) == 2
    assert all(a in legal for a in alts)


def test_mine_round_emits_one_row_per_alternative():
    agents = _agents()
    rows = mine_round(agents, _position(), round_idx=7, top_k=2)
    assert len(rows) > 0
    for row in rows:
        assert row["round_idx"] == 7
        assert isinstance(row["delta"], float)
        assert row["alt"] != row["chosen"]
        assert {"seat", "turn", "n_legal", "chosen_kind", "alt_kind"} <= set(row)
    # Decisions with several legal actions contribute multiple alternatives.
    by_turn: dict = {}
    for row in rows:
        by_turn.setdefault(row["turn"], 0)
        by_turn[row["turn"]] += 1
    assert max(by_turn.values()) == 2


def test_verify_candidate_reports_worlds_and_win_rate():
    agents = _agents()
    pos = _position()
    _, decisions = record_round(agents, pos)
    d = next(dec for dec in decisions
             if len(legal_actions_for(dec.state.private_view(dec.seat))) > 1)
    alt = next(a for a in legal_actions_for(d.state.private_view(d.seat))
               if a != d.chosen)
    out = verify_candidate(agents, d, alt, worlds=3, rng=random.Random(0))
    assert out["worlds"] == 3
    assert 0.0 <= out["alt_win_rate"] <= 1.0
    assert isinstance(out["mean_delta"], float)


def test_decision_context_yields_clusterable_keys():
    agents = _agents(calling_seat=2)
    pos = _position()
    _, decisions = record_round(agents, pos)
    ctx = decision_context(decisions[0].state, decisions[0].seat,
                           decisions[0].chosen, decisions[0].chosen)
    assert ctx["role"] in ("lead", "follow")
    assert ctx["phase"] in ("open", "mid", "late")
    assert ctx["caller_ctx"] in ("none", "self", "partner", "opp")


def test_playout_parity_holds_in_a_determinized_world():
    # In a resampled world the actor's hand and the public state are unchanged,
    # so the recorded chosen action must still be legal there.
    agents = _agents()
    pos = _position()
    _, decisions = record_round(agents, pos)
    d = decisions[len(decisions) // 2]
    from tichu_training.search.determinize import sample_determinized_world

    world = sample_determinized_world(d.state.private_view(d.seat), None,
                                      random.Random(3))
    assert world.hands[d.seat] == d.state.hands[d.seat]
    assert d.chosen in legal_actions_for(world.private_view(d.seat))
    totals = playout_from(agents, world, forced_action=d.chosen,
                          asked_tichu=d.asked_tichu, initial_scores=d.initial_scores)
    assert isinstance(totals, tuple) and len(totals) == 2


def test_candidate_seed_is_stable_across_processes_with_different_hash_salts():
    """A tier-2 verdict must be reproducible. The seed therefore may not depend on
    builtin hash(), which is PYTHONHASHSEED-salted and re-salted in every spawn
    worker — so two runs of the same candidate would draw different worlds."""
    assert _seed_in_subprocess("0") == _seed_in_subprocess("12345")


def test_candidate_seed_separates_candidates():
    """Distinct candidates must draw distinct worlds — a constant seed would
    satisfy the stability test above while collapsing every verdict onto one
    sample of the hidden-hand distribution."""
    base = candidate_seed(7, 13, "Pass()")
    assert candidate_seed(8, 13, "Pass()") != base
    assert candidate_seed(7, 14, "Pass()") != base
    assert candidate_seed(7, 13, "PlaySingle(rank=5)") != base
