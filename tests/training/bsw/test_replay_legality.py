"""Replay-legality test.

Every parsed action must be a legal engine action at the corresponding state.
This is the legality-only half of replay validation — score correctness comes
later (see test_replay_ergebnis.py once engine scoring is finished).
"""

from pathlib import Path

import pytest

from tichu_training.bsw.parser import parse_tch
from tichu_training.bsw.replay import replay_round


_SAMPLES = Path(__file__).resolve().parents[3] / "sample"


def test_cache_does_not_alias_states_within_a_round(monkeypatch):
    """Regression guard. The legal_actions cache is keyed on id(state).
    Without anchoring the cached states, CPython can reuse a dropped state's
    memory address for a freshly-allocated one within the same round —
    returning the old (wrong) legal-action set on a false cache hit, and
    causing step() to reject the now-illegal action. Pin the within-round
    invariant: every id we observe a second time during one round must yield
    the same legal_actions result."""
    import tichu_training.bsw.replay as replay_module
    original = replay_module.legal_actions

    per_round_observations: dict[int, frozenset] = {}
    aliases: list[tuple[frozenset, frozenset]] = []

    def checking(state):
        result = original(state)
        prev = per_round_observations.get(id(state))
        if prev is not None and prev != result:
            aliases.append((prev, result))
        per_round_observations[id(state)] = result
        return result
    monkeypatch.setattr(replay_module, "legal_actions", checking)

    for filename in ("2417500.tch", "2417501.tch"):
        game = parse_tch((_SAMPLES / filename).read_text(encoding="utf-8"), game_id=filename[:-4])
        for r in game.rounds:
            per_round_observations.clear()  # within-round scope only
            replay_round(r)

    assert not aliases, (
        f"id(state) collision detected within a round — two distinct states "
        f"shared an id and would cause stale cache hits. Got {len(aliases)}."
    )


def test_replay_round_calls_legal_actions_at_most_once_per_state(monkeypatch):
    """Performance invariant. Within one replay_round call, `legal_actions`
    should be evaluated at most once per unique state — the round-scoped
    cache must absorb the 4 redundant membership-check call sites that the
    profile showed cost ~86% of pipeline runtime."""
    import tichu_training.bsw.replay as replay_module
    original = replay_module.legal_actions
    queried_ids: list[int] = []

    def counting(state):
        queried_ids.append(id(state))
        return original(state)
    monkeypatch.setattr(replay_module, "legal_actions", counting)

    game = parse_tch((_SAMPLES / "2417500.tch").read_text(encoding="utf-8"), game_id="2417500")
    replay_round(game.rounds[0])

    duplicates = [sid for sid in queried_ids if queried_ids.count(sid) > 1]
    assert not duplicates, (
        f"legal_actions called {len(queried_ids)} times on {len(set(queried_ids))} "
        f"unique states — {len(duplicates) // 2 + 1} states were re-enumerated. "
        "The per-round cache should make each state's enumeration happen exactly once."
    )


@pytest.mark.parametrize("filename", ["2417500.tch", "2417501.tch"])
def test_every_parsed_action_is_legal_in_the_engine(filename):
    game = parse_tch((_SAMPLES / filename).read_text(encoding="utf-8"), game_id=filename[:-4])
    for r in game.rounds:
        result = replay_round(r)
        assert result.illegal_action is None, (
            f"{filename} round {r.round_index}: illegal action {result.illegal_action!r} "
            f"at step {result.steps_taken}"
        )
