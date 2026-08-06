"""`promotion_gate.extra_opponents` resolution: who blocks, who gets rolled out.

Two INDEPENDENT axes per entry, which the config previously conflated:

  `require` — does a CI-validated win over this opponent gate a promotion?
  `rollout` — does the learner best-respond to it during rollouts?

They were one flag until the v7 run needed the third combination: cpfix3328 as a
REQUIRED ship bar that the learner does NOT train against, so its margin stays a
held-out measurement rather than one the policy has been fitted to.

`rollout` defaults to `require`, so every pre-existing config resolves exactly as
before.
"""

import pytest

from tichu_training.cli.train_cotrain import resolve_gate_opponents

_BASE = [("champion", "champ.pt"), ("bc", "bc.pt")]


def _resolve(extra, *, greedy=True):
    return resolve_gate_opponents(extra, base_cycle=list(_BASE), greedy=greedy)


def test_an_extra_opponent_defaults_to_required_and_rolled_out():
    """The historical meaning of a bare `{name, path}` entry — beat-ALL semantics
    plus a slot in the rollout rotation."""
    r = _resolve([{"name": "cpfix", "path": "cpfix.pt"}])

    assert r.opponents == ("champion", "bc", "cpfix")
    assert ("cpfix", "cpfix.pt") in r.opp_cycle
    assert r.observe_only == ()


def test_require_false_is_observe_only_and_never_rolled_out():
    """The visibility stream: scored every window into the CSV, excluded from both
    the promote decision and the training distribution."""
    r = _resolve([{"name": "watch", "path": "w.pt", "require": False}])

    assert r.observe_only == ("watch",)
    assert ("watch", "w.pt") not in r.opp_cycle
    assert ("watch", "w.pt") in r.eval_cycle, "an observe-only stream is still scored"


def test_a_required_opponent_can_be_kept_out_of_the_rollout_cycle():
    """The held-out ship bar. `require: true, rollout: false` blocks promotions
    without letting the learner best-respond to it — otherwise the gate would be
    measuring the margin against the very opponent the policy was fitted to."""
    r = _resolve([{"name": "cpfix", "path": "cpfix.pt",
                   "require": True, "rollout": False}])

    assert r.observe_only == (), "require:true means it gates promotion"
    assert ("cpfix", "cpfix.pt") not in r.opp_cycle, "must not shift training data"
    assert ("cpfix", "cpfix.pt") in r.eval_cycle, "but the greedy eval still scores it"
    assert r.opp_cycle == _BASE, "the rollout rotation is unchanged"


def test_an_opponent_outside_the_rollout_cycle_needs_the_greedy_gate():
    """A sampled gate reads margins off the rollout stream, so a stream that is
    never rolled out can never fill its window — under `require` that deadlocks the
    ratchet forever. Reject it at setup instead."""
    with pytest.raises(ValueError, match="require:false"):
        _resolve([{"name": "cpfix", "path": "c.pt", "rollout": False}], greedy=False)


def test_a_rolled_out_but_unrequired_opponent_is_fine_without_the_greedy_gate():
    """The converse combination: its stream fills from the rollout like any other,
    it just never blocks. Nothing to reject."""
    r = _resolve([{"name": "spar", "path": "s.pt", "require": False, "rollout": True}],
                 greedy=False)

    assert r.observe_only == ("spar",)
    assert ("spar", "s.pt") in r.opp_cycle


def test_duplicate_opponent_names_are_rejected():
    """`champion` and `bc` are reserved; a collision would silently merge two
    margin streams into one."""
    with pytest.raises(ValueError, match="duplicate"):
        _resolve([{"name": "bc", "path": "other.pt"}])
