"""Behavior Bias — the Behavior Sensitivity Probe's perturbation (2026-09-27).

A **lever** is one behavior from the Behavioral Drift Benchmark expressed as a
situation (when it applies) and a target class of Play actions. The bias adds δ
to the play logit of every legal target action — only in the lever's situation,
only on a non-forced Play Decision — before the greedy argmax. With a greedy
policy that flips exactly the Decisions whose best target and best non-target
action lie within δ of each other: the marginal cases, which is what a
first-order EV slope needs. `lever_gap` returns that margin, so a baseline run
predicts how far a given δ moves the lever's rate before any EV is played.

Torch-free; the agent that applies it is `behavior_bias_agent.BiasedMLAgent`.
Pre-registration: docs/notes/2026-09-27-behavior-sensitivity-preregistration.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from tichu_engine.cards import DOG, PHOENIX
from tichu_engine.combinations import Single
from tichu_engine.legality import Pass, _cards_in
from tichu_training.featurizer import _combination_to_action_index


@dataclass(frozen=True)
class Lever:
    name: str
    description: str
    # (private_state, legal) -> whether this Decision is the lever's situation.
    applies: Callable[..., bool]
    # action -> whether it is in the target class δ is added to.
    targets: Callable[[object], bool]


def _is_play(pv) -> bool:
    return pv.public.pending_decision is None


def _leading(pv) -> bool:
    return pv.public.trick.leader is None


def _opponent_holds(pv) -> bool:
    leader = pv.public.trick.leader
    return leader is not None and leader % 2 != pv.player % 2


def _partner_called(pv) -> bool:
    partner = (pv.player + 2) % 4
    return partner in pv.public.tichu_callers or partner in pv.public.grand_tichu_callers


def _is_pass(a) -> bool:
    return isinstance(a, Pass)


def _is_dog(a) -> bool:
    return isinstance(a, Single) and a.card is DOG


def _is_natural_or_special_single(a) -> bool:
    """A Single other than the Dog — the drift benchmark's `single` lead label."""
    return isinstance(a, Single) and a.card is not DOG


def _is_phoenix_combination(a) -> bool:
    if isinstance(a, Pass):
        return False
    cards = _cards_in(a)
    return len(cards) > 1 and PHOENIX in cards


def _both_classes_legal(legal, targets) -> bool:
    hit = [targets(a) for a in legal]
    return any(hit) and not all(hit)


LEVERS: dict[str, Lever] = {
    lever.name: lever
    for lever in (
        Lever("pass_vs_opponent",
              "Pass when an opponent holds the Trick and a beat is legal.",
              lambda pv, legal: _is_play(pv) and _opponent_holds(pv)
              and _both_classes_legal(legal, _is_pass),
              _is_pass),
        Lever("phoenix_in_combination",
              "Play the Phoenix inside a multi-card Combination when one is legal.",
              lambda pv, legal: _is_play(pv)
              and _both_classes_legal(legal, _is_phoenix_combination),
              _is_phoenix_combination),
        Lever("dog_lead",
              "Lead the Dog when holding it and the partner has not called.",
              lambda pv, legal: _is_play(pv) and _leading(pv) and not _partner_called(pv)
              and _both_classes_legal(legal, _is_dog),
              _is_dog),
        Lever("single_lead",
              "Lead a Single (not the Dog).",
              lambda pv, legal: _is_play(pv) and _leading(pv)
              and _both_classes_legal(legal, _is_natural_or_special_single),
              _is_natural_or_special_single),
    )
}


def _target_indices(legal, lever: Lever, n: int) -> tuple[set[int], set[int]]:
    """(target, other) Intent indices over the legal set. Every concrete
    realisation of an Intent shares one logit, so indices — not actions — are
    what gets biased (once each)."""
    target: set[int] = set()
    other: set[int] = set()
    for a in legal:
        idx = _combination_to_action_index(a)
        if idx is None or not 0 <= idx < n:
            continue
        (target if lever.targets(a) else other).add(idx)
    return target, other


def bias_play_logits(pv, legal, logits: np.ndarray, lever: Lever, delta: float) -> np.ndarray:
    """`logits` with δ added on the lever's target Intents when the lever applies;
    otherwise `logits` itself. Never mutates the input (the agent's forward memo
    shares its buffer)."""
    if delta == 0.0 or len(legal) < 2 or not lever.applies(pv, legal):
        return logits
    target, _ = _target_indices(legal, lever, logits.shape[0])
    out = logits.copy()
    out[list(target)] += delta
    return out


def lever_gap(pv, legal, logits: np.ndarray, lever: Lever) -> float | None:
    """Best target logit minus best non-target logit, or None off the lever's
    situation. Negative: the greedy agent picks a non-target, and a bias
    δ > −gap flips it to the target; positive: it picks a target, and δ < −gap
    flips it away."""
    if len(legal) < 2 or not lever.applies(pv, legal):
        return None
    target, other = _target_indices(legal, lever, logits.shape[0])
    if not target or not other:
        return None
    return float(logits[list(target)].max() - logits[list(other)].max())


def choose_delta(gaps, *, share: float, cap: float) -> dict:
    """The pre-registered δ for one lever, from baseline `lever_gap`s.

    To first order +δ flips the situations with −δ < g < 0 to the target and −δ
    the ones with 0 < g < δ away from it. Per sign, returns the smallest |δ|
    whose predicted flip share reaches `share` of all situations — placed
    midway to the next gap, so float noise cannot decide a flip — capped at
    `cap`, with the share it actually predicts (`plus_share`, `minus_share`)."""
    g = np.asarray(gaps, dtype=np.float64)
    n = len(g)
    out = {}
    for sign, margins in (("plus", np.sort(-g[g < 0])), ("minus", np.sort(g[g > 0]))):
        k = int(np.ceil(share * n - 1e-9))
        if k == 0:
            delta = 0.0
        elif k <= len(margins):
            nxt = margins[k] if k < len(margins) else margins[k - 1] + 1.0
            delta = min(cap, (margins[k - 1] + nxt) / 2.0)
        else:
            delta = cap
        flipped = int(np.sum(margins < delta))
        out[sign] = delta if sign == "plus" else -delta
        out[f"{sign}_share"] = flipped / n if n else 0.0
    return out
