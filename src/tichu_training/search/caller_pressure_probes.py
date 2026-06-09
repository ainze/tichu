"""Caller-pressure probes — EV-test cotrain's handling of an OPPOSING caller (2026-06-08).

`scripts/mine_divergences.py` found that on held-out human games, cotrain diverges from
BC+human consensus most in caller states: it CEDES (passes) where strong humans contest an
opposing caller (cluster `follow/cotrain_cedes/opp_caller`, n=142 — mostly early-game
SINGLE/PAIR beats, even a few bombs, declined), and inversely CONTESTS where humans pass
(`cotrain_contests/opp_caller`, n=268). So cotrain handles an opposing caller on a DIFFERENT
calibration than humans — the user's "pressure the caller, don't let them finish easily"
intuition vs the literature's "let them go out, conserve for the double-out".

A blanket rule can't capture a state-dependent miscalibration, but a forced-action probe over
the actual trigger can. Both probes fire on the same state — FOLLOWING an OPPONENT's trick
while an OPPONENT has called (Grand-)Tichu — and force opposite corrections, wrapping the SAME
policy so the tournament delta vs the bare policy is purely the intervention:

  forced_press_opp_caller (cede->press): cotrain would Pass -> force its best non-bomb beat.
  forced_yield_opp_caller (contest->pass): cotrain would beat with a non-bomb -> force Pass.

Reading the pair (each vs the bare subject, via scripts/probe_heuristics.py):
  press +EV & yield -EV -> cotrain UNDER-pressures callers (the press is real fruit).
  press -EV & yield +EV -> cotrain OVER-pressures (conserving is better — literature wins).
  both ~0               -> cotrain is already calibrated; the human divergence is EV-neutral.
  both +EV              -> state-dependent miscalibration the blanket trigger over-broadens.

Distinct from `forced_press` (ADR-0031), which fires when the agent ITSELF is the caller and
cedes; these fire when an OPPONENT is the caller. Bombs are excluded from both interventions
so the probes isolate normal contesting from bomb-deployment (tested separately by forced_bomb).
"""

from tichu_engine.legality import PASS, Pass, legal_actions_for
from tichu_ml.registry import register_agent
from tichu_training.search.heuristic_probes import _BOMB_TYPES, _MasterProbe


def _following_opp_caller(pv) -> bool:
    """True iff the actor is following an OPPONENT's trick while an OPPONENT has called.
    (Leader on the opposing team avoids 'pressing' a partner's trick; an opposing caller
    is what makes the press/yield decision load-bearing.)"""
    pub = pv.public
    if pub.pending_decision is not None:
        return False
    leader = pub.trick.leader
    me = pv.player
    if leader is None or (leader % 2) == (me % 2):
        return False  # leading, or following partner/self — not an opponent's trick
    callers = pub.tichu_callers | pub.grand_tichu_callers
    return any((c % 2) != (me % 2) for c in callers)


@register_agent("forced_press_opp_caller")
class ForcedPressOppCallerAgent(_MasterProbe):
    """cede->press: where the master would Pass while following an opposing caller's trick,
    force its best non-bomb beat (contest the caller). Tests whether cotrain under-pressures."""

    def act(self, private_state):
        action = self._policy.act(private_state)
        if isinstance(action, Pass) and _following_opp_caller(private_state):
            beat = self._best_non_bomb_beat(private_state)
            if beat is not None:
                self.interventions += 1
                return beat
        return action

    def _best_non_bomb_beat(self, pv):
        ranked = self._policy.rank_actions(pv) or list(legal_actions_for(pv))
        for a in ranked:
            if not isinstance(a, (Pass, *_BOMB_TYPES)):
                return a
        return None


@register_agent("forced_yield_opp_caller")
class ForcedYieldOppCallerAgent(_MasterProbe):
    """contest->pass (the control): where the master would beat with a non-bomb while
    following an opposing caller's trick, force a Pass. Tests whether cotrain over-pressures."""

    def act(self, private_state):
        action = self._policy.act(private_state)
        if (
            not isinstance(action, (Pass, *_BOMB_TYPES))
            and _following_opp_caller(private_state)
            and any(isinstance(a, Pass) for a in legal_actions_for(private_state))
        ):
            self.interventions += 1
            return PASS
        return action
