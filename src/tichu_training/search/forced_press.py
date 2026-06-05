"""ForcedPressAgent — premise test for the caller-passivity pathology (ADR-0031).

Three value interventions failed to reduce caller passivity in search, so the bottleneck
is not the value. Before more engineering, test the *premise*: is pressing the lead in
caller-following states actually +EV? This agent plays the master everywhere EXCEPT the
exact states `play_full_round`'s telemetry flags as caller-passivity opportunities — a
Tichu/Grand caller, following an OPPONENT's top, with a legal beating play — where it forces
the master's best **non-Pass** action instead of (possibly) ceding. A head-to-head tournament
vs the bare master then measures whether that extra pressing helps or hurts:

  * forced-press scores BETTER -> pressing is +EV -> the passivity is a real pathology and
    the fix must live in the policy (search keeps declining it because the prior dominates);
  * forced-press scores <= master -> ceding is fine -> the "pathology" is mis-measured.

The detection mirrors `tichu_eval.play_full` exactly (callers read from the PublicState the
runner injects), so "force a beat here" lines up with "what the telemetry counts as a pass".
"""

from tichu_engine.legality import Pass, legal_actions_for
from tichu_ml.agent import Agent
from tichu_ml.registry import register_agent


@register_agent("forced_press")
class ForcedPressAgent(Agent):
    def __init__(
        self,
        checkpoint_path,
        *,
        skill_decile: int = 9,
        schupfen_path=None,
        tichu_call_path=None,
        grand_call_path=None,
    ) -> None:
        from tichu_inference.ml_agent import MLAgent

        self._policy = MLAgent(
            checkpoint_path,
            skill_decile=skill_decile,
            schupfen_path=schupfen_path,
            tichu_call_path=tichu_call_path,
            grand_call_path=grand_call_path,
        )

    @classmethod
    def from_policy(cls, policy) -> "ForcedPressAgent":
        self = cls.__new__(cls)
        self._policy = policy
        return self

    def act(self, private_state):
        if self._is_caller_following_with_beat(private_state):
            beat = self._best_non_pass(private_state)
            if beat is not None:
                return beat
        return self._policy.act(private_state)

    def should_call(self, private_state, kind: str) -> bool:
        return self._policy.should_call(private_state, kind)

    def rank_actions(self, private_state):
        return self._policy.rank_actions(private_state)

    @staticmethod
    def _is_caller_following_with_beat(pv) -> bool:
        # Mirror play_full's caller-passivity opportunity test exactly.
        pub = pv.public
        if pub.pending_decision is not None:
            return False
        cur = pv.player
        if cur not in pub.tichu_callers and cur not in pub.grand_tichu_callers:
            return False
        leader = pub.trick.leader
        if leader is None or leader == cur or (leader % 2) == (cur % 2):
            return False
        return any(not isinstance(a, Pass) for a in legal_actions_for(pv))

    def _best_non_pass(self, pv):
        ranked = self._policy.rank_actions(pv) or list(legal_actions_for(pv))
        for a in ranked:
            if not isinstance(a, Pass):
                return a
        return None
