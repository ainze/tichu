"""ForcedClaimAgent — Stage-2 EV-probe for the Claim Solver (ADR-0032).

Plays the master everywhere EXCEPT where, **leading a fresh trick**, the seat either
(a) called Tichu/Grand-Tichu, or (b) has its partner as the sole player out (a live
**Slam**), AND a **chain Guaranteed Out** provably exists from its Hand against the
**Unseen Cards** — there it forces the first combo of that out (recomputed each lead,
so the whole unbeatable chain runs statelessly). A head-to-head tournament vs the bare
master then measures the EV of executing guaranteed run-outs the master may fumble:

  * forced_claim EV > master  -> the master fumbles guaranteed outs; the Claim Solver is +EV;
  * forced_claim EV <= master -> the master already converts them (like forced_press/bomb).

v1 is **chain-only** (sound, lower-recall): you never cede the lead, so no opponent can
race you out — the guarantee holds worst-case. Regain-the-lead is deferred (ADR-0032).
"""

from tichu_engine.claim import guaranteed_out_chain, reclaim_out, unseen_cards
from tichu_engine.legality import legal_actions_for
from tichu_ml.agent import Agent
from tichu_ml.registry import register_agent

# Endgame scope: a chain Guaranteed Out while leading with a large Hand is implausibly
# rare and the solver's recursion is unbounded, so skip above this size (perf guard).
_MAX_SOLVE_HAND = 10


@register_agent("forced_claim")
class ForcedClaimAgent(Agent):
    def __init__(
        self,
        checkpoint_path,
        *,
        skill_decile: int = 9,
        schupfen_path=None,
        tichu_call_path=None,
        grand_call_path=None,
        reclaim: bool = False,
    ) -> None:
        from tichu_inference.ml_agent import MLAgent

        self._policy = MLAgent(
            checkpoint_path,
            skill_decile=skill_decile,
            schupfen_path=schupfen_path,
            tichu_call_path=tichu_call_path,
            grand_call_path=grand_call_path,
        )
        self._detector = reclaim_out if reclaim else guaranteed_out_chain
        self.forced_count = 0

    @classmethod
    def from_policy(cls, policy, *, reclaim: bool = False) -> "ForcedClaimAgent":
        self = cls.__new__(cls)
        self._policy = policy
        self._detector = reclaim_out if reclaim else guaranteed_out_chain
        self.forced_count = 0
        return self

    def act(self, private_state):
        forced = self._forced_claim_move(private_state)
        if forced is not None:
            self.forced_count += 1
            return forced
        return self._policy.act(private_state)

    def should_call(self, private_state, kind: str) -> bool:
        return self._policy.should_call(private_state, kind)

    def rank_actions(self, private_state):
        return self._policy.rank_actions(private_state)

    def _forced_claim_move(self, pv):
        pub = pv.public
        if pub.pending_decision is not None:
            return None
        if pub.trick.top_combination is not None:      # only when leading a fresh trick
            return None
        if pub.mahjong_wish is not None:               # a wish may forbid the chain's first combo
            return None
        if len(pv.hand) > _MAX_SOLVE_HAND:
            return None
        cur = pv.player
        partner = (cur + 2) % 4
        is_caller = cur in pub.tichu_callers or cur in pub.grand_tichu_callers
        is_slam = pub.out_order == (partner,)
        if not (is_caller or is_slam):
            return None
        out = self._detector(pv.hand, unseen_cards(pv))
        if not out:
            return None
        move = out[0]
        if move not in legal_actions_for(pv):          # never return an illegal action
            return None
        return move
