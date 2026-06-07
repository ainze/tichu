"""ForcedSchupfenAgent — schupfen-coupling EV-probe (ADR-0033 follow-up).

Plays the master everywhere EXCEPT the **Schupfen** Decision, which it overrides:
  * `mode="rule"`   — the RuleAgent heuristic (give the three lowest cards);
  * `mode="random"` — a random legal **SchupfenPass**.

Head-to-head vs the bare master, it tests whether master's *learned* schupfen
bottlenecks play EV — the play<->schupfen coupling hypothesis (ADR-0033). Play,
calls, wish and dragon are the master's own; only the schupfen card-pass changes,
so the tournament delta is attributable to schupfen.
"""

import random

from tichu_engine.legality import SchupfenPass, legal_actions_for
from tichu_engine.state import SchupfenPending
from tichu_ml.agent import Agent
from tichu_ml.registry import register_agent
from tichu_ml.rule_agent import RuleAgent


@register_agent("forced_schupfen")
class ForcedSchupfenAgent(Agent):
    def __init__(
        self,
        checkpoint_path,
        *,
        skill_decile: int = 9,
        schupfen_path=None,
        tichu_call_path=None,
        grand_call_path=None,
        mode: str = "rule",
        seed: int = 0,
    ) -> None:
        from tichu_inference.ml_agent import MLAgent

        self._policy = MLAgent(
            checkpoint_path,
            skill_decile=skill_decile,
            schupfen_path=schupfen_path,
            tichu_call_path=tichu_call_path,
            grand_call_path=grand_call_path,
        )
        self._init_override(mode=mode, seed=seed)

    @classmethod
    def from_policy(cls, policy, *, mode: str = "rule", seed: int = 0) -> "ForcedSchupfenAgent":
        self = cls.__new__(cls)
        self._policy = policy
        self._init_override(mode=mode, seed=seed)
        return self

    def _init_override(self, *, mode: str, seed: int) -> None:
        if mode not in ("rule", "random"):
            raise ValueError(f"mode must be 'rule' or 'random', got {mode!r}")
        self._mode = mode
        self._rule = RuleAgent()
        self._rng = random.Random(seed)
        self.forced_count = 0

    def act(self, private_state):
        if isinstance(private_state.public.pending_decision, SchupfenPending):
            self.forced_count += 1
            if self._mode == "rule":
                return self._rule.act(private_state)
            legal = [a for a in legal_actions_for(private_state) if isinstance(a, SchupfenPass)]
            return self._rng.choice(legal)
        return self._policy.act(private_state)

    def should_call(self, private_state, kind: str) -> bool:
        return self._policy.should_call(private_state, kind)

    def rank_actions(self, private_state):
        return self._policy.rank_actions(private_state)
