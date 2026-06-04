"""SearchAgent — the Phase-2 PIMC Search Agent (ADR-0030).

`act` runs PIMC (root-parallel determinized MCTS over the frozen `master` policy)
on **Play** Decisions and delegates every other Decision — pending wish/dragon/
schupfen via `act`, and Tichu/Grand-Tichu via `should_call` — to the wrapped
master policy. v1 is the offline strength experiment: belief-off sampling and the
fast-rollout leaf; both are pluggable upgrade points.

Registered as `"search"` so a Tournament config can build it like any agent. The
registered constructor builds a real `MLAgent`; `from_policy` injects a stand-in
for testing without torch.
"""

import random

from tichu_ml.agent import Agent
from tichu_ml.registry import register_agent
from tichu_training.search.determinize import sample_determinized_world
from tichu_training.search.engine_world import EngineWorld, fast_rollout_leaf
from tichu_training.search.mcts import pimc_decide


@register_agent("search")
class SearchAgent(Agent):
    def __init__(
        self,
        checkpoint_path,
        *,
        skill_decile: int = 9,
        schupfen_path=None,
        tichu_call_path=None,
        grand_call_path=None,
        worlds: int = 4,
        sims: int = 30,
        c_puct: float = 1.4,
        seed: int = 0,
        critic_path=None,
    ) -> None:
        # Lazy import: keeps the search package torch-free for unit tests that use
        # `from_policy`. The master supplies the PUCT prior, the in-tree env moves,
        # and the delegated calls/pending decisions.
        from tichu_inference.ml_agent import MLAgent

        master = MLAgent(
            checkpoint_path,
            skill_decile=skill_decile,
            schupfen_path=schupfen_path,
            tichu_call_path=tichu_call_path,
            grand_call_path=grand_call_path,
        )
        # Leaf evaluator: critic-bootstrap when a Value Baseline is supplied
        # (ADR-0030 Phase B), else the fast rollout. Pluggable seam — one of the two.
        leaf_fn = fast_rollout_leaf
        if critic_path is not None:
            from tichu_training.search.critic import load_critic

            leaf_fn = load_critic(critic_path)
        self._init(master, worlds=worlds, sims=sims, c_puct=c_puct, seed=seed, leaf_fn=leaf_fn)

    @classmethod
    def from_policy(cls, policy, *, worlds=4, sims=30, c_puct=1.4, seed=0, leaf_fn=None) -> "SearchAgent":
        self = cls.__new__(cls)
        self._init(policy, worlds=worlds, sims=sims, c_puct=c_puct, seed=seed, leaf_fn=leaf_fn)
        return self

    def _init(self, policy, *, worlds, sims, c_puct, seed, leaf_fn=None) -> None:
        self._policy = policy
        self._worlds = int(worlds)
        self._sims = int(sims)
        self._c_puct = float(c_puct)
        self._rng = random.Random(seed)
        self._leaf_fn = leaf_fn or fast_rollout_leaf

    def act(self, private_state):
        # Non-Play Decisions (pending wish/dragon/schupfen) are delegated — Play-only
        # search (ADR-0030); the call bonus still reaches the search via round_outcome.
        if private_state.public.pending_decision is not None:
            return self._policy.act(private_state)

        root = private_state.player
        policy = self._policy
        leaf_fn = self._leaf_fn

        def make_world(rng):
            world_state = sample_determinized_world(private_state, None, rng)  # belief-off
            return EngineWorld(root, policy, leaf_fn), world_state

        best, _, _ = pimc_decide(
            make_world, self._worlds, self._sims, self._rng, self._c_puct
        )
        return best

    def should_call(self, private_state, kind: str) -> bool:
        return self._policy.should_call(private_state, kind)
