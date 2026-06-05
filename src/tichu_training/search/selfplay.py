"""SelfPlaySearchAgent — the data-generation half of the search+learning loop (ADR-0031).

On a Play Decision it runs PIMC with root Dirichlet noise (Decision G) over the
opponent-sampling `EngineWorld` (Decision F), turns the aggregated root visits into
the policy-improvement target π (Decision C), **records** `(features, π)`, and advances
the game by sampling the action from π at temperature τ (so self-play explores the
states the visit distribution favours). Pending wish/dragon/schupfen and the Tichu/Grand
calls delegate to the wrapped policy, so the full call/leadership structure — hence the
`round_outcome` the value target needs — stays realistic.

`from_policy` injects a torch-free stand-in for unit tests; the registered/real entry
point builds an `MLAgent` (+ critic leaf) exactly like `SearchAgent`.
"""

import random

import numpy as np

from tichu_ml.agent import Agent
from tichu_training.action_space import ACTION_SPACE_SIZE
from tichu_training.featurizer import _combination_to_action_index, featurize
from tichu_training.search.determinize import sample_determinized_world
from tichu_training.search.engine_world import (
    EngineWorld,
    fast_rollout_leaf,
    sample_from_scores,
)
from tichu_training.search.mcts import pimc_decide, visit_policy


def play_target_vectors(pi: dict):
    """Vectorise a visit-distribution π (over legal ConcreteActions) into the play
    head's training target: ``(target_probs, legal_mask)`` of shape
    ``(ACTION_SPACE_SIZE,)``. Mass lands at each action's canonical intent index;
    actions the action space cannot represent (no index) are dropped and the target is
    renormalised over the representable legal actions, matching what the head can emit.
    `legal_mask` is True exactly on the represented legal slots (for `masked_kl_divergence`)."""
    probs = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    for action, p in pi.items():
        idx = _combination_to_action_index(action)
        if idx is not None and 0 <= idx < ACTION_SPACE_SIZE:
            probs[idx] += p
            mask[idx] = True
    total = float(probs.sum())
    if total > 0:
        probs /= total
    return probs, mask


class SelfPlaySearchAgent(Agent):
    def __init__(
        self,
        checkpoint_path,
        *,
        skill_decile: int = 9,
        schupfen_path=None,
        tichu_call_path=None,
        grand_call_path=None,
        worlds: int = 4,
        sims: int = 50,
        c_puct: float = 1.4,
        root_alpha: float = 1.0,
        root_eps: float = 0.25,
        temperature: float = 1.0,
        seed: int = 0,
        critic_path=None,
        records=None,
    ) -> None:
        from tichu_inference.ml_agent import MLAgent

        master = MLAgent(
            checkpoint_path,
            skill_decile=skill_decile,
            schupfen_path=schupfen_path,
            tichu_call_path=tichu_call_path,
            grand_call_path=grand_call_path,
        )
        leaf_fn = fast_rollout_leaf
        if critic_path is not None:
            from tichu_training.search.critic import load_critic

            leaf_fn = load_critic(critic_path)
        self._init(master, worlds=worlds, sims=sims, c_puct=c_puct, leaf_fn=leaf_fn,
                   root_alpha=root_alpha, root_eps=root_eps, temperature=temperature,
                   seed=seed, records=records)

    @classmethod
    def from_policy(cls, policy, *, worlds=4, sims=50, c_puct=1.4, leaf_fn=None,
                    root_alpha=1.0, root_eps=0.25, temperature=1.0, seed=0,
                    records=None) -> "SelfPlaySearchAgent":
        self = cls.__new__(cls)
        self._init(policy, worlds=worlds, sims=sims, c_puct=c_puct, leaf_fn=leaf_fn,
                   root_alpha=root_alpha, root_eps=root_eps, temperature=temperature,
                   seed=seed, records=records)
        return self

    def _init(self, policy, *, worlds, sims, c_puct, leaf_fn, root_alpha, root_eps,
              temperature, seed, records) -> None:
        self._policy = policy
        self._worlds = int(worlds)
        self._sims = int(sims)
        self._c_puct = float(c_puct)
        self._root_noise = (float(root_alpha), float(root_eps))
        self._temperature = float(temperature)
        self._rng = random.Random(seed)
        self._leaf_fn = leaf_fn or fast_rollout_leaf
        # Per-decision (features, π) targets, in play order. The collector reads this
        # after the round and attaches the team-relative outcome z to each entry.
        self.records: list = records if records is not None else []

    def act(self, private_state):
        if private_state.public.pending_decision is not None:
            return self._policy.act(private_state)

        root = private_state.player
        policy = self._policy
        leaf_fn = self._leaf_fn

        def make_world(rng):
            world_state = sample_determinized_world(private_state, None, rng)  # belief-off
            return EngineWorld(root, policy, leaf_fn, opponent_sample=True), world_state

        _, agg_visits, _ = pimc_decide(
            make_world, self._worlds, self._sims, self._rng, self._c_puct,
            root_noise=self._root_noise,
        )
        pi = visit_policy(agg_visits)
        self.records.append((featurize(private_state), pi))
        return self._sample_action(pi)

    def _sample_action(self, pi: dict):
        # τ→0 is greedy; τ=1 samples ∝ visits; in between, ∝ π^(1/τ).
        if self._temperature <= 0.0:
            return max(pi, key=pi.get)
        scored = [(a, p ** (1.0 / self._temperature)) for a, p in pi.items()]
        return sample_from_scores(scored, self._rng)

    def should_call(self, private_state, kind: str) -> bool:
        return self._policy.should_call(private_state, kind)
