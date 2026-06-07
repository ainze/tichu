"""Batched play-policy sampler for PPO Refine (ADR-0029).

The policy forward runs once over a batch of decision-points; this module turns
its play-head logits into sampled Intents plus the log-probs PPO's objective
needs, with illegal Intents masked out so a sampled action is always legal.
"""

from typing import NamedTuple

import numpy as np
import torch

from tichu_engine.legality import legal_actions_for
from tichu_training.action_space import play_intent_index
from tichu_training.bc.heads import HEAD_LOGIT_DIMS
from tichu_training.featurizer import featurize
from tichu_training.perfect_info import featurize_perfect_info
from tichu_training.ppo.rollout import PlayChoice

_PLAY_DIM = HEAD_LOGIT_DIMS["play"]


class SampledBatch(NamedTuple):
    indices: torch.Tensor   # (B,) sampled Intent index per row
    logprobs: torch.Tensor  # (B,) log-prob of the sampled Intent (legal-masked)
    values: torch.Tensor    # (B,) critic value estimate per row


class BatchedPolicy:
    """Wraps the policy `BCModel` and the separate critic, sampling a batch of
    Play decisions in one forward each.

    The critic is deliberately a separate network (ADR-0029 §value): the policy
    Trunk is shaped only by the policy gradient + KL anchor + entropy, never by
    value-loss gradients. The rollout forward runs under `no_grad` — the sampled
    log-probs and values are stored as constants; the PPO update re-runs the
    forward with gradients.
    """

    def __init__(
        self, model, critic, *, skill_decile: int = 9, generator=None,
        perfect_info: bool = False,
    ) -> None:
        self._model = model
        self._critic = critic
        self._skill_decile = int(skill_decile)  # fixed decile-9 self-play (ADR-0029)
        self._generator = generator
        # When True, the critic is asymmetric (Perfect-Info Critic, ADR-0033): it
        # values the 392-dim perfect-info features built from the GameState, while
        # the policy keeps featurizing the observable PrivateState.
        self._perfect_info = bool(perfect_info)

    def act_play_batch(self, decisions: list[tuple[int, object]]) -> list[PlayChoice]:
        """Decide a tick's worth of Play Decisions in one batched forward.

        `decisions` is `[(seat, private_state), ...]`. For each: featurize the
        PrivateState, build the legal-Intent mask from `legal_actions_for` via
        the canonical `play_intent_index` (same mapping training uses), and keep
        the resolution from sampled Intent index back to a legal ConcreteAction.
        Returns one `PlayChoice` per decision, in order.
        """
        masks = torch.zeros(len(decisions), _PLAY_DIM, dtype=torch.bool)
        feats: list[np.ndarray] = []
        pi_feats: list[np.ndarray] = []  # perfect-info; populated only when asymmetric
        resolvers: list[dict[int, object]] = []
        for row, (seat, private_state, game_state) in enumerate(decisions):
            feats.append(featurize(private_state))
            if self._perfect_info:
                pi_feats.append(featurize_perfect_info(game_state, seat))
            by_index: dict[int, object] = {}
            for action in legal_actions_for(private_state):
                try:
                    idx = play_intent_index(action)
                except ValueError:
                    continue  # shape the v1 Action Space can't represent
                if 0 <= idx < _PLAY_DIM:
                    masks[row, idx] = True
                    by_index.setdefault(idx, action)
            resolvers.append(by_index)

        features = torch.from_numpy(np.stack(feats))
        critic_features = (
            torch.from_numpy(np.stack(pi_feats)) if self._perfect_info else None
        )
        skill = torch.full((len(decisions),), self._skill_decile, dtype=torch.long)
        out = self.sample(
            features, skill, masks,
            critic_features=critic_features, generator=self._generator,
        )

        choices: list[PlayChoice] = []
        for row in range(len(decisions)):
            idx = int(out.indices[row])
            choices.append(
                PlayChoice(
                    concrete_action=resolvers[row][idx],
                    intent_index=idx,
                    logprob=float(out.logprobs[row]),
                    value=float(out.values[row]),
                    features=features[row],
                    legal_mask=masks[row],
                    critic_features=(
                        critic_features[row] if critic_features is not None else None
                    ),
                )
            )
        return choices

    def sample(
        self,
        features: torch.Tensor,
        skill: torch.Tensor,
        legal_masks: torch.Tensor,
        *,
        critic_features: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> SampledBatch:
        with torch.no_grad():
            play_logits = self._model(features, skill)["play"]
            indices, logprobs = sample_masked(play_logits, legal_masks, generator=generator)
            # Asymmetric critic values the perfect-info features when given; else
            # the symmetric path values the same observable features (ADR-0033).
            values = self._critic(
                critic_features if critic_features is not None else features
            )
        return SampledBatch(indices=indices, logprobs=logprobs, values=values)


def sample_masked(
    logits: torch.Tensor,
    legal_masks: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one Intent per row from `logits`, restricted to `legal_masks`.

    `logits` and `legal_masks` are both `(B, action_dim)`; the mask is boolean
    (True == legal). Returns `(indices, logprobs)`, each shape `(B,)`: the
    sampled Intent index and its log-probability under the legal-masked softmax.
    Illegal Intents are given zero probability mass, so a sampled index is
    always legal.
    """
    masked_logits = logits.masked_fill(~legal_masks, float("-inf"))
    logp = torch.log_softmax(masked_logits, dim=-1)
    indices = torch.multinomial(logp.exp(), num_samples=1, generator=generator).squeeze(-1)
    chosen_logp = logp.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    return indices, chosen_logp
