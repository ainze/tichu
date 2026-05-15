"""MLAgent — production inference adapter wrapping an exported policy.

Wraps a TorchScript-exported BCModel and presents the standard `Agent`
interface. The model is used for ordinary play decisions where the
canonical action space directly maps to engine actions; for pending
decisions (schupfen / mahjong wish / dragon give) the agent currently
falls through to `RuleAgent` heuristics. Wiring the pass_card /
wish_rank / dragon_give heads through to concrete engine actions is a
follow-up — see the FIXME-style comment in `_act_on_play`.

Fallback contract: any exception inside the inference path, NaN/inf
logits, or "no legal action with positive probability" condition logs
at ERROR level and returns a uniform-random legal action. The game
never stalls.
"""

import logging
import random
from pathlib import Path

import numpy as np
import torch

from tichu_engine.legality import (
    Action,
    DragonGivePending,
    MahjongWishPending,
    SchupfenPending,
    legal_actions_for,
)
from tichu_engine.state import PrivateState
from tichu_export.torchscript import (
    VersionMismatchError,
    load_exported,
)
from tichu_ml.agent import Agent
from tichu_ml.rule_agent import RuleAgent
from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.featurizer import FEATURIZER_VERSION, featurize
from tichu_training.featurizer import _combination_to_action_index


log = logging.getLogger(__name__)

_NEUTRAL_SKILL = 10  # the embedding's neutral / cold-start row


class MLAgent(Agent):
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        fallback_rng: random.Random | None = None,
    ) -> None:
        self._module = load_exported(
            checkpoint_path,
            expected_featurizer_version=FEATURIZER_VERSION,
            expected_action_space_version=ACTION_SPACE_VERSION,
        )
        self._rule_fallback = RuleAgent()
        self._rng = fallback_rng or random.Random(0)
        self.last_fallback_used: bool = False

    def act(self, private_state: PrivateState) -> Action:
        self.last_fallback_used = False
        pending = private_state.public.pending_decision
        if isinstance(pending, (SchupfenPending, DragonGivePending, MahjongWishPending)):
            # See module docstring: head-specific decoding for these is a
            # follow-up. Delegate to the rule agent so a legal action is
            # always produced; not counted as a fallback (it is a known
            # routing decision, not an error).
            return self._rule_fallback.act(private_state)

        try:
            return self._act_on_play(private_state)
        except Exception as exc:  # noqa: BLE001 — fallback is the explicit contract.
            log.error("inference fallback: %s; using random legal action", exc)
            self.last_fallback_used = True
            return self._random_legal(private_state)

    def rank_actions(self, private_state: PrivateState) -> list[Action] | None:
        pending = private_state.public.pending_decision
        if isinstance(pending, (SchupfenPending, DragonGivePending, MahjongWishPending)):
            return None  # head-specific ranking not implemented yet.
        try:
            return self._rank_play_actions(private_state)
        except Exception as exc:  # noqa: BLE001
            log.error("rank_actions fallback: %s", exc)
            return list(legal_actions_for(private_state))

    # ------------------------------------------------------------
    # Play-head inference.
    # ------------------------------------------------------------

    def _act_on_play(self, private_state: PrivateState) -> Action:
        legal = list(legal_actions_for(private_state))
        if not legal:
            raise RuntimeError("no legal actions available")
        play_logits = self._play_logits(private_state)
        if not _is_finite(play_logits):
            raise RuntimeError("non-finite logits from policy")

        ranked = _rank_legal_by_logits(legal, play_logits)
        if not ranked:
            raise RuntimeError("no legal action mapped into the action space")
        return ranked[0]

    def _rank_play_actions(self, private_state: PrivateState) -> list[Action]:
        legal = list(legal_actions_for(private_state))
        play_logits = self._play_logits(private_state)
        if not _is_finite(play_logits):
            raise RuntimeError("non-finite logits")
        ranked = _rank_legal_by_logits(legal, play_logits)
        # If no legal action mapped, return all legal in arbitrary order so callers
        # still see a non-empty list.
        return ranked or legal

    def _play_logits(self, private_state: PrivateState) -> np.ndarray:
        features_np = featurize(private_state)
        features = torch.from_numpy(features_np).unsqueeze(0)
        skill = torch.tensor([_NEUTRAL_SKILL], dtype=torch.long)
        out = self._module(features, skill)
        play = out["play"][0]
        return play.detach().cpu().numpy()

    def _random_legal(self, private_state: PrivateState) -> Action:
        legal = list(legal_actions_for(private_state))
        if not legal:
            raise RuntimeError("no legal actions for fallback")
        return self._rng.choice(legal)


def _is_finite(logits: np.ndarray) -> bool:
    return bool(np.isfinite(logits).all())


def _rank_legal_by_logits(legal: list[Action], play_logits: np.ndarray) -> list[Action]:
    """Order legal actions by descending policy logit. Actions that don't map to
    a canonical action-space index are placed at the end in their original order."""
    scored: list[tuple[float, int, Action]] = []
    unmapped: list[Action] = []
    for i, action in enumerate(legal):
        idx = _combination_to_action_index(action)
        if idx is None or not (0 <= idx < play_logits.shape[0]):
            unmapped.append(action)
            continue
        scored.append((float(play_logits[idx]), i, action))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [a for _, _, a in scored] + unmapped
