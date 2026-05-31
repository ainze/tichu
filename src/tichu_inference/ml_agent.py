"""MLAgent — production inference adapter wrapping the trained networks.

Wraps the TorchScript-exported policy and (optionally) the standalone
Schupfen / Tichu-call / Grand-Tichu-call networks, presenting the standard
`Agent` interface plus a `should_call` method for the call decisions (which
are not engine pending-decisions and so cannot flow through `act`).

Decision routing inside `act`:
  * play              -> policy `play` head, ranked over legal actions.
  * mahjong wish      -> policy `wish` head, scored over legal MahjongWish.
  * dragon give       -> policy `dragon_assignment` head, scored over legal DragonGive.
  * schupfen          -> standalone Schupfen Network, masked-greedy over the hand.

Call decisions (`should_call`):
  * tichu / grand     -> the matching standalone Call Network (binary).

Every decode is constrained to the engine's own legal-action set (wish/dragon)
or to distinct in-hand cards (schupfen), so a learned head can never emit an
illegal action. If a network is not supplied, or anything in the inference path
throws / produces non-finite logits, the agent falls back: pending decisions to
`RuleAgent`, play to a uniform-random legal action, and `should_call` to
declining. The game never stalls.
"""

import logging
import random
from pathlib import Path

import numpy as np
import torch

from tichu_engine.legality import (
    ConcreteAction,
    DragonGive,
    DragonGivePending,
    MahjongWish,
    MahjongWishPending,
    SchupfenPass,
    SchupfenPending,
    legal_actions_for,
)
from tichu_engine.state import PrivateState
from tichu_export.torchscript import (
    VersionMismatchError,
    load_exported,
)
from tichu_ml.agent import Agent
from tichu_ml.registry import register_agent
from tichu_ml.rule_agent import RuleAgent
from tichu_training.action_space import (
    ACTION_SPACE_VERSION,
    dragon_intent_index,
    wish_intent_index,
)
from tichu_training.card_slots import card_slot, slot_to_card
from tichu_training.featurizer import FEATURIZER_VERSION, featurize
from tichu_training.featurizer import _combination_to_action_index


log = logging.getLogger(__name__)

_NEUTRAL_SKILL = 10  # the embedding's neutral / cold-start row


@register_agent("ml")
class MLAgent(Agent):
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        schupfen_path: str | Path | None = None,
        tichu_call_path: str | Path | None = None,
        grand_call_path: str | Path | None = None,
        fallback_rng: random.Random | None = None,
    ) -> None:
        self._module = load_exported(
            checkpoint_path,
            expected_featurizer_version=FEATURIZER_VERSION,
            expected_action_space_version=ACTION_SPACE_VERSION,
        )
        # The standalone networks are exported with an empty action_space stamp
        # (they don't consume the play action space), so only the featurizer
        # version is asserted on load.
        self._schupfen = self._maybe_load(schupfen_path)
        self._tichu_call = self._maybe_load(tichu_call_path)
        self._grand_call = self._maybe_load(grand_call_path)
        self._rule_fallback = RuleAgent()
        self._rng = fallback_rng or random.Random(0)
        self.last_fallback_used: bool = False

    @staticmethod
    def _maybe_load(path: str | Path | None):
        if path is None:
            return None
        return load_exported(path, expected_featurizer_version=FEATURIZER_VERSION)

    def act(self, private_state: PrivateState) -> ConcreteAction:
        self.last_fallback_used = False
        pending = private_state.public.pending_decision

        if isinstance(pending, SchupfenPending):
            return self._dispatch_pending(
                private_state, self._act_schupfen, enabled=self._schupfen is not None
            )
        if isinstance(pending, MahjongWishPending):
            return self._dispatch_pending(private_state, self._act_wish, enabled=True)
        if isinstance(pending, DragonGivePending):
            return self._dispatch_pending(private_state, self._act_dragon, enabled=True)

        try:
            return self._act_on_play(private_state)
        except Exception as exc:  # noqa: BLE001 — fallback is the explicit contract.
            log.error("inference fallback: %s; using random legal action", exc)
            self.last_fallback_used = True
            return self._random_legal(private_state)

    def should_call(self, private_state: PrivateState, kind: str) -> bool:
        """Binary call decision for `kind` in {"tichu", "grand"}.

        Returns False (decline) if the matching Call Network was not supplied or
        the inference path fails — declining is always legal and never stalls.
        """
        if kind == "tichu":
            module = self._tichu_call
        elif kind == "grand":
            module = self._grand_call
        else:
            raise ValueError(f"unknown call kind: {kind!r} (expected 'tichu' or 'grand')")
        if module is None:
            return False
        try:
            logits = self._run(module, private_state)
            if not _is_finite(logits):
                raise RuntimeError("non-finite call logits")
            return bool(int(np.argmax(logits)) == 1)
        except Exception as exc:  # noqa: BLE001
            log.error("call inference fallback (%s): %s; declining", kind, exc)
            return False

    def rank_actions(self, private_state: PrivateState) -> list[ConcreteAction] | None:
        pending = private_state.public.pending_decision
        if isinstance(pending, (SchupfenPending, DragonGivePending, MahjongWishPending)):
            return None  # head-specific ranking not implemented for pending decisions.
        try:
            return self._rank_play_actions(private_state)
        except Exception as exc:  # noqa: BLE001
            log.error("rank_actions fallback: %s", exc)
            return list(legal_actions_for(private_state))

    # ------------------------------------------------------------
    # Pending-decision dispatch with rule fallback.
    # ------------------------------------------------------------

    def _dispatch_pending(self, private_state, handler, *, enabled: bool) -> ConcreteAction:
        # `enabled` is False when the relevant network wasn't supplied (schupfen);
        # routing straight to the rule agent is a known decision, not an error,
        # so it is not counted as a fallback.
        if not enabled:
            return self._rule_fallback.act(private_state)
        try:
            return handler(private_state)
        except Exception as exc:  # noqa: BLE001
            log.error("pending-decision fallback: %s; using rule agent", exc)
            self.last_fallback_used = True
            return self._rule_fallback.act(private_state)

    # ------------------------------------------------------------
    # Play-head inference.
    # ------------------------------------------------------------

    def _act_on_play(self, private_state: PrivateState) -> ConcreteAction:
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

    def _rank_play_actions(self, private_state: PrivateState) -> list[ConcreteAction]:
        legal = list(legal_actions_for(private_state))
        play_logits = self._play_logits(private_state)
        if not _is_finite(play_logits):
            raise RuntimeError("non-finite logits")
        ranked = _rank_legal_by_logits(legal, play_logits)
        return ranked or legal

    def _play_logits(self, private_state: PrivateState) -> np.ndarray:
        out = self._policy_forward(private_state)
        return out["play"][0].detach().cpu().numpy()

    # ------------------------------------------------------------
    # Wish / dragon — policy heads scored over the legal-action set.
    # ------------------------------------------------------------

    def _act_wish(self, private_state: PrivateState) -> ConcreteAction:
        out = self._policy_forward(private_state)
        wish_logits = out["wish"][0].detach().cpu().numpy()
        if not _is_finite(wish_logits):
            raise RuntimeError("non-finite wish logits")
        legal = [a for a in legal_actions_for(private_state) if isinstance(a, MahjongWish)]
        if not legal:
            raise RuntimeError("no legal MahjongWish actions")
        return max(legal, key=lambda a: float(wish_logits[wish_intent_index(a)]))

    def _act_dragon(self, private_state: PrivateState) -> ConcreteAction:
        pending = private_state.public.pending_decision
        winner = pending.winner  # DragonGivePending
        out = self._policy_forward(private_state)
        dragon_logits = out["dragon_assignment"][0].detach().cpu().numpy()
        if not _is_finite(dragon_logits):
            raise RuntimeError("non-finite dragon logits")
        legal = [a for a in legal_actions_for(private_state) if isinstance(a, DragonGive)]
        if not legal:
            raise RuntimeError("no legal DragonGive actions")
        return max(legal, key=lambda a: float(dragon_logits[dragon_intent_index(a, winner)]))

    # ------------------------------------------------------------
    # Schupfen — three 56-way heads, masked-greedy over the hand.
    # ------------------------------------------------------------

    def _act_schupfen(self, private_state: PrivateState) -> ConcreteAction:
        features, skill = _inputs(private_state)
        h_next, h_partner, h_prev = self._schupfen(features, skill)
        rows = [
            h_next[0].detach().cpu().numpy(),
            h_partner[0].detach().cpu().numpy(),
            h_prev[0].detach().cpu().numpy(),
        ]
        if not all(_is_finite(r) for r in rows):
            raise RuntimeError("non-finite schupfen logits")

        hand_slots = [card_slot(c) for c in private_state.hand]
        if len(hand_slots) < 3:
            raise RuntimeError("fewer than 3 cards in hand for schupfen")

        used: set[int] = set()
        chosen: list[int] = []
        for row in rows:
            slot = _best_slot(row, hand_slots, used)
            if slot is None:
                raise RuntimeError("could not pick a distinct schupfen card")
            used.add(slot)
            chosen.append(slot)

        return SchupfenPass(
            to_next=slot_to_card(chosen[0]),
            to_partner=slot_to_card(chosen[1]),
            to_previous=slot_to_card(chosen[2]),
        )

    # ------------------------------------------------------------
    # Shared forward helpers.
    # ------------------------------------------------------------

    def _policy_forward(self, private_state: PrivateState) -> dict:
        features, skill = _inputs(private_state)
        return self._module(features, skill)

    def _run(self, module, private_state: PrivateState) -> np.ndarray:
        features, skill = _inputs(private_state)
        out = module(features, skill)
        return out[0].detach().cpu().numpy()

    def _random_legal(self, private_state: PrivateState) -> ConcreteAction:
        legal = list(legal_actions_for(private_state))
        if not legal:
            raise RuntimeError("no legal actions for fallback")
        return self._rng.choice(legal)


def _inputs(private_state: PrivateState) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.from_numpy(featurize(private_state)).unsqueeze(0)
    skill = torch.tensor([_NEUTRAL_SKILL], dtype=torch.long)
    return features, skill


def _best_slot(logits_row: np.ndarray, allowed_slots: list[int], used: set[int]) -> int | None:
    best_slot: int | None = None
    best_val = -np.inf
    for slot in allowed_slots:
        if slot in used:
            continue
        val = float(logits_row[slot])
        if val > best_val:
            best_val = val
            best_slot = slot
    return best_slot


def _is_finite(logits: np.ndarray) -> bool:
    return bool(np.isfinite(logits).all())


def _rank_legal_by_logits(legal: list[ConcreteAction], play_logits: np.ndarray) -> list[ConcreteAction]:
    """Order legal actions by descending policy logit. Actions that don't map to
    a canonical action-space index are placed at the end in their original order."""
    scored: list[tuple[float, int, ConcreteAction]] = []
    unmapped: list[ConcreteAction] = []
    for i, action in enumerate(legal):
        idx = _combination_to_action_index(action)
        if idx is None or not (0 <= idx < play_logits.shape[0]):
            unmapped.append(action)
            continue
        scored.append((float(play_logits[idx]), i, action))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [a for _, _, a in scored] + unmapped
