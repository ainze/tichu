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

import dataclasses
import logging
import random
from pathlib import Path

import numpy as np
import torch

from tichu_engine.combinations import FourOfAKindBomb, StraightFlushBomb
from tichu_engine.legality import (
    PASS,
    ConcreteAction,
    DragonGive,
    DragonGivePending,
    MahjongWish,
    MahjongWishPending,
    Pass,
    SchupfenPass,
    SchupfenPending,
    _cards_in,
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
# The featurizer is INJECTED per agent (default = the live v6 module) so a
# v5-trained export can play in a v6 process via `featurizer_v5_frozen` for a
# cross-version tournament. An injected `featurizer` must expose `featurize` and
# `FEATURIZER_VERSION`. Only `featurize()` is version-dependent and routed per
# agent; `_combination_to_action_index` is ACTION-SPACE-bound (v1, shared across
# featurizer generations — verified byte-identical), so it stays module-level.
from tichu_training import featurizer as _DEFAULT_FEATURIZER
from tichu_training.featurizer import _combination_to_action_index


log = logging.getLogger(__name__)

_NEUTRAL_SKILL = 10  # the embedding's neutral / cold-start row

_BOMB_TYPES = (FourOfAKindBomb, StraightFlushBomb)


def suppress_partner_trick_bomb(private_state, action, legal) -> bool:
    """Keep-partner-trick guard: True iff `action` bombs a trick the agent's own
    partner already tops (the bomb would only steal the partner's lead) and a Pass
    is legal. Carve-out: a (Grand-)Tichu caller going OUT on that bomb keeps it —
    banking the call outweighs the stolen lead.

    Shipped on logic + live observation per the 2026-06-08 probe arc
    (`forced_keep_partner_trick`): the trigger fires ~3/200 deals, below what a
    tournament CI can adjudicate, but the blunder was observed in real play and
    passing is near-always correct in-trigger."""
    if not isinstance(action, _BOMB_TYPES):
        return False
    pub = private_state.public
    if pub.pending_decision is not None:
        return False
    partner = (private_state.player + 2) % 4
    if pub.trick.leader != partner:
        return False
    if not any(isinstance(a, Pass) for a in legal):
        return False
    is_caller = (
        private_state.player in pub.tichu_callers
        or private_state.player in pub.grand_tichu_callers
    )
    if is_caller and len(_cards_in(action)) == len(private_state.hand):
        return False
    return True


def load_policy_module(path: str | Path, *, featurizer=_DEFAULT_FEATURIZER):
    """Load an exported **policy** module, asserting featurizer + action-space versions.

    The featurizer version asserted is the INJECTED featurizer's (default v6), so a
    v5-stamped export loads iff built with `featurizer=featurizer_v5_frozen`. The
    action-space version is shared across featurizer generations (still `v1`)."""
    return load_exported(
        path,
        expected_featurizer_version=featurizer.FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )


def load_standalone_net(path: str | Path, *, featurizer=_DEFAULT_FEATURIZER):
    """Load an exported **standalone** net (schupfen / call). These carry an empty
    action_space stamp (they don't consume the play action space), so only the
    featurizer version is asserted on load."""
    return load_exported(path, expected_featurizer_version=featurizer.FEATURIZER_VERSION)


@register_agent("ml")
class MLAgent(Agent):
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        skill_decile: int = _NEUTRAL_SKILL,
        schupfen_path: str | Path | None = None,
        tichu_call_path: str | Path | None = None,
        grand_call_path: str | Path | None = None,
        fallback_rng: random.Random | None = None,
        tichu_threshold: float = 0.5,
        grand_threshold: float = 0.5,
        partner_trick_guard: bool = True,
        featurizer=_DEFAULT_FEATURIZER,
    ) -> None:
        # Load each artifact from its path, then hand off to the shared
        # initialiser. `from_loaded` is the path-free entry point the serve
        # builder uses to share one already-loaded module across tier-views.
        # `featurizer` (default v6) selects the feature generation + load-guard
        # version — pass `featurizer_v5_frozen` to run a v5-trained export here.
        self._init_with_modules(
            load_policy_module(checkpoint_path, featurizer=featurizer),
            skill_decile=skill_decile,
            schupfen=self._maybe_load(schupfen_path, featurizer=featurizer),
            tichu_call=self._maybe_load(tichu_call_path, featurizer=featurizer),
            grand_call=self._maybe_load(grand_call_path, featurizer=featurizer),
            featurizer=featurizer,
            fallback_rng=fallback_rng,
            tichu_threshold=tichu_threshold,
            grand_threshold=grand_threshold,
            partner_trick_guard=partner_trick_guard,
        )

    @classmethod
    def from_loaded(
        cls,
        policy_module,
        *,
        skill_decile: int = _NEUTRAL_SKILL,
        schupfen=None,
        tichu_call=None,
        grand_call=None,
        fallback_rng: random.Random | None = None,
        tichu_threshold: float = 0.5,
        grand_threshold: float = 0.5,
        partner_trick_guard: bool = True,
        featurizer=_DEFAULT_FEATURIZER,
    ) -> "MLAgent":
        """Build an agent over already-loaded modules, skipping disk I/O.

        Lets the serve builder load one policy module (and the shared standalone
        nets) a single time and construct several tier-views over it that differ
        only by `skill_decile` — one Model in memory, not one per tier (ADR-0027).
        """
        self = cls.__new__(cls)
        self._init_with_modules(
            policy_module,
            skill_decile=skill_decile,
            schupfen=schupfen,
            tichu_call=tichu_call,
            grand_call=grand_call,
            featurizer=featurizer,
            fallback_rng=fallback_rng,
            tichu_threshold=tichu_threshold,
            grand_threshold=grand_threshold,
            partner_trick_guard=partner_trick_guard,
        )
        return self

    def _init_with_modules(
        self,
        policy_module,
        *,
        skill_decile: int,
        schupfen,
        tichu_call,
        grand_call,
        featurizer=_DEFAULT_FEATURIZER,
        fallback_rng: random.Random | None,
        tichu_threshold: float = 0.5,
        grand_threshold: float = 0.5,
        partner_trick_guard: bool = True,
    ) -> None:
        # Skill Embedding input fed to every network at inference. 0..9 are the
        # BSW skill deciles (9 = strongest players); 10 is the neutral/cold-start
        # row. Default is neutral per ADR-0005; pass `skill_decile=9` to condition
        # on top-tier play.
        if not 0 <= int(skill_decile) <= _NEUTRAL_SKILL:
            raise ValueError(
                f"skill_decile must be in [0, {_NEUTRAL_SKILL}], got {skill_decile}"
            )
        # Call decision threshold: call iff P(call) >= threshold. 0.5 == argmax
        # (the historical behaviour). Lower => call more (less conservative); a
        # per-tier lever for calibrating call rate / tournament EV. Tichu and
        # Grand-Tichu are independent networks, so each gets its own threshold.
        for name, t in (("tichu", tichu_threshold), ("grand", grand_threshold)):
            if not 0.0 < float(t) < 1.0:
                raise ValueError(f"{name}_threshold must be in (0, 1), got {t}")
        self._call_thresholds = {
            "tichu": float(tichu_threshold), "grand": float(grand_threshold),
        }
        self._skill_decile = int(skill_decile)
        self._fz = featurizer  # injected featurizer (default v6); see __init__.
        self._module = policy_module
        self._schupfen = schupfen
        self._tichu_call = tichu_call
        self._grand_call = grand_call
        self._rule_fallback = RuleAgent()
        self._rng = fallback_rng or random.Random(0)
        self._partner_trick_guard = bool(partner_trick_guard)
        self.last_fallback_used: bool = False

    @staticmethod
    def _maybe_load(path: str | Path | None, *, featurizer=_DEFAULT_FEATURIZER):
        if path is None:
            return None
        return load_standalone_net(path, featurizer=featurizer)

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
            # P(call) = softmax over [skip, call] = sigmoid(logit_call - logit_skip).
            # Call iff P(call) >= threshold; threshold 0.5 reproduces argmax.
            p_call = 1.0 / (1.0 + float(np.exp(logits[0] - logits[1])))
            return p_call >= self._call_thresholds[kind]
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
        choice = ranked[0]
        if self._partner_trick_guard and suppress_partner_trick_bomb(
            private_state, choice, legal
        ):
            return PASS
        return choice

    def _rank_play_actions(self, private_state: PrivateState) -> list[ConcreteAction]:
        legal = list(legal_actions_for(private_state))
        play_logits = self._play_logits(private_state)
        if not _is_finite(play_logits):
            raise RuntimeError("non-finite logits")
        ranked = _rank_legal_by_logits(legal, play_logits)
        return ranked or legal

    def play_action_scores(
        self, private_state: PrivateState
    ) -> list[tuple[ConcreteAction, float]]:
        """Legal Play actions with their policy probability (softmax over the
        legal set's logits), sorted descending. Diagnostic only (decision tapes
        / analysis) — NOT on the `act()` hot path. Unmapped actions get 0.0."""
        legal = list(legal_actions_for(private_state))
        if not legal:
            return []
        play_logits = self._play_logits(private_state)
        idxs = [_combination_to_action_index(a) for a in legal]
        scored = [
            (a, float(play_logits[i]))
            for a, i in zip(legal, idxs)
            if i is not None and 0 <= i < play_logits.shape[0]
        ]
        if not scored:
            return [(a, 0.0) for a in legal]
        logits = np.array([s for _, s in scored], dtype=np.float64)
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        out = [(a, float(p)) for (a, _), p in zip(scored, probs)]
        out.sort(key=lambda t: -t[1])
        return out

    def _play_logits(self, private_state: PrivateState) -> np.ndarray:
        out = self._policy_forward(private_state)
        return out["play"][0].detach().cpu().numpy()

    # ------------------------------------------------------------
    # Diagnostic score methods (decision tapes) for the non-Play Decisions —
    # NOT on the act() hot path.
    # ------------------------------------------------------------

    def wish_action_scores(self, private_state: PrivateState) -> list[tuple[ConcreteAction, float]]:
        """Legal Mahjong-Wish ranks with policy probability, sorted descending."""
        out = self._policy_forward(private_state)
        wish_logits = out["wish"][0].detach().cpu().numpy()
        legal = [a for a in legal_actions_for(private_state) if isinstance(a, MahjongWish)]
        return _softmax_scores([(a, float(wish_logits[wish_intent_index(a)])) for a in legal])

    def dragon_action_scores(self, private_state: PrivateState) -> list[tuple[ConcreteAction, float]]:
        """The two DragonGive targets with policy probability, sorted descending."""
        pending = private_state.public.pending_decision
        winner = getattr(pending, "winner", None)
        out = self._policy_forward(private_state)
        dl = out["dragon_assignment"][0].detach().cpu().numpy()
        legal = [a for a in legal_actions_for(private_state) if isinstance(a, DragonGive)]
        return _softmax_scores([(a, float(dl[dragon_intent_index(a, winner)])) for a in legal])

    def schupfen_action_scores(self, private_state: PrivateState) -> dict[str, list[tuple]]:
        """Per-direction ranked candidate cards (softmax over the Hand) for the
        Schupfen Network's three 56-way heads — {next/partner/previous: [(card, p)]}."""
        if self._schupfen is None:
            return {}
        features, skill = self._inputs(private_state)
        heads = self._schupfen(features, skill)
        hand = list(private_state.hand)
        out: dict[str, list[tuple]] = {}
        for name, head in zip(("next", "partner", "previous"), heads):
            row = head[0].detach().cpu().numpy()
            out[name] = _softmax_scores([(c, float(row[card_slot(c)])) for c in hand])
        return out

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
        features, skill = self._inputs(private_state)
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

    def _inputs(self, private_state: PrivateState) -> tuple[torch.Tensor, torch.Tensor]:
        private_state = _with_round_local_scores(private_state)
        features = torch.from_numpy(self._fz.featurize(private_state)).unsqueeze(0)
        skill = torch.tensor([self._skill_decile], dtype=torch.long)
        return features, skill

    def _policy_forward(self, private_state: PrivateState) -> dict:
        features, skill = self._inputs(private_state)
        return self._module(features, skill)

    def _run(self, module, private_state: PrivateState) -> np.ndarray:
        features, skill = self._inputs(private_state)
        out = module(features, skill)
        return out[0].detach().cpu().numpy()

    def _random_legal(self, private_state: PrivateState) -> ConcreteAction:
        legal = list(legal_actions_for(private_state))
        if not legal:
            raise RuntimeError("no legal actions for fallback")
        return self._rng.choice(legal)


def round_local_team_scores(public) -> tuple[int, int]:
    """The within-round team scores the nets were trained on.

    Every training round (BC replay, co-train rollout, the position pool) starts
    at `scores=(0, 0)`, and the engine credits each trick's points to BOTH
    `scores` and `round_points_by_player` from zero (engine.py). So the per-team
    sum of `round_points_by_player` IS the round-local team score — whereas the
    `scores` field arriving over /act is the GAME-cumulative total (0..1000+),
    which the nets never saw and react to spuriously. Team 0 = seats {0, 2},
    team 1 = seats {1, 3} (`_team_of` = seat % 2)."""
    rp = public.round_points_by_player
    return (rp[0] + rp[2], rp[1] + rp[3])


def _with_round_local_scores(private_state: PrivateState) -> PrivateState:
    """Return `private_state` with `public.scores` replaced by the round-local
    team score, so featurization matches the training regime (see
    `round_local_team_scores`)."""
    pub = private_state.public
    pub = dataclasses.replace(pub, scores=round_local_team_scores(pub))
    return dataclasses.replace(private_state, public=pub)


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


def _softmax_scores(pairs: list[tuple]) -> list[tuple]:
    """[(item, logit)] -> [(item, prob)] softmaxed over the set, sorted desc."""
    if not pairs:
        return []
    logits = np.array([s for _, s in pairs], dtype=np.float64)
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    out = [(item, float(p)) for (item, _), p in zip(pairs, probs)]
    out.sort(key=lambda t: -t[1])
    return out


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
