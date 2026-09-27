"""Heuristic guardrail probes — premise tests for the deep-research Tichu rules (2026-06-08).

A deep-research pass over the Tichu strategy literature (Fuegi BU guide, BGA Tips,
Spotlight on Games, hfog, Next Level Card Games) surfaced a handful of widely-agreed
"clear mistakes to avoid". Before spending reward-shaping budget on any of them, test
the *premise* the same way `forced_press` (ADR-0031) did: wrap the master, force the
heuristic's prescribed action in EXACTLY the states it applies to, and tournament the
wrapped agent against the bare master. Because both sides load the identical policy,
any score delta is purely the intervention:

  * probe EV > master (CI clears 0)  -> the master makes this mistake; the heuristic is
    free EV and worth a guardrail / reward nudge;
  * probe EV ~= master (CI spans 0)  -> the master already plays this rule (or it is
    EV-neutral); do NOT add a guardrail — it buys nothing;
  * probe EV < master               -> the "rule" is wrong for this policy (master's
    learned play beats the heuristic); leave it alone.

Each probe also counts how often its trigger fired (`interventions`), so a ~0 EV delta
is interpretable: fired-often-but-neutral (master handles it) vs never-fired (no
opportunities in the pool). Interventions are read in-process by `scripts/probe_heuristics.py`;
the spawned tournament workers rebuild fresh agents, so their counters do not return.

The five probes, mapped to the surviving high-confidence findings:

  forced_split_aces   -- "split Aces": a master-led NATURAL pair of Aces becomes a single
                         Ace lead (a pair spends two leads' worth of Aces for one lead).
  forced_follow_low   -- "don't waste high cards if you can follow low": when following,
                         if the master beats with a premium (Dragon/Phoenix) or an
                         over-high natural card while a CHEAPER natural beat is legal,
                         force the cheapest natural beat (the ladder principle).
  forced_support_tichu-- "support a called partner; don't overtake": when the partner has
                         called (Grand-)Tichu and currently holds the trick top, force a
                         Pass instead of overtaking their winning trick.
  forced_keep_partner_trick -- "don't bomb a trick your own team already wins": when the
                         master would bomb while the PARTNER holds the trick top, force a
                         Pass (the bomb would only steal a lead from the partner).
  partner_trick_guard  -- the RETIRED served guard (`MLAgent`, 2026-06-09 .. 2026-09-27):
                         forced_keep_partner_trick plus the carve-out that a caller
                         going OUT on the bomb keeps it. Removed from serving after the
                         v7 A/B (`scripts/ab_partner_trick_guard.py`) found it worth
                         +0.04/Round [-0.05, +0.16]; kept here so that A/B reproduces.
  forced_dragon_lastout-- Dragon trick-give: override the master's target with the simple
                         expert heuristic — give to the opponent expected to go out LAST
                         (the one holding more cards; tie-break to the player on the right,
                         the BGA default).

NOTE on direction. `forced_follow_low`, `forced_split_aces`, and the Dragon probe are
SUBSTITUTIONS (always re-route the master's choice in-trigger), so they also fire in the
minority of positions where the blanket rule is wrong (a monster hand wants paired Aces;
sometimes over-beating to hold the lead is correct). That is intended: the tournament
measures the NET EV of the rule as a blanket nudge, exactly the lever a reward shaper
would pull. `forced_support_tichu` / `forced_keep_partner_trick` are SUPPRESSIONS that
only ever convert an overtake/bomb into a Pass, so they can only fire on the mistake.
"""

from tichu_engine.cards import DRAGON, PHOENIX, Card
from tichu_engine.combinations import (
    FourOfAKindBomb,
    Pair,
    Single,
    StraightFlushBomb,
)
from tichu_engine.legality import (
    DragonGive,
    Pass,
    PASS,
    _cards_in,
    legal_actions_for,
)
from tichu_engine.state import DragonGivePending
from tichu_ml.agent import Agent
from tichu_ml.registry import register_agent

_BOMB_TYPES = (FourOfAKindBomb, StraightFlushBomb)
_ACE_RANK = 14


def _contains_premium(combo) -> bool:
    """True iff the combination spends a premium special (Dragon or Phoenix)."""
    return any(c is DRAGON or c is PHOENIX for c in _cards_in(combo))


class _MasterProbe(Agent):
    """Shared scaffold: wrap an MLAgent master, expose the same call/rank surface, and
    count trigger fires. Subclasses override `act` and bump `self.interventions` when
    they re-route the master. `from_policy` builds one over a bare policy for tests
    (no torch / no checkpoints)."""

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
        self.interventions = 0

    @classmethod
    def from_policy(cls, policy) -> "_MasterProbe":
        self = cls.__new__(cls)
        self._policy = policy
        self.interventions = 0
        return self

    def should_call(self, private_state, kind: str) -> bool:
        return self._policy.should_call(private_state, kind)

    def rank_actions(self, private_state):
        return self._policy.rank_actions(private_state)

    def act(self, private_state):  # pragma: no cover - abstract
        raise NotImplementedError


@register_agent("forced_split_aces")
class ForcedSplitAcesAgent(_MasterProbe):
    """Finding: split Aces. When the master LEADS a natural pair of Aces, lead a single
    Ace instead (keeping the second Ace for a second lead). Only natural Ace pairs are
    split — an Ace+Phoenix "pair" is left alone (splitting it raises the question of
    where the Phoenix goes, outside this rule's scope)."""

    def act(self, private_state):
        action = self._policy.act(private_state)
        if (
            private_state.public.trick.top_combination is None
            and isinstance(action, Pair)
            and all(isinstance(c, Card) and c.rank == _ACE_RANK for c in (action.a, action.b))
        ):
            self.interventions += 1
            return Single(action.a)
        return action


@register_agent("forced_follow_low")
class ForcedFollowLowAgent(_MasterProbe):
    """Finding: don't waste high cards if you can follow low. When FOLLOWING, if the
    master's chosen beat is a premium (Dragon/Phoenix) or a higher natural card while a
    strictly cheaper NATURAL beat is legal, force the cheapest natural beat. Bombs are
    left alone (deploying a bomb is a separate decision, covered by other probes)."""

    def act(self, private_state):
        action = self._policy.act(private_state)
        # Only a normal following Play Decision. A pending decision (Mahjong wish,
        # Dragon-give, Schupfen) exposes non-combination legal actions with no cards
        # to ladder over — leave the master's choice untouched.
        if private_state.public.pending_decision is not None:
            return action
        top = private_state.public.trick.top_combination
        if top is None or isinstance(action, (Pass, *_BOMB_TYPES)):
            return action
        cheapest = self._cheapest_natural_beat(private_state)
        if cheapest is not None and self._cost(action) > self._cost(cheapest):
            self.interventions += 1
            return cheapest
        return action

    @staticmethod
    def _cost(combo):
        # Premium beats are the most wasteful regardless of rank; otherwise cheaper = lower rank.
        return (1, 0.0) if _contains_premium(combo) else (0, float(combo.rank))

    def _cheapest_natural_beat(self, pv):
        best = None
        best_cost = None
        for a in legal_actions_for(pv):
            if isinstance(a, (Pass, *_BOMB_TYPES)) or _contains_premium(a):
                continue
            cost = self._cost(a)
            if best_cost is None or cost < best_cost:
                best, best_cost = a, cost
        return best


@register_agent("forced_support_tichu")
class ForcedSupportTichuAgent(_MasterProbe):
    """Finding: support a called partner; don't overtake their winning trick. When the
    partner has called (Grand-)Tichu and currently holds the trick top, and the master
    would play any non-Pass (overtaking the partner), force a Pass instead — provided
    Pass is legal (a Mahjong wish can forbid it)."""

    def act(self, private_state):
        action = self._policy.act(private_state)
        pub = private_state.public
        if pub.pending_decision is not None or isinstance(action, Pass):
            return action
        partner = (private_state.player + 2) % 4
        if (
            (partner in pub.tichu_callers or partner in pub.grand_tichu_callers)
            and pub.trick.leader == partner
            and any(isinstance(a, Pass) for a in legal_actions_for(private_state))
        ):
            self.interventions += 1
            return PASS
        return action


@register_agent("forced_keep_partner_trick")
class ForcedKeepPartnerTrickAgent(_MasterProbe):
    """Finding: never bomb a trick your own team already wins. When the master would
    play a bomb while the PARTNER holds the trick top, force a Pass instead (the bomb
    would only steal the lead from the partner). Bombing an OPPONENT's top is left
    untouched — that is the standard, correct use of a bomb."""

    def act(self, private_state):
        action = self._policy.act(private_state)
        if not isinstance(action, _BOMB_TYPES):
            return action
        partner = (private_state.player + 2) % 4
        if private_state.public.trick.leader == partner and any(
            isinstance(a, Pass) for a in legal_actions_for(private_state)
        ):
            self.interventions += 1
            return PASS
        return action


def suppress_partner_trick_bomb(private_state, action, legal) -> bool:
    """The retired served guard's trigger: True iff `action` bombs a trick the
    agent's own partner already tops and a Pass is legal. Carve-out: a
    (Grand-)Tichu caller going OUT on that bomb keeps it."""
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


@register_agent("partner_trick_guard")
class PartnerTrickGuardAgent(_MasterProbe):
    """The retired served guard, as a probe: the master with every bomb over the
    partner's winning trick turned into a Pass (caller-going-out carve-out kept)."""

    def act(self, private_state):
        action = self._policy.act(private_state)
        if suppress_partner_trick_bomb(private_state, action,
                                       list(legal_actions_for(private_state))):
            self.interventions += 1
            return PASS
        return action


@register_agent("forced_dragon_lastout")
class ForcedDragonLastoutAgent(_MasterProbe):
    """Finding: give the Dragon trick to the opponent who goes out LAST. Override the
    master's Dragon-give target with the simple expert heuristic — the opponent holding
    more cards (expected out last), tie-breaking to the player on the winner's right
    (the BGA `pass-right` default). All other decisions pass through to the master."""

    def act(self, private_state):
        pending = private_state.public.pending_decision
        if isinstance(pending, DragonGivePending):
            self.interventions += 1
            return DragonGive(target=self._last_out_opponent(private_state, pending.winner))
        return self._policy.act(private_state)

    @staticmethod
    def _last_out_opponent(pv, winner: int) -> int:
        hand_sizes = pv.public.hand_sizes
        right = (winner - 1) % 4  # the opponent seated immediately to the winner's right
        opponents = [p for p in range(4) if p % 2 != winner % 2]
        # Larger hand => expected out last; tie-break True (== right) beats False.
        return max(opponents, key=lambda p: (hand_sizes[p], p == right))
