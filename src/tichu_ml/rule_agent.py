"""Rule-based baseline agent.

`RuleAgent` plays a deterministic, hand-coded strategy strong enough to beat
`RandomAgent` reliably without claiming to be strong in absolute terms. The
goal is a stable reference point for the tournament harness, not a competitor.

Heuristics:
  * Schupfen: give the three lowest cards in hand — lowest to next, then
    partner, then previous. Keeps strong cards for play.
  * Dragon-give: hand the trick to whichever opponent currently holds fewer
    cards (close to going out) — a mild defensive choice.
  * Mahjong-wish: decline (rank=None). Wish strategy is non-trivial; declining
    is safe and never illegal.
  * Leading: prefer the cheapest non-bomb combination. Save bombs and full
    houses for later in the round.
  * Following: play the cheapest non-bomb combination that beats the top, or
    pass. Bombs are held in reserve.
"""

from dataclasses import dataclass

from tichu_engine.cards import DRAGON, MAHJONG, PHOENIX, Card, SpecialCard
from tichu_engine.combinations import (
    FourOfAKindBomb,
    FullHouse,
    Pair,
    PairStep,
    Single,
    Straight,
    StraightFlushBomb,
    Triple,
)
from tichu_engine.legality import (
    PASS,
    Action,
    DragonGive,
    MahjongWish,
    Pass,
    SchupfenPass,
    legal_actions_for,
)
from tichu_engine.state import (
    DragonGivePending,
    MahjongWishPending,
    PrivateState,
    SchupfenPending,
)

from tichu_ml.agent import Agent
from tichu_ml.registry import register_agent


# Lower priority = preferred when leading or following.
_TYPE_PRIORITY: dict[type, int] = {
    Single: 0,
    Pair: 1,
    Straight: 2,
    PairStep: 2,
    Triple: 3,
    FullHouse: 4,
    FourOfAKindBomb: 100,
    StraightFlushBomb: 100,
}


def _card_low_key(card) -> float:
    """Sort key for individual cards — low cards first, specials placed where they fit.

    Mahjong is the lowest card; Dog is treated as low (you usually want to lead it
    to give partner the trick); Phoenix and Dragon are high.
    """
    if isinstance(card, SpecialCard):
        if card is MAHJONG:
            return 1
        if card.name == "dog":
            return 1.5
        if card is PHOENIX:
            return 14.5
        if card is DRAGON:
            return 15
    if isinstance(card, Card):
        return card.rank
    raise AssertionError(f"unexpected card: {card!r}")


def _combo_cost(combo) -> tuple[int, float]:
    """Sort key for combinations: (type priority, rank). Lower wins."""
    priority = _TYPE_PRIORITY.get(type(combo), 50)
    rank = getattr(combo, "rank", 0) or 0
    return (priority, rank)


@register_agent("rule")
class RuleAgent(Agent):
    def act(self, private_state: PrivateState) -> Action:
        legal = legal_actions_for(private_state)
        assert legal, "RuleAgent received empty legal-action set"

        pending = private_state.public.pending_decision
        if isinstance(pending, SchupfenPending):
            return self._schupfen(private_state, legal)
        if isinstance(pending, DragonGivePending):
            return self._dragon_give(private_state, legal)
        if isinstance(pending, MahjongWishPending):
            return self._mahjong_wish(legal)

        top = private_state.public.trick.top_combination
        if top is None:
            return self._lead(legal)
        return self._follow(legal)

    # ---- Pending decisions ----

    def _schupfen(self, private_state: PrivateState, legal) -> SchupfenPass:
        # Pass the three lowest cards. Lowest -> next, second-lowest -> partner,
        # third-lowest -> previous. This is a legal SchupfenPass by construction
        # (3 distinct cards from hand).
        hand_sorted = sorted(private_state.hand, key=_card_low_key)
        chosen = SchupfenPass(
            to_next=hand_sorted[0],
            to_partner=hand_sorted[1],
            to_previous=hand_sorted[2],
        )
        # `legal` is built by the same enumeration so `chosen` must be present.
        # If a future change breaks that invariant, fall back to any legal option
        # to avoid producing an illegal action.
        if chosen in legal:
            return chosen
        return next(iter(legal))

    def _dragon_give(self, private_state: PrivateState, legal) -> DragonGive:
        # Give the trick to whichever opponent has fewer remaining cards.
        opponents = [a for a in legal if isinstance(a, DragonGive)]
        hand_sizes = private_state.public.hand_sizes
        opponents.sort(key=lambda a: hand_sizes[a.target])
        return opponents[0]

    def _mahjong_wish(self, legal) -> MahjongWish:
        for action in legal:
            if isinstance(action, MahjongWish) and action.rank is None:
                return action
        return next(iter(legal))  # defensive fallback

    # ---- Play decisions ----

    def _lead(self, legal) -> Action:
        # Every action is a combination (no PASS when leading).
        candidates = [a for a in legal if not isinstance(a, Pass)]
        candidates.sort(key=_combo_cost)
        return candidates[0]

    def _follow(self, legal) -> Action:
        plays = [a for a in legal if not isinstance(a, Pass)]
        non_bombs = [
            a for a in plays
            if not isinstance(a, (FourOfAKindBomb, StraightFlushBomb))
        ]
        if non_bombs:
            non_bombs.sort(key=_combo_cost)
            return non_bombs[0]
        if PASS in legal:
            return PASS
        # No pass available means the Mahjong wish is forcing us to play —
        # use the cheapest legal play (which may be a bomb).
        plays.sort(key=_combo_cost)
        return plays[0]
