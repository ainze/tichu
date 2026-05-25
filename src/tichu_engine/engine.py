"""Tichu state-transition engine.

`step(state, action)` applies one action and returns the next state. It validates
that the action is legal, removes played cards from the actor's hand, advances
the turn (skipping players who have passed or who are out of cards), and resolves
the trick when only the leader remains.

Trick resolution at this stage only changes `current_player` and clears the
trick. Point scoring, Dog-to-partner, Dragon-give, Mahjong-wish declaration, and
out-of-turn bomb interrupts are handled in their own sub-slices.
"""

from dataclasses import replace

from tichu_engine.cards import Card, DOG, DRAGON, MAHJONG, PHOENIX, SpecialCard
from tichu_engine.combinations import CardOrSpecial, Single
from tichu_engine.legality import (
    ConcreteAction,
    BombInterrupt,
    Combination,
    DragonGive,
    MahjongWish,
    PASS,
    Pass,
    SchupfenPass,
    legal_actions,
    legal_bomb_interrupts,
)
from tichu_engine.legality import _cards_in
from tichu_engine.state import (
    DragonGivePending,
    GameState,
    MahjongWishPending,
    NUM_PLAYERS,
    Play,
    PublicState,
    SchupfenPending,
    Trick,
)


def step(state: GameState, action: ConcreteAction) -> tuple[GameState, float, bool, dict]:
    """Apply an action; return (next_state, reward, done, info).

    Raises ValueError if the action is not legal in the given state.
    """
    # BombInterrupt is validated separately because it comes from a non-current player.
    if isinstance(action, BombInterrupt):
        if action.bomb not in legal_bomb_interrupts(state, action.player):
            raise ValueError(f"BombInterrupt {action!r} is not legal in this state")
        return _apply_bomb_interrupt(state, action)

    if action not in legal_actions(state):
        raise ValueError(f"action {action!r} is not legal in this state")

    pending = state.public.pending_decision

    # Pending-decision actions short-circuit normal play.
    if isinstance(action, DragonGive):
        assert isinstance(pending, DragonGivePending)
        new_scores = _add_to_team(state.public.scores, _team_of(action.target), pending.points)
        new_round_points = _add_to_player(
            state.public.round_points_by_player, action.target, pending.points
        )
        next_public = replace(
            state.public,
            scores=new_scores,
            round_points_by_player=new_round_points,
            pending_decision=None,
        )
        new_state = GameState(hands=state.hands, public=next_public)
        if _round_done_state(new_state):
            new_state = _finalise_round(new_state)
        return new_state, 0.0, _round_done_state(new_state), {}

    if isinstance(action, SchupfenPass):
        assert isinstance(pending, SchupfenPending)
        return _apply_schupfen(state, action, pending)

    if isinstance(action, MahjongWish):
        assert isinstance(pending, MahjongWishPending)
        # Now advance the turn that was deferred by the Mahjong play.
        wisher = pending.player
        next_player, _ = _advance_or_resolve(
            trick=state.public.trick,
            hand_sizes=state.public.hand_sizes,
            current=wisher,
        )
        next_public = replace(
            state.public,
            current_player=next_player,
            mahjong_wish=action.rank,
            pending_decision=None,
        )
        return GameState(hands=state.hands, public=next_public), 0.0, _round_done(state.public.hand_sizes), {}

    current = state.public.current_player

    # Dog short-circuits normal trick mechanics: lead passes to partner, trick clears.
    if isinstance(action, Single) and action.card is DOG:
        next_hands = _remove_from_hand(state.hands, current, frozenset({DOG}))
        next_hand_sizes = tuple(len(h) for h in next_hands)
        partner = _partner_target(partner=(current + 2) % NUM_PLAYERS, hand_sizes=next_hand_sizes)  # type: ignore[arg-type]
        new_out_order = state.public.out_order
        if (
            state.public.hand_sizes[current] > 0
            and next_hand_sizes[current] == 0
            and current not in new_out_order
        ):
            new_out_order = new_out_order + (current,)
        next_public = replace(
            state.public,
            current_player=partner,
            hand_sizes=next_hand_sizes,  # type: ignore[arg-type]
            trick=Trick.empty(),
            out_order=new_out_order,
        )
        new_state = GameState(hands=next_hands, public=next_public)
        if _round_done_state(new_state):
            new_state = _finalise_round(new_state)
        return new_state, 0.0, _round_done_state(new_state), {}  # type: ignore[arg-type]

    if isinstance(action, Pass):
        next_trick = state.public.trick.add_pass(current)
        next_hands = state.hands
    else:
        played_cards = frozenset(_cards_in(action))
        next_hands = _remove_from_hand(state.hands, current, played_cards)
        next_trick = state.public.trick.add_play(player=current, combination=action)

    next_hand_sizes = tuple(len(h) for h in next_hands)

    # A play that contains Mahjong defers turn advancement: the player owes a
    # wish-rank declaration (or decline) before anyone else acts.
    if not isinstance(action, Pass) and MAHJONG in _cards_in(action):
        next_public = replace(
            state.public,
            hand_sizes=next_hand_sizes,  # type: ignore[arg-type]
            trick=next_trick,
            pending_decision=MahjongWishPending(player=current),
        )
        return GameState(hands=next_hands, public=next_public), 0.0, _round_done(next_hand_sizes), {}  # type: ignore[arg-type]

    # Advance turn or resolve trick.
    next_player, resolved_trick = _advance_or_resolve(
        trick=next_trick,
        hand_sizes=next_hand_sizes,  # type: ignore[arg-type]
        current=current,
    )

    # If the trick just resolved, award its points to the leader's team —
    # or defer the decision when the trick was won by the Dragon. The trick
    # "resolves" either via the normal pass-around (resolved_trick.leader is
    # None) or because the round ends mid-trick: when 3 of 4 players are out,
    # no further plays are possible and BSW credits the trick to whoever holds
    # the current top combination.
    new_scores = state.public.scores
    new_round_points = state.public.round_points_by_player
    new_pending: object | None = None
    trick_resolved = next_trick.leader is not None and resolved_trick.leader is None
    round_ends_mid_trick = (
        next_trick.leader is not None
        and resolved_trick.leader is not None
        and _round_done(next_hand_sizes)  # type: ignore[arg-type]
    )
    if trick_resolved or round_ends_mid_trick:
        winner = next_trick.leader
        points = _trick_points(next_trick)
        top = next_trick.top_combination
        if isinstance(top, Single) and top.card is DRAGON:
            new_pending = DragonGivePending(winner=winner, points=points)
        else:
            new_scores = _add_to_team(new_scores, _team_of(winner), points)
            new_round_points = _add_to_player(new_round_points, winner, points)
        if round_ends_mid_trick:
            resolved_trick = Trick.empty()

    # Clear an active Mahjong wish once any play fulfils it (contains a
    # natural card of the wished rank).
    new_wish = state.public.mahjong_wish
    if (
        not isinstance(action, Pass)
        and new_wish is not None
        and _play_fulfils_wish(action, new_wish)  # type: ignore[arg-type]
    ):
        new_wish = None

    # Track who has just gone out (hand became empty this step) so we can
    # apply slam / redistribution rules at end-of-round.
    new_out_order = state.public.out_order
    prev_sizes = state.public.hand_sizes
    for p in range(NUM_PLAYERS):
        if prev_sizes[p] > 0 and next_hand_sizes[p] == 0 and p not in new_out_order:
            new_out_order = new_out_order + (p,)

    next_public = replace(
        state.public,
        current_player=next_player,
        hand_sizes=next_hand_sizes,  # type: ignore[arg-type]
        trick=resolved_trick,
        scores=new_scores,
        round_points_by_player=new_round_points,
        out_order=new_out_order,
        mahjong_wish=new_wish,
        pending_decision=new_pending,  # type: ignore[arg-type]
    )
    new_state = GameState(hands=next_hands, public=next_public)
    if new_pending is None and _round_done_state(new_state):
        new_state = _finalise_round(new_state)
    return new_state, 0.0, _round_done_state(new_state), {}  # type: ignore[arg-type]


def _play_fulfils_wish(combo: Combination, wish_rank: int) -> bool:
    """True if `combo` contains a natural card of the wished rank (Phoenix
    substitution doesn't count)."""
    for c in _cards_in(combo):
        if isinstance(c, Card) and c.rank == wish_rank:
            return True
    return False


def _round_done(hand_sizes: tuple[int, int, int, int]) -> bool:
    """A round ends when 3 of the 4 players have emptied their hands (normal)."""
    return sum(1 for s in hand_sizes if s == 0) >= 3


def _round_done_state(state: GameState) -> bool:
    """A round ends on the normal condition, OR on a slam: the first two
    players to go out are partners."""
    if _round_done(state.public.hand_sizes):
        return True
    out_order = state.public.out_order
    if (
        len(out_order) >= 2
        and _team_of(out_order[0]) == _team_of(out_order[1])
    ):
        return True
    return False


def _team_of(player: int) -> int:
    """Players 0 and 2 form team 0; players 1 and 3 form team 1."""
    return player % 2


def _add_to_team(scores: tuple[int, int], team: int, points: int) -> tuple[int, int]:
    new = list(scores)
    new[team] += points
    return (new[0], new[1])


def _add_to_player(
    points: tuple[int, int, int, int], player: int, delta: int
) -> tuple[int, int, int, int]:
    new = list(points)
    new[player] += delta
    return (new[0], new[1], new[2], new[3])


def _finalise_round(state: GameState) -> GameState:
    """Apply end-of-round adjustments to scores.

    Rules:
      * Doppelsieg (slam): if the first two players to go out are on the same
        team, that team gets +200 instead of any card-point distribution. The
        mid-round card-point accumulation (which was already added to scores)
        is undone.
      * Otherwise, the last player still holding cards transfers:
          - their accumulated trick points to the first-out player's team
          - the point value of their remaining hand to the opposing team
      * Each Tichu caller: +100 if first-out, -100 otherwise.
      * Each Grand Tichu caller: +200 if first-out, -200 otherwise.

    Round-only state (round_points_by_player, out_order, tichu/grand_tichu
    callers) is reset for the next round.
    """
    public = state.public
    out_order = public.out_order
    first_out = out_order[0] if out_order else 0
    last_in_candidates = [p for p in range(NUM_PLAYERS) if state.hands[p]]
    last_in = last_in_candidates[0] if last_in_candidates else None

    scores = public.scores
    # Slam check: first two out are on the same team.
    is_slam = (
        len(out_order) >= 2
        and _team_of(out_order[0]) == _team_of(out_order[1])
    )
    if is_slam:
        # Undo card-point distribution (we credited mid-round to scores).
        round_team_points = [0, 0]
        for p in range(NUM_PLAYERS):
            round_team_points[_team_of(p)] += public.round_points_by_player[p]
        scores = (scores[0] - round_team_points[0], scores[1] - round_team_points[1])
        # Slam bonus.
        winning_team = _team_of(out_order[0])
        scores = _add_to_team(scores, winning_team, 200)
    else:
        if last_in is not None:
            # Last-in's collected tricks transfer to first-out's team.
            last_in_points = public.round_points_by_player[last_in]
            last_in_team = _team_of(last_in)
            first_out_team = _team_of(first_out)
            if last_in_team != first_out_team:
                scores = _add_to_team(scores, last_in_team, -last_in_points)
                scores = _add_to_team(scores, first_out_team, last_in_points)
            # Last-in's hand value transfers to opponent team.
            hand_points = sum(_card_value(c) for c in state.hands[last_in])
            opposing_team = 1 - last_in_team
            scores = _add_to_team(scores, opposing_team, hand_points)

    # Tichu / Grand Tichu effects.
    for caller in public.tichu_callers:
        bonus = 100 if caller == first_out else -100
        scores = _add_to_team(scores, _team_of(caller), bonus)
    for caller in public.grand_tichu_callers:
        bonus = 200 if caller == first_out else -200
        scores = _add_to_team(scores, _team_of(caller), bonus)

    next_public = replace(
        public,
        scores=scores,
        round_points_by_player=(0, 0, 0, 0),
        out_order=(),
        tichu_callers=frozenset(),
        grand_tichu_callers=frozenset(),
    )
    return GameState(hands=state.hands, public=next_public)


_CARD_RANK_POINTS = {5: 5, 10: 10, 13: 10}
_SPECIAL_POINTS = {DRAGON: 25, PHOENIX: -25}


def _card_value(card: CardOrSpecial) -> int:
    if isinstance(card, SpecialCard):
        return _SPECIAL_POINTS.get(card, 0)
    return _CARD_RANK_POINTS.get(card.rank, 0)


def _trick_points(trick: Trick) -> int:
    from tichu_engine.legality import _cards_in  # local import to avoid cycle

    total = 0
    for play in trick.plays:
        for c in _cards_in(play.combination):  # type: ignore[arg-type]
            total += _card_value(c)
    return total


def _apply_schupfen(
    state: GameState, action: SchupfenPass, pending: SchupfenPending
) -> tuple[GameState, float, bool, dict]:
    """Record one player's schupfen pass. When all four have submitted, exchange
    the cards and set the current player to the Mahjong holder."""
    current = state.public.current_player
    cards = (action.to_next, action.to_partner, action.to_previous)
    new_submitted = list(pending.submitted)
    new_submitted[current] = cards

    if all(s is not None for s in new_submitted):
        new_hands = [set(h) for h in state.hands]
        for sender in range(NUM_PLAYERS):
            sent = new_submitted[sender]
            for c in sent:
                new_hands[sender].discard(c)
        for sender in range(NUM_PLAYERS):
            sent = new_submitted[sender]
            for offset, c in zip((1, 2, 3), sent):
                new_hands[(sender + offset) % NUM_PLAYERS].add(c)
        frozen_hands = tuple(frozenset(h) for h in new_hands)
        starting = next(p for p, h in enumerate(frozen_hands) if MAHJONG in h)
        next_public = replace(
            state.public,
            current_player=starting,
            hand_sizes=tuple(len(h) for h in frozen_hands),  # type: ignore[arg-type]
            pending_decision=None,
        )
        return GameState(hands=frozen_hands, public=next_public), 0.0, False, {}

    next_public = replace(
        state.public,
        current_player=(current + 1) % NUM_PLAYERS,
        pending_decision=SchupfenPending(submitted=tuple(new_submitted)),
    )
    return GameState(hands=state.hands, public=next_public), 0.0, False, {}


def _apply_bomb_interrupt(state: GameState, action: BombInterrupt) -> tuple[GameState, float, bool, dict]:
    """Apply an out-of-turn bomb. The bomber becomes the new trick leader; previously
    passed players are cleared (everyone gets to react to the new top); turn advances
    clockwise from the bomber."""
    played_cards = frozenset(_cards_in(action.bomb))  # type: ignore[arg-type]
    next_hands = _remove_from_hand(state.hands, action.player, played_cards)
    next_hand_sizes = tuple(len(h) for h in next_hands)
    bombed_trick = Trick(
        plays=state.public.trick.plays + (Play(player=action.player, combination=action.bomb),),
        leader=action.player,
        passes=frozenset(),  # reset: everyone may react to the new top
    )
    next_player, resolved_trick = _advance_or_resolve(
        trick=bombed_trick,
        hand_sizes=next_hand_sizes,  # type: ignore[arg-type]
        current=action.player,
    )
    new_scores = state.public.scores
    new_round_points = state.public.round_points_by_player
    new_pending: object | None = None
    trick_resolved = bombed_trick.leader is not None and resolved_trick.leader is None
    round_ends_mid_trick = (
        bombed_trick.leader is not None
        and resolved_trick.leader is not None
        and _round_done(next_hand_sizes)  # type: ignore[arg-type]
    )
    if trick_resolved or round_ends_mid_trick:
        winner = bombed_trick.leader
        points = _trick_points(bombed_trick)
        new_scores = _add_to_team(new_scores, _team_of(winner), points)
        new_round_points = _add_to_player(new_round_points, winner, points)
        if round_ends_mid_trick:
            resolved_trick = Trick.empty()

    new_out_order = state.public.out_order
    prev_sizes = state.public.hand_sizes
    for p in range(NUM_PLAYERS):
        if prev_sizes[p] > 0 and next_hand_sizes[p] == 0 and p not in new_out_order:
            new_out_order = new_out_order + (p,)

    next_public = replace(
        state.public,
        current_player=next_player,
        hand_sizes=next_hand_sizes,  # type: ignore[arg-type]
        trick=resolved_trick,
        scores=new_scores,
        round_points_by_player=new_round_points,
        out_order=new_out_order,
        pending_decision=new_pending,  # type: ignore[arg-type]
    )
    new_state = GameState(hands=next_hands, public=next_public)
    if new_pending is None and _round_done_state(new_state):
        new_state = _finalise_round(new_state)
    return new_state, 0.0, _round_done_state(new_state), {}  # type: ignore[arg-type]


def _partner_target(partner: int, hand_sizes: tuple[int, int, int, int]) -> int:
    """When the Dog passes lead to partner: partner if they hold cards, else the
    next player clockwise who does."""
    if hand_sizes[partner] > 0:
        return partner
    for offset in range(1, NUM_PLAYERS):
        p = (partner + offset) % NUM_PLAYERS
        if hand_sizes[p] > 0:
            return p
    return partner  # defensive: should never happen unless all hands are empty


def _remove_from_hand(
    hands: tuple[frozenset[CardOrSpecial], ...],
    player: int,
    cards: frozenset[CardOrSpecial],
) -> tuple[frozenset[CardOrSpecial], ...]:
    new_hands = list(hands)
    new_hands[player] = hands[player] - cards
    return tuple(new_hands)


def _advance_or_resolve(
    trick: Trick,
    hand_sizes: tuple[int, int, int, int],
    current: int,
) -> tuple[int, Trick]:
    """Advance to the next eligible player, or resolve the trick.

    A trick resolves when, after passing the turn forward, the next player would
    be the leader (every other in-trick player has passed). In that case the
    trick clears and play continues from the leader.
    """
    leader = trick.leader
    # If no leader yet (empty trick, shouldn't happen after a play/pass), just advance.
    if leader is None:
        return (current + 1) % NUM_PLAYERS, trick

    # Walk forward from current+1, skipping passed-out and empty-hand players.
    for offset in range(1, NUM_PLAYERS + 1):
        p = (current + offset) % NUM_PLAYERS
        if p == leader:
            # All others have passed — trick resolves. If the leader is out of
            # cards (they won the trick by going out), pass the lead to the next
            # clockwise player who still has cards.
            return _next_active_from(leader, hand_sizes), Trick.empty()
        if p in trick.passes:
            continue
        if hand_sizes[p] == 0:
            continue
        return p, trick
    # Defensive fallback: should never reach here because the loop always hits the leader.
    return _next_active_from(leader, hand_sizes), Trick.empty()


def _next_active_from(start: int, hand_sizes: tuple[int, int, int, int]) -> int:
    """Return `start` if they have cards, else the next clockwise player who does.
    If every player is out, returns `start` — the round-end check then fires."""
    if hand_sizes[start] > 0:
        return start
    for offset in range(1, NUM_PLAYERS):
        p = (start + offset) % NUM_PLAYERS
        if hand_sizes[p] > 0:
            return p
    return start
