"""Replay a ParsedRound through the rules engine.

The replay function constructs the initial engine state from the parsed start
hands, runs schupfen, then steps through every parsed play / pass / wish /
dragon-give in order. At each step it checks whether the chosen engine action
is in the legal-action set. If not, it returns the first failure for the
caller to log into `known_bad_games.txt`.

Mid-round Tichu / Grand-Tichu calls are not yet engine actions — they are
captured by the parser and consumed at score time. The replay loop simply
passes them through without invoking the engine.

Phoenix substitution rank is inferred when ambiguous:
  - As a Single following another single: the engine's enumeration produces
    one Phoenix-following candidate of rank top+0.5 — uniquely determined.
  - Inside a straight: when multiple Phoenix positions produce the same card
    multiset, we pick the lowest-rank candidate (a conservative heuristic;
    full BSW logs occasionally encode the rank elsewhere).
"""

from dataclasses import dataclass, field
from typing import Iterable

from tichu_engine.cards import MAHJONG, PHOENIX
from tichu_engine.combinations import CardOrSpecial
from tichu_engine.engine import step
from tichu_engine.legality import (
    ConcreteAction,
    BombInterrupt,
    DragonGive,
    MahjongWish,
    PASS,
    Pass,
    SchupfenPass,
    _cards_in,
    legal_actions,
    legal_bomb_interrupts,
)
from tichu_engine.state import (
    DragonGivePending,
    GameState,
    MahjongWishPending,
    NUM_PLAYERS,
    PublicState,
    SchupfenPending,
    Trick,
)
from tichu_training.bsw.records import ParsedAction, ParsedRound


@dataclass
class ReplayResult:
    """Outcome of replaying one round through the engine."""

    final_state: GameState | None
    steps_taken: int
    illegal_action: ParsedAction | None = None
    illegal_reason: str | None = None
    # Decision records produced during replay (one per Parsed action that we
    # successfully fed to the engine, in execution order):
    decisions: list[tuple[ParsedAction, ConcreteAction]] = field(default_factory=list)
    # Engine state observed immediately before each decision was applied, in
    # lockstep with `decisions`. `None` for pseudo-decisions (Tichu/Grand-Tichu
    # call passthroughs and phantom passes) where no engine step happened.
    pre_decision_states: list[GameState | None] = field(default_factory=list)


def replay_round(parsed: ParsedRound) -> ReplayResult:
    """Step a `ParsedRound` through the engine. Stop at the first illegal action."""
    state = _build_initial_state(parsed)
    result = ReplayResult(final_state=None, steps_taken=0)

    # Round-scoped cache for legal_actions. Within one parsed action we hit
    # legal_actions(state) at 2-4 sites (validation, sync_for, match_play);
    # state changes only via step(), so id(state) is a sufficient key. We
    # anchor each cached state in `_anchors` to prevent CPython from reusing
    # its id() after GC — otherwise a freshly-allocated state could land at
    # the dropped state's address and inherit stale entries. Both fall out of
    # scope when this function returns. The full-enumeration cost was 86% of
    # pipeline runtime in profiling — see ADR-0009.
    cache: dict[int, frozenset] = {}
    _anchors: list[GameState] = []

    def legal(s: GameState) -> frozenset:
        actions = cache.get(id(s))
        if actions is None:
            actions = legal_actions(s)
            cache[id(s)] = actions
            _anchors.append(s)
        return actions

    # 1) Schupfen submissions (in seat order).
    for sub in parsed.schupfen:
        action = SchupfenPass(
            to_next=sub.schupfen_to_next,
            to_partner=sub.schupfen_to_partner,
            to_previous=sub.schupfen_to_previous,
        )
        if action not in legal(state):
            result.illegal_action = sub
            result.illegal_reason = "schupfen submission not legal"
            return result
        result.pre_decision_states.append(state)
        state, _, _, _ = step(state, action)
        result.steps_taken += 1
        result.decisions.append((sub, action))

    # 2) Play sequence.
    for parsed_action in parsed.plays:
        if parsed_action.kind in ("tichu", "grand_tichu"):
            # Engine doesn't model these yet — capture but skip stepping.
            result.decisions.append((parsed_action, _CallPassthrough(parsed_action.kind)))
            result.pre_decision_states.append(None)
            continue

        # Auto-decline a pending MahjongWish when the next BSW action is not
        # a "wish". BSW emits passes by other players BEFORE the optional
        # "Wunsch:" line, but the engine pins current_player to the wisher
        # until the wish is resolved — which would mis-classify those passes
        # as phantom. Decline first so passes can apply against the advanced
        # state. (Dragon-give is the opposite case: the passes between the
        # Dragon-trick and the "Drache an:" line ARE phantom and should be
        # filtered out first — handled below after the phantom-pass check.)
        if (
            isinstance(state.public.pending_decision, MahjongWishPending)
            and parsed_action.kind != "wish"
        ):
            state, _, _, _ = step(state, MahjongWish(rank=None))
            result.steps_taken += 1

        # BSW emits liberal `passt.` lines that don't correspond to engine
        # decisions — for already-passed players, for players whose turn it
        # isn't, and even trailing passes from the trick winner before they
        # lead the next trick. Treat a parsed pass as phantom if either the
        # player isn't the engine's current player, or PASS isn't legal.
        if parsed_action.kind == "pass" and (
            parsed_action.player != state.public.current_player
            or PASS not in legal(state)
        ):
            result.decisions.append((parsed_action, PASS))
            result.pre_decision_states.append(None)
            continue

        # BSW sometimes omits the "Drache an:" line after a Dragon-winning
        # trick — typically when the winner is about to play again
        # immediately. The engine is then stuck in DragonGivePending and
        # rejects the next play as "no legal combination matches". Synthesise
        # the missing DragonGive (defaulting to the left opponent, which is
        # the BSW majority preference at ~63% across logged dragon-gives) so
        # replay can proceed. Skipped when the next parsed action IS a
        # dragon_give (use the parsed target instead).
        while (
            isinstance(state.public.pending_decision, DragonGivePending)
            and parsed_action.kind != "dragon_give"
        ):
            winner = state.public.pending_decision.winner
            synthetic_target = (winner + 3) % NUM_PLAYERS  # left opponent
            synthetic = DragonGive(target=synthetic_target)
            state, _, _, _ = step(state, synthetic)
            result.steps_taken += 1

        # BSW often omits the trailing pass lines that precede a trick
        # resolution or pending decision. Insert synthetic PASSes to bring the
        # engine to the state where the parsed action can be applied.
        synced = _sync_for(parsed_action, state, legal_fn=legal)
        if synced is not None:
            state, n_inserted = synced
            result.steps_taken += n_inserted

        # An out-of-turn play after failed sync is a bomb interrupt.
        if (
            parsed_action.kind == "play"
            and parsed_action.player != state.public.current_player
        ):
            try:
                bomb_action = _match_bomb_interrupt(parsed_action, state)
            except ValueError as exc:
                result.illegal_action = parsed_action
                result.illegal_reason = str(exc)
                return result
            state_before_bomb = state
            state, _, _, _ = step(state, bomb_action)
            result.steps_taken += 1
            result.decisions.append((parsed_action, bomb_action))
            result.pre_decision_states.append(state_before_bomb)
            continue

        try:
            engine_action = _to_engine_action(parsed_action, state, legal_fn=legal)
        except ValueError as exc:
            result.illegal_action = parsed_action
            result.illegal_reason = str(exc)
            return result
        if engine_action not in legal(state):
            result.illegal_action = parsed_action
            result.illegal_reason = f"engine action {engine_action!r} not in legal set"
            return result
        result.pre_decision_states.append(state)
        state, _, _, _ = step(state, engine_action)
        result.steps_taken += 1
        result.decisions.append((parsed_action, engine_action))

    # End-of-round normalisation: BSW may end the round with a Mahjong-last-
    # card play and no following "Wunsch:" line, leaving the engine stuck in
    # MahjongWishPending. Auto-decline so the MahjongWish handler can finalise.
    if isinstance(state.public.pending_decision, MahjongWishPending):
        state, _, _, _ = step(state, MahjongWish(rank=None))
        result.steps_taken += 1

    result.final_state = state
    return result


@dataclass(frozen=True)
class _CallPassthrough:
    """Marker placed in `decisions` for Tichu/Grand-Tichu calls that weren't
    stepped through the engine — preserved for downstream training-record output."""

    kind: str


def _build_initial_state(parsed: ParsedRound) -> GameState:
    """Build a GameState whose hands are the round's start hands and whose
    pending decision is SchupfenPending — seat 0 acts first. Tichu / Grand-
    Tichu calls are pre-populated from the parsed round so end-of-round scoring
    can apply their effects."""
    public = PublicState(
        current_player=0,
        hand_sizes=tuple(len(h) for h in parsed.start_hands),  # type: ignore[arg-type]
        scores=(0, 0),
        trick=Trick.empty(),
        pending_decision=SchupfenPending(submitted=(None, None, None, None)),
        tichu_callers=parsed.tichu_callers,
        grand_tichu_callers=parsed.grand_tichu_callers,
    )
    return GameState(hands=parsed.start_hands, public=public)


def _to_engine_action(
    parsed: ParsedAction, state: GameState, *, legal_fn=legal_actions,
) -> ConcreteAction:
    """Convert a ParsedAction into the matching engine `ConcreteAction` for `state`."""
    if parsed.kind == "pass":
        return PASS
    if parsed.kind == "wish":
        return MahjongWish(rank=parsed.wish_rank)
    if parsed.kind == "dragon_give":
        if parsed.dragon_target is None:
            raise ValueError("dragon_give without target")
        return DragonGive(target=parsed.dragon_target)
    if parsed.kind == "play":
        return _match_play(parsed.cards, state, legal_fn=legal_fn)
    raise ValueError(f"cannot translate parsed kind {parsed.kind!r} to engine action")


def _sync_for(
    parsed: ParsedAction, state: GameState, *, max_steps: int = 4, legal_fn=legal_actions,
) -> tuple[GameState, int] | None:
    """Insert synthetic PASSes to bring the engine to a state where `parsed`
    can be applied.

    Conditions per kind:
      * `play`:        current_player == parsed.player AND pending_decision is None
      * `dragon_give`: pending_decision is DragonGivePending
      * `wish` / `pass`: no sync attempted (handled elsewhere)
    """
    def satisfied(s: GameState) -> bool:
        if parsed.kind == "play":
            return (
                s.public.current_player == parsed.player
                and s.public.pending_decision is None
            )
        if parsed.kind == "dragon_give":
            return isinstance(s.public.pending_decision, DragonGivePending)
        return True

    if satisfied(state):
        return state, 0

    inserted = 0
    while not satisfied(state):
        if inserted >= max_steps:
            return None
        # A pending Mahjong-wish with no upcoming `wish` parsed action means
        # the player declined; clear it with an explicit no-wish.
        if (
            isinstance(state.public.pending_decision, MahjongWishPending)
            and parsed.kind != "wish"
        ):
            state, _, _, _ = step(state, MahjongWish(rank=None))
            inserted += 1
            continue
        if PASS not in legal_fn(state):
            return None
        state, _, _, _ = step(state, PASS)
        inserted += 1
    return state, inserted


def _match_bomb_interrupt(parsed: ParsedAction, state: GameState) -> BombInterrupt:
    """Match a parsed out-of-turn play to a `BombInterrupt` action."""
    target = frozenset(parsed.cards)
    bombs = legal_bomb_interrupts(state, parsed.player)
    candidates = [b for b in bombs if frozenset(_cards_in(b)) == target]
    if not candidates:
        raise ValueError(
            f"out-of-turn play by seat {parsed.player} is not a legal bomb interrupt: "
            f"{sorted(map(repr, parsed.cards))}"
        )
    return BombInterrupt(player=parsed.player, bomb=candidates[0])


def _match_play(
    cards: tuple[CardOrSpecial, ...], state: GameState, *, legal_fn=legal_actions,
) -> ConcreteAction:
    """Find the engine combination from `legal_actions(state)` whose constituent
    cards match `cards` as a set. If multiple match (Phoenix-rank ambiguity),
    pick the lowest-rank candidate.
    """
    target = frozenset(cards)
    candidates = []
    for a in legal_fn(state):
        if isinstance(a, (Pass, MahjongWish, DragonGive, SchupfenPass, BombInterrupt)):
            continue
        if frozenset(_cards_in(a)) == target:  # type: ignore[arg-type]
            candidates.append(a)
    if not candidates:
        raise ValueError(f"no legal combination matches cards {sorted(map(repr, cards))}")
    if len(candidates) == 1:
        return candidates[0]
    # Disambiguate: prefer the lowest-rank candidate.
    def _key(a):
        return float(getattr(a, "rank", 0))
    candidates.sort(key=_key)
    return candidates[0]
