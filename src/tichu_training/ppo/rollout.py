"""Self-play rollout driver for PPO Refine (ADR-0029).

Drives self-play Rounds through the rules engine under an injected play-policy
and emits one `Trajectory` per learner-team seat, each carrying the terminal
`round_outcome` reward (team-relative, call-bonus-inclusive — the bonus is
folded into `scores` by `_finalise_round` when callers are set on the state).

Only the *Play* Decision is routed to the learning policy; the rare non-Play
Decisions (Schupfen / Wish / Dragon) are handled inline by a frozen baseline
agent, and Calls are out of PPO scope. The driver records steps only for the
two learner-team seats (parameter-shared partner: both are on-policy for the
shared policy; opponents are environment dynamics).

This module is single-game for now; vectorized many-game stepping is a later
slice that preserves this public interface.
"""

from dataclasses import dataclass, field, replace
from typing import NamedTuple, Protocol, Sequence

from tichu_engine.engine import step
from tichu_engine.legality import ConcreteAction, Pass
from tichu_engine.state import (
    GameState, MahjongWishPending, PublicState, SchupfenPending, Trick,
)
from tichu_ml.agent import Agent
from tichu_ml.rule_agent import RuleAgent

_NUM_PLAYERS = 4
_MAX_STEPS = 10_000  # safety net against an infinite-loop bug.


class PlayChoice(NamedTuple):
    """A play-policy's decision at one Play Decision.

    `concrete_action` is what the engine steps. The remaining fields are the
    PPO bookkeeping the learner records for its own seats; a non-learning policy
    (e.g. a frozen opponent or a deterministic fake) leaves them `None`.
    """

    concrete_action: ConcreteAction
    intent_index: int | None = None
    logprob: float | None = None
    value: float | None = None
    features: object | None = None    # observable featurized state (policy re-forward)
    legal_mask: object | None = None  # legal-Intent mask at this decision
    critic_features: object | None = None  # perfect-info features (asymmetric critic;
    #                                        None => symmetric, value uses `features`)


class SchupfenChoice(NamedTuple):
    """A schupfen-policy's decision at one Schupfen Decision (ADR-0034).

    `concrete_action` is the `SchupfenPass` the engine steps. The rest is the PPO
    bookkeeping recorded for a learner seat: `card_indices` is the sampled
    `(to_next, to_partner, to_previous)` card-slot triple (the without-replacement
    action encoding), `legal_masks` the per-head hand masks. A non-learning policy
    leaves the bookkeeping `None`.
    """

    concrete_action: ConcreteAction
    card_indices: tuple[int, int, int] | None = None
    logprob: float | None = None
    value: float | None = None
    features: object | None = None
    legal_masks: object | None = None
    critic_features: object | None = None


class WishChoice(NamedTuple):
    """A wish-policy's decision at one Mahjong-Wish Decision (ADR-0034 addendum).

    `concrete_action` is the `MahjongWish` the engine steps. The rest is the PPO
    bookkeeping recorded for a learner seat: `intent_index` is the wish-head index
    (0 == decline, 1..13 == ranks 2..14), the action the 14-way wish head emits.
    There is no legal mask — every wish slot is legal during `MahjongWishPending`.
    A non-learning policy leaves the bookkeeping `None`. The wish head lives on the
    play net's trunk (option (a)); the collector treats it as its own decision type."""

    concrete_action: ConcreteAction
    intent_index: int | None = None
    logprob: float | None = None
    value: float | None = None
    features: object | None = None
    critic_features: object | None = None


class CallChoice(NamedTuple):
    """A call-policy's decision at one Tichu / Grand-Tichu Call Decision (ADR-0034).

    Calls are injected into the state (the engine does not step them — it reads
    `tichu_callers` / `grand_tichu_callers` at `_finalise_round`), so there is no
    `concrete_action`: `called` is the binary action. The rest is the PPO
    bookkeeping recorded for a learner seat (`called` doubles as the action index:
    1 == call, 0 == skip)."""

    called: bool
    logprob: float | None = None
    value: float | None = None
    features: object | None = None
    critic_features: object | None = None


class RolloutPolicy(Protocol):
    """The policy bundle injected into the driver.

    `act_play_batch` receives all Play Decisions live across the concurrent
    games at one tick — a list of `(seat, private_state, game_state)` — and
    returns one `PlayChoice` per decision, in the same order. The `game_state`
    carries all four hands for an asymmetric Perfect-Info Critic (ADR-0033);
    symmetric policies ignore it. Batching the forward over many games at once
    is the throughput design of ADR-0029.

    A full-stack co-training policy (ADR-0034) MAY additionally implement
    `act_schupfen_batch` (same `(seat, private, game_state)` batch shape,
    returning `SchupfenChoice`s) so its Schupfen Network is driven + recorded;
    a play-only policy omits it and Schupfen falls back to the frozen seat agent.
    """

    def act_play_batch(
        self, decisions: list[tuple[int, object, object]]
    ) -> list[PlayChoice]: ...


class TrajectoryStep(NamedTuple):
    # `intent_index` is an int for Play / Call decisions, or the
    # `(to_next, to_partner, to_previous)` card-slot triple for Schupfen.
    # `decision_type` routes the step to its net + frozen-BC anchor in the update
    # (ADR-0034); it defaults to "play" so existing play records are unchanged.
    intent_index: object | None
    logprob: float | None
    value: float | None
    features: object | None = None
    legal_mask: object | None = None
    critic_features: object | None = None
    decision_type: str = "play"


@dataclass
class Trajectory:
    """One learner-team seat's recorded experience for a single Round."""

    seat: int
    team: int
    reward: float = 0.0
    steps: list[TrajectoryStep] = field(default_factory=list)


@dataclass
class _PlayGroup:
    """One serving model's batched Play Decisions for a single tick."""

    batch: list[tuple[int, object, object]] = field(default_factory=list)  # (seat, private, game_state)
    meta: list = field(default_factory=list)                               # (run, seat) parallel to batch


@dataclass
class _GameRun:
    """Mutable per-game state threaded through the vectorized drive loop."""

    state: GameState
    seat_agents: Sequence[Agent]
    learner_team: int
    initial_scores: tuple[int, int]
    trajs: dict[int, Trajectory]
    done: bool = False
    asked_tichu: set[int] = field(default_factory=set)  # seats already solicited for Tichu


def collect_rollout(
    positions: Sequence,
    policy: RolloutPolicy,
    *,
    opponent_policy: RolloutPolicy | None = None,
    learner_team: int = 0,
    seat_agents: Sequence[Agent] | None = None,
) -> list[Trajectory]:
    """Roll out the Starting Positions concurrently under `policy`, returning one
    Trajectory per learner-team seat per Round.

    The games advance in lockstep: at each tick every live game contributes its
    current Decision. Play Decisions are split by team and batched into one
    `act_play_batch` call per serving model — the learner-team seats go to
    `policy`, the opposing seats to `opponent_policy` (the league / frozen-BC
    opponent; defaults to `policy` for pure self-play). The rare non-Play
    Decisions (Schupfen / Wish / Dragon) plus Calls are served inline by the
    four frozen per-seat `seat_agents` (default: four RuleAgents, which decline
    every call).
    """
    if opponent_policy is None:
        opponent_policy = policy
    if seat_agents is None:
        seat_agents = [RuleAgent() for _ in range(_NUM_PLAYERS)]
    if len(seat_agents) != _NUM_PLAYERS:
        raise ValueError(f"expected {_NUM_PLAYERS} seat_agents, got {len(seat_agents)}")
    learner_seats = [s for s in range(_NUM_PLAYERS) if s % 2 == learner_team]

    runs: list[_GameRun] = []
    grand_prefixes: list = []
    for pos in positions:
        runs.append(
            _GameRun(
                state=pos.state,
                seat_agents=seat_agents,
                learner_team=learner_team,
                initial_scores=pos.state.public.scores,
                trajs={
                    seat: Trajectory(seat=seat, team=learner_team)
                    for seat in learner_seats
                },
            )
        )
        grand_prefixes.append(pos.grand_prefixes)

    # Grand-Tichu is decided at deal time on the synthetic (8,8,8,8) prefix state.
    # A call-capable model decides + records it through `act_call_batch`; a play-only
    # model falls back to the frozen seat agent's `should_call` (ADR-0029 behavior).
    _solicit_grand(runs, grand_prefixes, policy, opponent_policy, learner_team)

    _drive(runs, policy, opponent_policy, learner_team)

    out: list[Trajectory] = []
    for run in runs:
        out.extend(run.trajs.values())
    return out


def _drive(
    runs: list[_GameRun],
    policy: RolloutPolicy,
    opponent_policy: RolloutPolicy,
    learner_team: int,
) -> None:
    # Each live game advances exactly one Decision per tick, so the longest game
    # bounds the tick count.
    for _ in range(_MAX_STEPS):
        if all(run.done for run in runs):
            break
        # Play Decisions are grouped by serving model so each model runs a single
        # batched forward per tick (ADR-0029). learner-team seats -> `policy`,
        # opposing seats -> `opponent_policy`.
        groups: dict[RolloutPolicy, _PlayGroup] = {
            policy: _PlayGroup(), opponent_policy: _PlayGroup(),
        }
        # Schupfen Decisions are batched per serving model too (ADR-0034). A model
        # without `act_schupfen_batch` (a play-only policy) leaves Schupfen on the
        # frozen inline seat agent — the ADR-0029 behavior.
        schupfen_groups: dict[RolloutPolicy, _PlayGroup] = {
            policy: _PlayGroup(), opponent_policy: _PlayGroup(),
        }
        # Mahjong-Wish Decisions are batched per serving model too (ADR-0034 addendum).
        # A model without `act_wish_batch` leaves the wish on the frozen inline seat
        # agent (which declines) — the ADR-0029 behavior.
        wish_groups: dict[RolloutPolicy, _PlayGroup] = {
            policy: _PlayGroup(), opponent_policy: _PlayGroup(),
        }
        for run in runs:
            if run.done:
                continue
            current = run.state.public.current_player
            private = run.state.private_view(current)
            model = policy if current % 2 == learner_team else opponent_policy
            pending = run.state.public.pending_decision
            if pending is None:  # a Play Decision
                group = groups[model]
                group.meta.append((run, current))
                # (seat, private_state, game_state): the GameState carries all four
                # hands for an asymmetric Perfect-Info Critic (ADR-0033); symmetric
                # policies ignore it.
                group.batch.append((current, private, run.state))
            elif isinstance(pending, SchupfenPending) and _supports_schupfen(model):
                group = schupfen_groups[model]
                group.meta.append((run, current))
                group.batch.append((current, private, run.state))
            elif isinstance(pending, MahjongWishPending) and _supports_wish(model):
                group = wish_groups[model]
                group.meta.append((run, current))
                group.batch.append((current, private, run.state))
            else:  # Dragon, or Wish/Schupfen for a model without that head — inline
                _advance(run, run.seat_agents[current].act(private))
        for model, group in groups.items():
            if not group.batch:
                continue
            choices = model.act_play_batch(group.batch)
            for (run, seat), choice in zip(group.meta, choices):
                if seat in run.trajs:
                    run.trajs[seat].steps.append(
                        TrajectoryStep(
                            choice.intent_index, choice.logprob, choice.value,
                            choice.features, choice.legal_mask, choice.critic_features,
                        )
                    )
                # Tichu Call: solicited once per seat at its first non-Pass Play
                # (ADR-0018). A call-capable model decides + records it via
                # `act_call_batch`; a play-only model falls back to `should_call`. The
                # +/-100 bonus lands in `round_outcome` either way.
                _maybe_solicit_tichu(run, seat, choice.concrete_action, model)
                _advance(run, choice.concrete_action)
        for model, group in schupfen_groups.items():
            if not group.batch:
                continue
            choices = model.act_schupfen_batch(group.batch)
            for (run, seat), choice in zip(group.meta, choices):
                if seat in run.trajs:
                    run.trajs[seat].steps.append(
                        TrajectoryStep(
                            choice.card_indices, choice.logprob, choice.value,
                            choice.features, choice.legal_masks, choice.critic_features,
                            decision_type="schupfen",
                        )
                    )
                _advance(run, choice.concrete_action)
        for model, group in wish_groups.items():
            if not group.batch:
                continue
            choices = model.act_wish_batch(group.batch)
            for (run, seat), choice in zip(group.meta, choices):
                if seat in run.trajs:
                    run.trajs[seat].steps.append(
                        TrajectoryStep(
                            choice.intent_index, choice.logprob, choice.value,
                            choice.features, None, choice.critic_features,
                            decision_type="wish",
                        )
                    )
                _advance(run, choice.concrete_action)
    else:
        raise RuntimeError(
            f"rollout exceeded {_MAX_STEPS} ticks without resolving — "
            "likely an infinite loop in policy or engine."
        )

    for run in runs:
        final = run.state.public.scores
        other = 1 - run.learner_team
        reward = float(
            (final[run.learner_team] - run.initial_scores[run.learner_team])
            - (final[other] - run.initial_scores[other])
        )
        for traj in run.trajs.values():
            traj.reward = reward


def _supports_schupfen(model: RolloutPolicy) -> bool:
    return callable(getattr(model, "act_schupfen_batch", None))


def _supports_wish(model: RolloutPolicy) -> bool:
    return callable(getattr(model, "act_wish_batch", None))


def _supports_calls(model: RolloutPolicy) -> bool:
    return callable(getattr(model, "act_call_batch", None))


def _solicit_grand(
    runs: list[_GameRun],
    grand_prefixes: list,
    policy: RolloutPolicy,
    opponent_policy: RolloutPolicy,
    learner_team: int,
) -> None:
    """Decide every seat's Grand-Tichu Call on its synthetic deal-time state and
    inject the callers. Call-capable models decide (and learner seats record) via a
    per-model batched `act_call_batch("grand", ...)`; the rest fall back to the
    seat agent's `should_call`. Mirrors `play_full_round`'s independent per-seat ask."""
    groups: dict[RolloutPolicy, list] = {policy: [], opponent_policy: []}
    fallback: list[tuple[_GameRun, int, object]] = []
    callers: dict[int, set[int]] = {id(run): set() for run in runs}
    for run, prefixes in zip(runs, grand_prefixes):
        grand_state = _grand_state(prefixes)
        for seat in range(_NUM_PLAYERS):
            private = grand_state.private_view(seat)
            model = policy if seat % 2 == learner_team else opponent_policy
            if _supports_calls(model):
                groups[model].append((run, seat, private, grand_state))
            else:
                fallback.append((run, seat, private))
    for model, items in groups.items():
        if not items:
            continue
        decisions = [(seat, private, gs) for _run, seat, private, gs in items]
        choices = model.act_call_batch("grand", decisions)
        for (run, seat, _private, _gs), choice in zip(items, choices):
            _record_call(run, seat, choice, "grand")
            if choice.called:
                callers[id(run)].add(seat)
    for run, seat, private in fallback:
        if _calls(run.seat_agents[seat], private, "grand"):
            callers[id(run)].add(seat)
    for run in runs:
        run.state = _inject_callers(run.state, grand=frozenset(callers[id(run)]))


def _record_call(run: _GameRun, seat: int, choice, kind: str) -> None:
    """Record a learner seat's binary Call decision as a 1-step trajectory entry
    (the `called` flag is the action index: 1 == call, 0 == skip)."""
    if seat in run.trajs:
        run.trajs[seat].steps.append(
            TrajectoryStep(
                int(choice.called), choice.logprob, choice.value,
                choice.features, None, choice.critic_features, decision_type=kind,
            )
        )


def _advance(run: _GameRun, action: ConcreteAction) -> None:
    run.state, _, done, _ = step(run.state, action)
    if done:
        run.done = True


def _maybe_solicit_tichu(
    run: _GameRun, seat: int, action: ConcreteAction, model: RolloutPolicy
) -> None:
    """Ask `seat` for a regular Tichu Call at its first non-Pass Play (ADR-0018) and
    inject the caller into the state (preserving any Grand-Tichu callers). A
    call-capable `model` decides (and a learner seat records) via `act_call_batch`;
    otherwise the seat's frozen agent `should_call` decides. Grand-Tichu callers are
    never asked Tichu (grand supersedes). A no-op on Pass and already-asked seats —
    mirrors `tichu_eval.play_full.play_full_round`."""
    if isinstance(action, Pass) or seat in run.asked_tichu:
        return
    run.asked_tichu.add(seat)
    if seat in run.state.public.grand_tichu_callers:
        return
    private = run.state.private_view(seat)
    if _supports_calls(model):
        choice = model.act_call_batch("tichu", [(seat, private, run.state)])[0]
        _record_call(run, seat, choice, "tichu")
        called = bool(choice.called)
    else:
        called = _calls(run.seat_agents[seat], private, "tichu")
    if called:
        callers = frozenset(run.state.public.tichu_callers | {seat})
        run.state = GameState(
            hands=run.state.hands,
            public=replace(run.state.public, tichu_callers=callers),
        )


def _grand_state(grand_prefixes: Sequence) -> GameState:
    """The synthetic (8,8,8,8) deal-time state the Grand-Tichu Call is decided on —
    no prior callers visible, matching how the Grand-Tichu Network was trained."""
    hands = tuple(frozenset(p) for p in grand_prefixes)
    return GameState(
        hands=hands,
        public=PublicState(
            current_player=0,
            hand_sizes=tuple(len(h) for h in hands),
            scores=(0, 0),
            trick=Trick.empty(),
        ),
    )


def _calls(agent: Agent, private_state, kind: str) -> bool:
    should_call = getattr(agent, "should_call", None)
    return bool(should_call(private_state, kind)) if should_call else False


def _inject_callers(state: GameState, *, grand: frozenset[int]) -> GameState:
    return GameState(
        hands=state.hands,
        public=replace(state.public, grand_tichu_callers=grand),
    )
