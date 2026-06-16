"""Tichu game state.

`Trick` records the cards currently on the table within one trick.
`PublicState` is the view all players share — whose turn it is, hand sizes,
scores, and the current trick.
`PrivateState` is a per-player view: own hand plus the public state.
`GameState` is the engine's full ground truth from which any player's view is
derivable. The deal_initial_state factory produces a GameState from a seed.

All types are immutable frozen dataclasses; updates return new instances.
"""

import random
from dataclasses import dataclass, field
from typing import Union

from tichu_engine.cards import Card, MAHJONG, SpecialCard
from tichu_engine.combinations import CardOrSpecial
from tichu_engine.deck import fresh_deck


NUM_PLAYERS = 4
INITIAL_HAND_SIZE = 14


@dataclass(frozen=True)
class Play:
    """One played combination within a trick. The combination type is opaque here —
    it's whatever the engine considers a legal Combination."""

    player: int
    combination: object


@dataclass(frozen=True)
class Trick:
    """Cards played in the current trick. `passes` records players who have passed
    since the last play — they are out for the duration of this trick (except via
    bomb interrupts, which clear their pass status)."""

    plays: tuple[Play, ...]
    leader: int | None
    passes: frozenset[int] = frozenset()

    @classmethod
    def empty(cls) -> "Trick":
        return cls(plays=(), leader=None, passes=frozenset())

    @property
    def top_combination(self) -> object | None:
        return self.plays[-1].combination if self.plays else None

    def add_play(self, player: int, combination: object) -> "Trick":
        # Any new play opens a fresh response cycle — previously-passed players
        # may react. This matches BSW's behaviour (and avoids stranding the
        # trick when a passed player later has a chance to beat a new top).
        return Trick(
            plays=self.plays + (Play(player=player, combination=combination),),
            leader=player,
            passes=frozenset(),
        )

    def add_pass(self, player: int) -> "Trick":
        return Trick(plays=self.plays, leader=self.leader, passes=self.passes | {player})


@dataclass(frozen=True)
class DragonGivePending:
    """Awaiting the trick winner's choice of which opponent receives the trick."""

    winner: int
    points: int


@dataclass(frozen=True)
class MahjongWishPending:
    """Awaiting the Mahjong player's wish declaration (a rank or no wish)."""

    player: int


@dataclass(frozen=True)
class SchupfenPending:
    """Pre-play card-pass phase. Each player passes one card to each opponent.

    `submitted` is a 4-tuple of either None or a 3-tuple of cards in
    `(to-next, to-partner, to-previous)` order, where directions are relative
    to the player at the index. The exchange is applied once all four are non-None.
    """

    submitted: tuple


PendingDecision = Union[DragonGivePending, MahjongWishPending, SchupfenPending]


@dataclass(frozen=True)
class PublicState:
    current_player: int
    hand_sizes: tuple[int, int, int, int]
    scores: tuple[int, int]
    trick: Trick
    mahjong_wish: int | None = None
    pending_decision: PendingDecision | None = None
    # Round-only state, reset at the start of each round.
    round_points_by_player: tuple[int, int, int, int] = (0, 0, 0, 0)
    out_order: tuple[int, ...] = ()
    tichu_callers: frozenset[int] = frozenset()
    grand_tichu_callers: frozenset[int] = frozenset()
    played_cards_this_round: frozenset[CardOrSpecial] = frozenset()
    # v6 (ADR-0038): per-player provenance of cards played this round, by
    # absolute seat. The featurizer reads this relative-seat ordered. Reset
    # per round alongside `played_cards_this_round`.
    played_cards_by_player: tuple[
        frozenset[CardOrSpecial], frozenset[CardOrSpecial],
        frozenset[CardOrSpecial], frozenset[CardOrSpecial],
    ] = (frozenset(), frozenset(), frozenset(), frozenset())
    # v6 (ADR-0028 B-core): per-absolute-seat cross-Trick decline accumulators.
    # declined_top_by_player[seat] = max primary-rank that seat declined to beat,
    # per non-bomb intent type (Single, Pair, Triple, FullHouse, PairStep,
    # Straight); 0 = never. Raw ranks — the featurizer normalises rank/14.
    declined_top_by_player: tuple[
        tuple[int, int, int, int, int, int], ...
    ] = ((0, 0, 0, 0, 0, 0),) * 4
    # v6 (ADR-0028 B-core): per-absolute-seat (tricks_led, lowest_single_lead_rank).
    # lowest_single_lead_rank 0 = no single lead yet. Raw — featurizer normalises.
    lead_summary_by_player: tuple[tuple[int, int], ...] = ((0, 0),) * 4
    # v6 (ADR-0028 B-core): per-absolute-seat (n_passes, n_play_decisions).
    # pass_pressure = n_passes / n_play_decisions.
    pass_stats_by_player: tuple[tuple[int, int], ...] = ((0, 0),) * 4

    def __post_init__(self) -> None:
        if not 0 <= self.current_player < NUM_PLAYERS:
            raise ValueError(f"current_player must be in [0, {NUM_PLAYERS}), got {self.current_player}")
        if len(self.hand_sizes) != NUM_PLAYERS:
            raise ValueError(f"hand_sizes must have {NUM_PLAYERS} entries, got {len(self.hand_sizes)}")
        if len(self.scores) != 2:
            raise ValueError(f"scores must have 2 entries (team 0, team 1), got {len(self.scores)}")
        if self.mahjong_wish is not None and not 2 <= self.mahjong_wish <= 14:
            raise ValueError(f"mahjong_wish must be a rank in [2, 14] or None, got {self.mahjong_wish}")


@dataclass(frozen=True)
class PrivateState:
    player: int
    hand: frozenset[CardOrSpecial]
    public: PublicState
    # v6 (ADR-0038): self-only schupfen provenance — the three cards this player
    # received, by relative give-direction (from_next, from_partner,
    # from_previous). Private to the player (never opponents' exchanges). None
    # before the exchange resolves.
    schupfen_received: tuple[
        CardOrSpecial | None, CardOrSpecial | None, CardOrSpecial | None,
    ] = (None, None, None)

    def __post_init__(self) -> None:
        if not 0 <= self.player < NUM_PLAYERS:
            raise ValueError(f"player must be in [0, {NUM_PLAYERS}), got {self.player}")
        if len(self.hand) != self.public.hand_sizes[self.player]:
            raise ValueError(
                f"hand size {len(self.hand)} does not match public hand_sizes[{self.player}]"
                f"={self.public.hand_sizes[self.player]}"
            )


@dataclass(frozen=True)
class GameState:
    """Engine ground truth: every player's hand plus the public state."""

    hands: tuple[frozenset[CardOrSpecial], ...]
    public: PublicState
    # v6 (ADR-0038): per-player schupfen provenance — the three cards each seat
    # received, by relative give-direction (from_next, from_partner,
    # from_previous), or None before/without an exchange. Private per seat, so it
    # lives on GameState (never serialised to agents) and is surfaced only via
    # private_view, NOT on the shared PublicState.
    schupfen_received: tuple = (None, None, None, None)

    def __post_init__(self) -> None:
        if len(self.hands) != NUM_PLAYERS:
            raise ValueError(f"hands must have {NUM_PLAYERS} entries, got {len(self.hands)}")

    def private_view(self, player: int) -> PrivateState:
        if not 0 <= player < NUM_PLAYERS:
            raise ValueError(f"player must be in [0, {NUM_PLAYERS}), got {player}")
        received = self.schupfen_received[player]
        return PrivateState(
            player=player, hand=self.hands[player], public=self.public,
            schupfen_received=received if received is not None else (None, None, None),
        )


def deal_initial_state(seed: int) -> GameState:
    """Shuffle the deck with the given seed and deal 14 cards to each of 4 players.

    Skips schupfen — the starting player is set directly to whoever holds the Mahjong.
    Use `deal_for_schupfen` for the full pre-play card-pass phase.
    """
    rng = random.Random(seed)
    deck = list(fresh_deck())
    rng.shuffle(deck)
    hands: list[frozenset[CardOrSpecial]] = []
    for i in range(NUM_PLAYERS):
        chunk = deck[i * INITIAL_HAND_SIZE : (i + 1) * INITIAL_HAND_SIZE]
        hands.append(frozenset(chunk))
    starting_player = next(p for p, h in enumerate(hands) if MAHJONG in h)
    public = PublicState(
        current_player=starting_player,
        hand_sizes=tuple(len(h) for h in hands),  # type: ignore[arg-type]
        scores=(0, 0),
        trick=Trick.empty(),
    )
    return GameState(hands=tuple(hands), public=public)


def deal_for_schupfen(seed: int) -> GameState:
    """Like `deal_initial_state`, but begins in the schupfen (card-pass) phase.

    Player 0 owes the first schupfen submission; play proceeds clockwise. After
    all four players submit, the engine applies the exchange and sets the
    current player to the Mahjong holder.
    """
    rng = random.Random(seed)
    deck = list(fresh_deck())
    rng.shuffle(deck)
    hands: list[frozenset[CardOrSpecial]] = []
    for i in range(NUM_PLAYERS):
        chunk = deck[i * INITIAL_HAND_SIZE : (i + 1) * INITIAL_HAND_SIZE]
        hands.append(frozenset(chunk))
    public = PublicState(
        current_player=0,
        hand_sizes=tuple(len(h) for h in hands),  # type: ignore[arg-type]
        scores=(0, 0),
        trick=Trick.empty(),
        pending_decision=SchupfenPending(submitted=(None, None, None, None)),
    )
    return GameState(hands=tuple(hands), public=public)
