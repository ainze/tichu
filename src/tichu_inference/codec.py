"""JSON codec for `PrivateState` and engine `ConcreteAction`s.

The HTTP `POST /act` body uses exactly this shape:
  {
    "private_state": { ... private_state_to_json output ... },
    "difficulty": "easy" | "medium" | "hard" | "master"
  }

The response:
  {
    "action": { ... action_to_json output ... },
    "action_index": int | null,
    "fallback_used": bool
  }
"""

from typing import Any

from tichu_engine.cards import Card, SpecialCard, Suit
from tichu_engine.combinations import (
    CardOrSpecial,
    FourOfAKindBomb,
    FullHouse,
    Pair,
    PairStep,
    Single,
    Straight,
    StraightFlushBomb,
    Triple,
)
from tichu_engine.deck import fresh_deck
from tichu_engine.legality import (
    BombInterrupt,
    DragonGive,
    MahjongWish,
    PASS,
    Pass,
    SchupfenPass,
)
from tichu_engine.rich_history import RichHistory
from tichu_engine.state import (
    DragonGivePending,
    MahjongWishPending,
    Play,
    PrivateState,
    PublicState,
    SchupfenPending,
    Trick,
)


_DECK = fresh_deck()
_CARD_TO_ID: dict = {card: i for i, card in enumerate(_DECK)}
_ID_TO_CARD: tuple = tuple(_DECK)


# ---------------------- Cards ----------------------

def card_to_id(card: CardOrSpecial) -> int:
    return _CARD_TO_ID[card]


def id_to_card(card_id: int) -> CardOrSpecial:
    return _ID_TO_CARD[card_id]


# ---------------------- Combinations ----------------------

def combination_to_json(combo: Any) -> dict:
    if isinstance(combo, Single):
        return {
            "kind": "Single",
            "card": card_to_id(combo.card),
            "as_rank": combo.as_rank,
        }
    if isinstance(combo, Pair):
        return {"kind": "Pair", "cards": [card_to_id(combo.a), card_to_id(combo.b)]}
    if isinstance(combo, Triple):
        return {
            "kind": "Triple",
            "cards": [card_to_id(combo.a), card_to_id(combo.b), card_to_id(combo.c)],
        }
    if isinstance(combo, FullHouse):
        return {
            "kind": "FullHouse",
            "triple": combination_to_json(combo.triple),
            "pair": combination_to_json(combo.pair),
        }
    if isinstance(combo, PairStep):
        return {
            "kind": "PairStep",
            "pairs": [combination_to_json(p) for p in combo.pairs],
        }
    if isinstance(combo, Straight):
        return {
            "kind": "Straight",
            "cards": [card_to_id(c) for c in combo.cards],
            "phoenix_as_rank": combo.phoenix_as_rank,
        }
    if isinstance(combo, FourOfAKindBomb):
        return {
            "kind": "FourOfAKindBomb",
            "cards": [card_to_id(combo.a), card_to_id(combo.b),
                      card_to_id(combo.c), card_to_id(combo.d)],
        }
    if isinstance(combo, StraightFlushBomb):
        return {
            "kind": "StraightFlushBomb",
            "cards": [card_to_id(c) for c in combo.cards],
        }
    raise ValueError(f"unknown combination type: {type(combo).__name__}")


def combination_from_json(blob: dict) -> Any:
    kind = blob["kind"]
    if kind == "Single":
        return Single(id_to_card(blob["card"]), as_rank=blob.get("as_rank"))
    if kind == "Pair":
        a, b = blob["cards"]
        return Pair(id_to_card(a), id_to_card(b))
    if kind == "Triple":
        a, b, c = blob["cards"]
        return Triple(id_to_card(a), id_to_card(b), id_to_card(c))
    if kind == "FullHouse":
        return FullHouse(
            triple=combination_from_json(blob["triple"]),
            pair=combination_from_json(blob["pair"]),
        )
    if kind == "PairStep":
        return PairStep(tuple(combination_from_json(p) for p in blob["pairs"]))
    if kind == "Straight":
        return Straight(
            cards=tuple(id_to_card(i) for i in blob["cards"]),
            phoenix_as_rank=blob.get("phoenix_as_rank"),
        )
    if kind == "FourOfAKindBomb":
        a, b, c, d = blob["cards"]
        return FourOfAKindBomb(
            id_to_card(a), id_to_card(b), id_to_card(c), id_to_card(d),
        )
    if kind == "StraightFlushBomb":
        return StraightFlushBomb(tuple(id_to_card(i) for i in blob["cards"]))
    raise ValueError(f"unknown combination kind: {kind!r}")


# ---------------------- Actions ----------------------

def action_to_json(action: Any) -> dict:
    if isinstance(action, Pass):
        return {"kind": "Pass"}
    if isinstance(action, DragonGive):
        return {"kind": "DragonGive", "target": action.target}
    if isinstance(action, MahjongWish):
        return {"kind": "MahjongWish", "rank": action.rank}
    if isinstance(action, SchupfenPass):
        return {
            "kind": "SchupfenPass",
            "to_next": card_to_id(action.to_next),
            "to_partner": card_to_id(action.to_partner),
            "to_previous": card_to_id(action.to_previous),
        }
    if isinstance(action, BombInterrupt):
        return {
            "kind": "BombInterrupt",
            "player": action.player,
            "bomb": combination_to_json(action.bomb),
        }
    return combination_to_json(action)


# ---------------------- Trick + pending ----------------------

def trick_to_json(trick: Trick) -> dict:
    return {
        "plays": [
            {"player": p.player, "combination": combination_to_json(p.combination)}
            for p in trick.plays
        ],
        "leader": trick.leader,
        "passes": sorted(trick.passes),
    }


def trick_from_json(blob: dict) -> Trick:
    plays = tuple(
        Play(player=int(p["player"]), combination=combination_from_json(p["combination"]))
        for p in blob.get("plays", [])
    )
    leader = blob.get("leader")
    passes = frozenset(int(p) for p in blob.get("passes", []))
    return Trick(plays=plays, leader=leader, passes=passes)


def _pending_to_json(pending) -> dict | None:
    if pending is None:
        return None
    if isinstance(pending, SchupfenPending):
        return {
            "kind": "schupfen",
            "submitted": [
                None if s is None else [card_to_id(c) for c in s]
                for s in pending.submitted
            ],
        }
    if isinstance(pending, DragonGivePending):
        return {"kind": "dragon_give", "winner": pending.winner, "points": pending.points}
    if isinstance(pending, MahjongWishPending):
        return {"kind": "mahjong_wish", "player": pending.player}
    raise ValueError(f"unknown pending decision: {type(pending).__name__}")


def _pending_from_json(blob):
    if blob is None:
        return None
    kind = blob["kind"]
    if kind == "schupfen":
        submitted = tuple(
            None if s is None else tuple(id_to_card(i) for i in s)
            for s in blob["submitted"]
        )
        return SchupfenPending(submitted=submitted)
    if kind == "dragon_give":
        return DragonGivePending(winner=int(blob["winner"]), points=int(blob["points"]))
    if kind == "mahjong_wish":
        return MahjongWishPending(player=int(blob["player"]))
    raise ValueError(f"unknown pending kind: {kind!r}")


# ---------------------- Public + Private ----------------------

def public_state_to_json(public: PublicState) -> dict:
    return {
        "current_player": public.current_player,
        "hand_sizes": list(public.hand_sizes),
        "scores": list(public.scores),
        "trick": trick_to_json(public.trick),
        "mahjong_wish": public.mahjong_wish,
        "pending_decision": _pending_to_json(public.pending_decision),
        "round_points_by_player": list(public.round_points_by_player),
        "out_order": list(public.out_order),
        "tichu_callers": sorted(public.tichu_callers),
        "grand_tichu_callers": sorted(public.grand_tichu_callers),
        # v6 (ADR-0038): per-player provenance + cross-Trick decline accumulators.
        "played_cards_by_player": [
            [card_to_id(c) for c in sorted(s, key=card_to_id)]
            for s in public.played_cards_by_player
        ],
        "declined_top_by_player": [list(t) for t in public.declined_top_by_player],
        "lead_summary_by_player": [list(t) for t in public.lead_summary_by_player],
        "pass_stats_by_player": [list(t) for t in public.pass_stats_by_player],
        # v7 (ADR-0044): the Rich History Block accumulator.
        "rich_history": _rich_history_to_json(public.rich_history),
    }


def _rich_history_to_json(rich: RichHistory) -> dict:
    return {
        "declined_ctx": [list(r) for r in rich.declined_ctx],
        "declined_counts": [list(r) for r in rich.declined_counts],
        "declined_lengths": [list(r) for r in rich.declined_lengths],
        "declined_stake": list(rich.declined_stake),
        "declined_bomb": list(rich.declined_bomb),
        "wish_void": [list(r) for r in rich.wish_void],
        "wish_soft": [list(r) for r in rich.wish_soft],
        "lead_high_single": list(rich.lead_high_single),
        "card_play_order": list(rich.card_play_order),
        "plays_so_far": rich.plays_so_far,
    }


def _rich_history_from_json(blob: dict) -> RichHistory:
    def rows(key):
        return tuple(tuple(int(x) for x in row) for row in blob[key])

    def flat(key):
        return tuple(int(x) for x in blob[key])

    return RichHistory(
        declined_ctx=rows("declined_ctx"),
        declined_counts=rows("declined_counts"),
        declined_lengths=rows("declined_lengths"),
        declined_stake=flat("declined_stake"),
        declined_bomb=flat("declined_bomb"),
        wish_void=rows("wish_void"),
        wish_soft=rows("wish_soft"),
        lead_high_single=flat("lead_high_single"),
        card_play_order=flat("card_play_order"),
        plays_so_far=int(blob["plays_so_far"]),
    )


def public_state_from_json(blob: dict, *, require_v7: bool = False) -> PublicState:
    return PublicState(
        current_player=int(blob["current_player"]),
        hand_sizes=tuple(int(x) for x in blob["hand_sizes"]),  # type: ignore[arg-type]
        scores=tuple(int(x) for x in blob["scores"]),  # type: ignore[arg-type]
        trick=trick_from_json(blob["trick"]),
        mahjong_wish=blob.get("mahjong_wish"),
        pending_decision=_pending_from_json(blob.get("pending_decision")),
        round_points_by_player=tuple(int(x) for x in blob.get("round_points_by_player", (0, 0, 0, 0))),  # type: ignore[arg-type]
        out_order=tuple(int(x) for x in blob.get("out_order", ())),
        tichu_callers=frozenset(int(x) for x in blob.get("tichu_callers", ())),
        grand_tichu_callers=frozenset(int(x) for x in blob.get("grand_tichu_callers", ())),
        # v6 (ADR-0038): default to the empty accumulators so older clients that
        # omit these keys deserialize to a valid (zeroed) state.
        played_cards_by_player=tuple(
            frozenset(id_to_card(i) for i in s)
            for s in blob.get("played_cards_by_player", ([], [], [], []))
        ),  # type: ignore[arg-type]
        declined_top_by_player=tuple(
            tuple(int(x) for x in row)
            for row in blob.get("declined_top_by_player", ((0,) * 6,) * 4)
        ),  # type: ignore[arg-type]
        lead_summary_by_player=tuple(
            tuple(int(x) for x in row)
            for row in blob.get("lead_summary_by_player", ((0, 0),) * 4)
        ),  # type: ignore[arg-type]
        pass_stats_by_player=tuple(
            tuple(int(x) for x in row)
            for row in blob.get("pass_stats_by_player", ((0, 0),) * 4)
        ),  # type: ignore[arg-type]
        # v7 (ADR-0044): see `_decode_rich_history` for why this one has a strict
        # mode when the v6 fields above do not.
        rich_history=_decode_rich_history(blob, require_v7=require_v7),
    )


def _decode_rich_history(blob: dict, *, require_v7: bool) -> RichHistory:
    """Decode the Rich History Block, or fail loudly when asked to.

    The v6 accumulators above default to zeros on a missing key. That
    leniency is why the served agent can today run with ~42% of its Feature
    Vector pinned to zero and nothing anywhere saying so — the same failure
    shape as the `team_scores` train/inference skew. v7 would take that to ~59%.

    The default stays lenient so the CURRENT client keeps working while the v7
    lineage is being trained (ADR-0044: update the client only after beating
    cpfix3328). `require_v7=True` is the switch to flip the moment it is updated:
    it turns a silent, invisible degradation into an immediate, obvious failure,
    which is the only way a half-updated client cannot quietly corrupt inputs.
    """
    raw = blob.get("rich_history")
    if raw is None:
        if require_v7:
            raise ValueError(
                "wire PublicState is missing 'rich_history': the client predates "
                "featurizer v7 (ADR-0044). Serving it would feed the policy 233 "
                "zeroed dims that training saw populated. Update the client, or "
                "unset require_v7 to accept the degraded input deliberately."
            )
        return RichHistory()
    return _rich_history_from_json(raw)


def private_state_to_json(ps: PrivateState) -> dict:
    return {
        "player": ps.player,
        "hand": [card_to_id(c) for c in sorted(ps.hand, key=card_to_id)],
        "public": public_state_to_json(ps.public),
        # v6 (ADR-0038): self-only schupfen provenance [from_next, from_partner,
        # from_previous]; per-slot null before the exchange resolves.
        "schupfen_received": [
            None if c is None else card_to_id(c) for c in ps.schupfen_received
        ],
    }


def private_state_from_json(blob: dict, *, require_v7: bool = False) -> PrivateState:
    """`require_v7` is forwarded to `public_state_from_json` — see
    `_decode_rich_history`. Every serving decode goes through here, so this is
    the parameter the HTTP layer threads its strict-mode flag into."""
    if "player" not in blob or "hand" not in blob or "public" not in blob:
        raise ValueError(
            "private_state_from_json requires keys {'player', 'hand', 'public'}; "
            f"got {sorted(blob.keys())}"
        )
    received = blob.get("schupfen_received", (None, None, None))
    return PrivateState(
        player=int(blob["player"]),
        hand=frozenset(id_to_card(i) for i in blob["hand"]),
        public=public_state_from_json(blob["public"], require_v7=require_v7),
        schupfen_received=tuple(
            None if i is None else id_to_card(i) for i in received
        ),  # type: ignore[arg-type]
    )
