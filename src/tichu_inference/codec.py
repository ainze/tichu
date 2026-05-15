"""JSON codec for `PrivateState` and engine `Action`s.

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
    }


def public_state_from_json(blob: dict) -> PublicState:
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
    )


def private_state_to_json(ps: PrivateState) -> dict:
    return {
        "player": ps.player,
        "hand": [card_to_id(c) for c in sorted(ps.hand, key=card_to_id)],
        "public": public_state_to_json(ps.public),
    }


def private_state_from_json(blob: dict) -> PrivateState:
    if "player" not in blob or "hand" not in blob or "public" not in blob:
        raise ValueError(
            "private_state_from_json requires keys {'player', 'hand', 'public'}; "
            f"got {sorted(blob.keys())}"
        )
    return PrivateState(
        player=int(blob["player"]),
        hand=frozenset(id_to_card(i) for i in blob["hand"]),
        public=public_state_from_json(blob["public"]),
    )
