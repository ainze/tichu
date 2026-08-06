"""The v7 Rich History Block accumulator (ADR-0044).

Round-long, per-seat facts that a single `PublicState` snapshot cannot recover:
*why* a seat declined (who was winning), what it declined, and what that proves.
Path-dependent by construction, so it is folded as each Decision resolves — the
ADR-0038 pattern — and lives on `PublicState` as ONE canonical definition read by
the featurizer for training and inference alike.

Deliberately carries only what v6 does not already have. `pass_stats_by_player`
and `lead_summary_by_player` stay the source for pass pressure and the low-lead
profile; duplicating them here would create two definitions that can drift.
"""

from dataclasses import dataclass, replace

from tichu_engine.card_slots import CARD_SLOTS as NUM_CARD_SLOTS, card_slot

NUM_PLAYERS = 4

# The six non-bomb Intent types, in the order `combo_type_and_rank` yields and
# `declined_top_by_player` already uses.
INTENT_SINGLE = 0
INTENT_PAIR = 1
INTENT_TRIPLE = 2
INTENT_FULL_HOUSE = 3
INTENT_PAIR_STEP = 4
INTENT_STRAIGHT = 5
NUM_INTENT_TYPES = 6

# Who was winning the Trick when the seat declined, from the DECLINER's own
# perspective. The distinction is the channel's reason for existing: declining
# into a partner's winning Trick says almost nothing about the hand, whereas
# declining while an opponent is winning is real evidence.
CTX_PARTNER_WINNING = 0
CTX_OPPONENT_WINNING = 1
CTX_NO_WINNER = 2
NUM_CONTEXTS = 3

_CTX_WIDTH = NUM_INTENT_TYPES * NUM_CONTEXTS  # 18 per seat

# Only PairStep and Straight vary in length; the other four Intent types are
# fixed-size, so a length channel for them would be a constant.
_LENGTH_SLOT: dict[int, int] = {INTENT_PAIR_STEP: 0, INTENT_STRAIGHT: 1}

# Wish ranks 2..14, one flag per rank per seat.
WISH_RANK_MIN = 2
WISH_RANK_MAX = 14
NUM_WISH_RANKS = WISH_RANK_MAX - WISH_RANK_MIN + 1  # 13


@dataclass(frozen=True)
class RichHistory:
    """Per-ABSOLUTE-seat accumulators, raw. The featurizer normalises and
    re-orders to relative seats, exactly as it does for the v6 channels."""

    # [seat][intent_type * NUM_CONTEXTS + context] = max primary rank declined
    # in that (type, context) pair; 0 = never.
    declined_ctx: tuple[tuple[int, ...], ...] = (
        (0,) * _CTX_WIDTH,
    ) * NUM_PLAYERS

    # [seat][rank - WISH_RANK_MIN] = 1 iff `seat` is PROVEN void in that rank.
    # Set only when the seat LED a fresh Trick under an active wish without
    # fulfilling it — see `with_proven_void` for why nothing else qualifies.
    wish_void: tuple[tuple[int, ...], ...] = (
        (0,) * NUM_WISH_RANKS,
    ) * NUM_PLAYERS

    # [seat] = times `seat` declined to beat a Bomb. v6 drops this entirely:
    # `combo_type_and_rank` returns None for a Bomb, so `declined_top` never
    # sees it, and the evidence — bombs are rare, and a seat holding one almost
    # always over-bombs here — is lost.
    declined_bomb: tuple[int, ...] = (0,) * NUM_PLAYERS

    # [seat][intent_type] = how many times `seat` declined that type. v6 keeps
    # only the max rank, so a seat that passed once on a King and one that
    # passed nine times look identical.
    declined_counts: tuple[tuple[int, ...], ...] = (
        (0,) * NUM_INTENT_TYPES,
    ) * NUM_PLAYERS
    # [seat][0] = longest PairStep declined, [seat][1] = longest Straight. The
    # only two variable-length Intent types.
    declined_lengths: tuple[tuple[int, ...], ...] = ((0, 0),) * NUM_PLAYERS
    # [seat] = card points in the richest Trick `seat` declined. Declining a fat
    # Trick is far stronger evidence than declining an empty one.
    declined_stake: tuple[int, ...] = (0,) * NUM_PLAYERS

    # [seat] = highest single-card lead. v6's `lead_summary` keeps only the
    # lowest; the pair describes a seat's lead range rather than one end of it.
    lead_high_single: tuple[int, ...] = (0,) * NUM_PLAYERS
    # [card_slot] = 1-based index of the play that contained this card; 0 = not
    # played. v6's `played_by` is a position-less set, so it cannot tell an Ace
    # dumped on Trick one from an Ace held to the endgame.
    card_play_order: tuple[int, ...] = (0,) * NUM_CARD_SLOTS
    plays_so_far: int = 0

    def lead_max_single(self, seat: int) -> int:
        """Highest single-card lead by `seat`; 0 if it has led no single."""
        return self.lead_high_single[seat]

    def play_order(self, card) -> int:
        """1-based play index in which `card` appeared; 0 if unplayed."""
        return self.card_play_order[card_slot(card)]

    def with_lead_single(self, seat: int, rank: int) -> "RichHistory":
        highs = list(self.lead_high_single)
        highs[seat] = max(highs[seat], rank)
        return replace(self, lead_high_single=tuple(highs))

    def with_play(self, card_slots) -> "RichHistory":
        """Stamp the next play index onto every card slot in one play."""
        slots = tuple(card_slots)
        if not slots:
            return self
        idx = self.plays_so_far + 1
        order = list(self.card_play_order)
        for s in slots:
            order[s] = idx
        return replace(
            self, card_play_order=tuple(order), plays_so_far=idx,
        )

    def declined_count(self, seat: int, intent_type: int) -> int:
        """Times `seat` declined `intent_type`."""
        return self.declined_counts[seat][intent_type]

    def declined_length(self, seat: int, intent_type: int) -> int:
        """Longest declined combination for the variable-length types; 0 for the
        fixed-length ones, which carry no length information."""
        slot = _LENGTH_SLOT.get(intent_type)
        return 0 if slot is None else self.declined_lengths[seat][slot]

    def declined_stakes(self, seat: int) -> int:
        """Card points in the richest Trick `seat` declined."""
        return self.declined_stake[seat]

    def declined_bomb_count(self, seat: int) -> int:
        """Times `seat` declined to beat a Bomb."""
        return self.declined_bomb[seat]

    def with_declined_bomb(self, seat: int) -> "RichHistory":
        counts = list(self.declined_bomb)
        counts[seat] += 1
        return replace(self, declined_bomb=tuple(counts))

    def declined_rank(self, seat: int, intent_type: int, context: int) -> int:
        """Max primary rank `seat` declined to beat, for one (type, context)."""
        return self.declined_ctx[seat][intent_type * NUM_CONTEXTS + context]

    def with_decline(
        self, seat: int, intent_type: int, context: int, rank: int,
        *, length: int = 0, stakes: int = 0,
    ) -> "RichHistory":
        """Fold one decline: rank (monotone max, per context), frequency, the
        declined combination's length, and the Trick's card points (monotone
        max)."""
        slot = intent_type * NUM_CONTEXTS + context
        ctx_rows = [list(r) for r in self.declined_ctx]
        ctx_rows[seat][slot] = max(ctx_rows[seat][slot], rank)

        count_rows = [list(r) for r in self.declined_counts]
        count_rows[seat][intent_type] += 1

        len_rows = [list(r) for r in self.declined_lengths]
        len_slot = _LENGTH_SLOT.get(intent_type)
        if len_slot is not None and length:
            len_rows[seat][len_slot] = max(len_rows[seat][len_slot], length)

        stake = list(self.declined_stake)
        stake[seat] = max(stake[seat], stakes)

        return replace(
            self,
            declined_ctx=tuple(tuple(r) for r in ctx_rows),
            declined_counts=tuple(tuple(r) for r in count_rows),
            declined_lengths=tuple(tuple(r) for r in len_rows),
            declined_stake=tuple(stake),
        )


    # [seat][rank - WISH_RANK_MIN] = 1 iff `seat` had a chance to fulfil an
    # active wish for that rank and did not. EVIDENCE, never proof — kept in a
    # separate channel from `wish_void` precisely so a consumer can weigh the
    # guess without the certainty ever being contaminated by it.
    wish_soft: tuple[tuple[int, ...], ...] = (
        (0,) * NUM_WISH_RANKS,
    ) * NUM_PLAYERS

    def wish_evidence(self, seat: int, rank: int) -> bool:
        """Whether `seat` passed up a chance to fulfil a wish for `rank`. A
        strong hint, not a certainty — use `is_wish_void` for that."""
        if not WISH_RANK_MIN <= rank <= WISH_RANK_MAX:
            return False
        return bool(self.wish_soft[seat][rank - WISH_RANK_MIN])

    def with_wish_evidence(self, seat: int, rank: int) -> "RichHistory":
        if not WISH_RANK_MIN <= rank <= WISH_RANK_MAX:
            return self
        slot = rank - WISH_RANK_MIN
        if self.wish_soft[seat][slot]:
            return self
        rows = [list(r) for r in self.wish_soft]
        rows[seat][slot] = 1
        return replace(self, wish_soft=tuple(tuple(r) for r in rows))

    def is_wish_void(self, seat: int, rank: int) -> bool:
        """Whether `seat` is PROVEN void in `rank`. Never a guess."""
        if not WISH_RANK_MIN <= rank <= WISH_RANK_MAX:
            return False
        return bool(self.wish_void[seat][rank - WISH_RANK_MIN])

    def with_proven_void(self, seat: int, rank: int) -> "RichHistory":
        """Record a PROVEN void.

        Soundness is the whole value of this channel and it is narrow: only a
        LEAD under an active wish qualifies. A seat leading a fresh Trick may
        play any Combination in hand, so holding the wished rank would have
        forced a wish-fulfilling lead — `_apply_wish` restricts to
        wish-fulfilling actions *among already-legal ones*, and on a lead that is
        everything. Following and passing are EVIDENCE, not proof: a seat can
        hold the rank and still have no legal beating combination containing it.
        Do not widen this to those cases — a false void silently corrupts every
        downstream inference built on it.
        """
        if not WISH_RANK_MIN <= rank <= WISH_RANK_MAX:
            return self
        slot = rank - WISH_RANK_MIN
        if self.wish_void[seat][slot]:
            return self
        row = list(self.wish_void[seat])
        row[slot] = 1
        rows = list(self.wish_void)
        rows[seat] = tuple(row)
        return replace(self, wish_void=tuple(rows))


def context_for(decliner_seat: int, winner_seat: int | None) -> int:
    """Trick-winner context from the decliner's perspective. Teams are seat % 2,
    so a seat that is somehow winning its own declined Trick reads as
    partner-winning — the "declining says little" bucket, which is correct."""
    if winner_seat is None:
        return CTX_NO_WINNER
    if winner_seat % 2 == decliner_seat % 2:
        return CTX_PARTNER_WINNING
    return CTX_OPPONENT_WINNING
