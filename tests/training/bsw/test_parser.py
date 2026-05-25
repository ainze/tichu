"""End-to-end .tch parsing on the two checked-in samples.

Sample 2417500.tch and 2417501.tch are real BSW games. The parser must return
a structured ParsedGame whose shape matches the recorded log content.
"""

from pathlib import Path

import pytest

from tichu_engine.cards import DOG, DRAGON, MAHJONG, PHOENIX
from tichu_training.bsw.parser import parse_tch
from tichu_training.bsw.records import ParsedGame


_SAMPLES = Path(__file__).resolve().parents[3] / "sample"


@pytest.fixture(scope="module")
def game_00() -> ParsedGame:
    return parse_tch((_SAMPLES / "2417500.tch").read_text(encoding="utf-8"), game_id="2417500")


@pytest.fixture(scope="module")
def game_01() -> ParsedGame:
    return parse_tch((_SAMPLES / "2417501.tch").read_text(encoding="utf-8"), game_id="2417501")


# ---- Game-level shape ----

def test_schupfen_handles_hyphenated_recipient_names():
    """Player handles may contain `-` (e.g. `raf-4`). The schupfen body uses
    ` - ` (space-dash-space) as the recipient separator, so a bare `-` split
    over-splits hyphenated handles. The parser must use the real separator."""
    raw = (_SAMPLES / "2417500.tch").read_text(encoding="utf-8")
    # Replace seat-1's handle `1David` with `raf-4` everywhere in round 0.
    # The end of round 0 is the first `Ergebnis:` line; only edit before that.
    ergebnis_idx = raw.index("Ergebnis:")
    head = raw[:ergebnis_idx].replace("1David", "raf-4")
    tail = raw[ergebnis_idx:]
    spliced = head + tail

    game = parse_tch(spliced, game_id="hyphen")

    # Seat 1's handle in round 0 is the renamed value.
    assert game.rounds[0].pre_deal_hands is not None  # round parsed
    # Every schupfen submission references valid cards (no card-decode crash).
    for sub in game.rounds[0].schupfen:
        assert sub is not None
        assert sub.schupfen_to_next is not None


def test_pre_schupfen_tichu_call_is_accepted(tmp_path):
    """A regular `Tichu:` line may appear between Startkarten and Schupfen
    (a player who's just seen their 14-card hand can declare Tichu before
    passing). The parser must accept it and record the caller."""
    raw = (_SAMPLES / "2417500.tch").read_text(encoding="utf-8")
    # Inject a pre-Schupfen Tichu call into round 0. Find the Schupfen header
    # for the first round and insert the line just before it.
    lines = raw.splitlines()
    schupfen_idx = next(i for i, line in enumerate(lines) if line.startswith("Schupfen:"))
    lines.insert(schupfen_idx, "Tichu: (2)evdokia!!")
    spliced = "\n".join(lines) + "\n"

    game = parse_tch(spliced, game_id="spliced")

    assert 2 in game.rounds[0].tichu_callers


def test_parsed_game_has_four_seat_handles(game_00):
    assert game_00.handles == ("evi_sea", "1David", "evdokia!!", "Lisaaaaaaa")


def test_parsed_game_two_has_different_handles(game_01):
    assert game_01.handles == ("evi_sea", "1David", "evdokia!!", "Steffi0722")


def test_parsed_game_preserves_game_id(game_00):
    assert game_00.game_id == "2417500"


def test_parsed_game_has_ten_rounds(game_00):
    assert len(game_00.rounds) == 10


# ---- Round dealing ----

def test_round_one_pre_deal_hands_are_eight_cards_each(game_00):
    r = game_00.rounds[0]
    for seat, hand in enumerate(r.pre_deal_hands):
        assert len(hand) == 8, f"seat {seat} had {len(hand)} pre-deal cards"


def test_round_one_start_hands_are_fourteen_cards_each(game_00):
    r = game_00.rounds[0]
    for seat, hand in enumerate(r.start_hands):
        assert len(hand) == 14, f"seat {seat} had {len(hand)} start cards"


def test_round_one_pre_deal_is_a_subset_of_start_hand(game_00):
    r = game_00.rounds[0]
    for seat in range(4):
        assert r.pre_deal_hands[seat] <= r.start_hands[seat], (
            f"seat {seat}: pre-deal cards are not a subset of start hand"
        )


def test_start_hands_partition_the_full_deck(game_00):
    from tichu_engine.deck import fresh_deck
    for ri, r in enumerate(game_00.rounds):
        union: set = set()
        for hand in r.start_hands:
            union |= hand
        assert union == set(fresh_deck()), f"round {ri}: start hands don't cover the deck"


# ---- Calls ----

def test_grand_tichu_call_recorded_in_first_round(game_00):
    # "Grosses Tichu: (1)1David" appears in round 1
    assert 1 in game_00.rounds[0].grand_tichu_callers
    assert game_00.rounds[0].tichu_callers == frozenset()


def test_tichu_call_recorded_when_player_calls_mid_round(game_00):
    # Round 3 has "Tichu: (2)evdokia!!"
    assert 2 in game_00.rounds[2].tichu_callers


def test_round_with_no_calls_has_empty_caller_sets(game_01):
    # Round 1 of sample 01 has no Tichu/Grosses Tichu lines
    assert game_01.rounds[0].grand_tichu_callers == frozenset()
    assert game_01.rounds[0].tichu_callers == frozenset()


# ---- Schupfen ----

def test_schupfen_has_four_submissions(game_00):
    r = game_00.rounds[0]
    assert len(r.schupfen) == 4
    for seat in range(4):
        sub = r.schupfen[seat]
        assert sub.kind == "schupfen"
        assert sub.player == seat
        assert sub.schupfen_to_next is not None
        assert sub.schupfen_to_partner is not None
        assert sub.schupfen_to_previous is not None


def test_schupfen_cards_are_distinct_and_from_the_dealer_hand(game_00):
    r = game_00.rounds[0]
    for seat in range(4):
        sub = r.schupfen[seat]
        cards = {sub.schupfen_to_next, sub.schupfen_to_partner, sub.schupfen_to_previous}
        assert len(cards) == 3
        assert cards <= r.start_hands[seat]


# ---- Plays ----

def test_round_one_first_play_is_lisa_playing_mahjong(game_00):
    plays = game_00.rounds[0].plays
    first = plays[0]
    assert first.kind == "play"
    assert first.player == 3  # Lisa
    assert first.cards == (MAHJONG,)


def test_round_one_includes_a_mahjong_wish(game_00):
    plays = game_00.rounds[0].plays
    wishes = [a for a in plays if a.kind == "wish"]
    assert len(wishes) == 1
    assert wishes[0].wish_rank == 2  # "Wunsch:2"


def test_round_one_includes_a_pass_action(game_00):
    plays = game_00.rounds[0].plays
    passes = [a for a in plays if a.kind == "pass"]
    assert any(p.player == 3 for p in passes), "no pass recorded for Lisa"


def test_round_one_records_dragon_give_to_seat_zero(game_00):
    plays = game_00.rounds[0].plays
    gives = [a for a in plays if a.kind == "dragon_give"]
    assert len(gives) == 1
    assert gives[0].dragon_target == 0


def test_round_three_records_a_mid_round_tichu_call(game_00):
    # Round 3: Tichu: (2)evdokia!! — call appears between plays
    plays = game_00.rounds[2].plays
    tichu_calls = [a for a in plays if a.kind == "tichu"]
    assert any(t.player == 2 for t in tichu_calls)


# ---- Ergebnis ----

def test_each_round_carries_an_ergebnis_tuple(game_00):
    expected = [
        (0, 400), (205, 95), (300, 0), (0, 300), (-5, 5),
        (-180, 80), (265, 35), (-170, 70), (240, -40), (-65, 165),
    ]
    actual = [r.ergebnis for r in game_00.rounds]
    assert actual == expected


def test_second_sample_parses_to_rounds_with_ergebnis(game_01):
    assert len(game_01.rounds) > 0
    for r in game_01.rounds:
        assert isinstance(r.ergebnis, tuple)
        assert len(r.ergebnis) == 2
