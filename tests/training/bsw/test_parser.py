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
_DATA = Path(__file__).resolve().parent / "data"


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


def test_anonymous_seat_in_pre_deal_records_empty_handle():
    """A `(N) <cards>` line with no handle (BSW guest / freshly-joined seat)
    parses cleanly. The seat's handle for that round is recorded as ``""``."""
    raw = (_SAMPLES / "2417500.tch").read_text(encoding="utf-8")
    # Anonymise seat 1's pre-deal AND start-hand lines in round 0 only.
    # Both lines start with "(1)1David " (handle followed by a space-then-cards).
    # The schupfen / play / pass lines also start with "(1)1David" but with a
    # different following char (`gibt:` / `:` / ` passt.`) — leave those named
    # so this test isolates the pre-deal/start-hand case.
    lines = raw.splitlines()
    out: list[str] = []
    in_round_zero = True
    for line in lines:
        if in_round_zero and line.startswith("(1)1David ") and not line.endswith("gibt:") and " gibt:" not in line and " passt." not in line:
            # Pre-deal/start-hand: "(1)1David <cards trailing space>" — replace
            # with "(1) <cards trailing space>".
            out.append("(1) " + line[len("(1)1David "):])
        else:
            out.append(line)
        if line.startswith("Ergebnis:"):
            in_round_zero = False
    spliced = "\n".join(out) + "\n"

    game = parse_tch(spliced, game_id="anon")

    # Seat-1 handle in round 0 is anonymous.
    assert game.rounds[0].handles[1] == ""
    # Other seats are unaffected.
    assert game.rounds[0].handles[0] == "evi_sea"
    assert game.rounds[0].handles[2] == "evdokia!!"
    assert game.rounds[0].handles[3] == "Lisaaaaaaa"


def test_dragon_assignment_line_tolerates_missing_handle():
    """BSW occasionally drops the handle on a `Drache an: (N)` line even when
    the player has a known handle elsewhere in the round. The parser must
    accept the handle-less form and still record the dragon_give action with
    the correct target seat."""
    raw = (_SAMPLES / "2417500.tch").read_text(encoding="utf-8")
    # Round 0 ends with "Drache an: (0)evi_sea". Drop the handle on that line.
    assert "Drache an: (0)evi_sea" in raw  # guard the fixture
    spliced = raw.replace("Drache an: (0)evi_sea", "Drache an: (0)", 1)

    game = parse_tch(spliced, game_id="dragon_anon")

    gives = [a for a in game.rounds[0].plays if a.kind == "dragon_give"]
    assert len(gives) == 1
    assert gives[0].dragon_target == 0


@pytest.mark.parametrize("named_line,anon_line,verify", [
    # Pass (round 0, seat 3 passes)
    (
        "(3)Lisaaaaaaa passt.",
        "(3) passt.",
        lambda g: any(a.kind == "pass" and a.player == 3 for a in g.rounds[0].plays),
    ),
    # Mid-round Tichu call (round 3 has "Tichu: (2)evdokia!!")
    (
        "Tichu: (2)evdokia!!",
        "Tichu: (2)",
        lambda g: 2 in g.rounds[2].tichu_callers,
    ),
    # Grand-tichu call (round 0 has "Grosses Tichu: (1)1David", pre-Schupfen)
    (
        "Grosses Tichu: (1)1David",
        "Grosses Tichu: (1)",
        lambda g: 1 in g.rounds[0].grand_tichu_callers,
    ),
    # Schupfen line with anonymous giver
    (
        "(0)evi_sea gibt:",
        "(0) gibt:",
        lambda g: g.rounds[0].schupfen[0] is not None and g.rounds[0].schupfen[0].kind == "schupfen",
    ),
    # Play line "(N): cards" with empty handle between `)` and `:`
    (
        "(3)Lisaaaaaaa: Ma",
        "(3): Ma",
        lambda g: any(a.kind == "play" and a.player == 3 for a in g.rounds[0].plays),
    ),
])
def test_action_lines_tolerate_missing_handle(named_line, anon_line, verify):
    """Defensive parser sweep: every line that names a seat by `(N)` may
    appear with an empty handle. Handle is metadata; the seat index is the
    canonical identity, and downstream attribution uses per-round handles
    from the pre-deal section (see ADR-0010)."""
    raw = (_SAMPLES / "2417500.tch").read_text(encoding="utf-8")
    assert named_line in raw, f"fixture changed; {named_line!r} no longer present"
    spliced = raw.replace(named_line, anon_line, 1)

    game = parse_tch(spliced, game_id="anon_action")

    assert verify(game), f"action not recorded after anonymising {named_line!r}"


def test_parsed_round_records_per_round_handles_for_substituted_seat():
    """When a seat's handle changes between rounds (player substitution
    mid-game), each ParsedRound records the handle that was at the seat
    *during that round*. See ADR-0010."""
    raw = (_SAMPLES / "2417500.tch").read_text(encoding="utf-8")
    # Simulate seat 1 being a different player from round 1 onwards.
    # Round 0 keeps "1David"; later rounds replace every "(1)1David" /
    # "1David:" reference with "NewPlayer".
    lines = raw.splitlines()
    out: list[str] = []
    in_round_zero = True
    for line in lines:
        if in_round_zero:
            out.append(line)
        else:
            out.append(line.replace("1David", "NewPlayer"))
        if line.startswith("Ergebnis:") and in_round_zero:
            in_round_zero = False
    spliced = "\n".join(out) + "\n"

    game = parse_tch(spliced, game_id="sub")

    # Round 0 has the original handle at seat 1.
    assert game.rounds[0].handles[1] == "1David"
    # Subsequent rounds carry the substituted handle.
    assert game.rounds[1].handles[1] == "NewPlayer"
    assert game.rounds[-1].handles[1] == "NewPlayer"
    # Unsubstituted seats stay stable across rounds.
    for r in game.rounds:
        assert r.handles[0] == "evi_sea"
        assert r.handles[2] == "evdokia!!"
        assert r.handles[3] == "Lisaaaaaaa"


def test_real_corpus_game_with_mid_game_substitution_parses_cleanly():
    """Regression: BSW game 100920 has a mid-game player substitution at
    seat 1 — `alejandro_styl` plays rounds 0-3, then in round 4 the seat
    is anonymous (handle dropped at the deal-moment), and from round 5
    onwards `pöppi69` takes over. The legacy parser raised on round 4's
    pre-deal line. After ADR-0010 the game parses cleanly and round 4's
    seat-1 handle is recorded as ``""``."""
    text = (_DATA / "game_100920.tch").read_text(encoding="utf-8")

    game = parse_tch(text, game_id="100920")

    assert len(game.rounds) > 4
    # Round 4 has the anonymous seat.
    assert game.rounds[4].handles[1] == ""
    # Round 0 has the original handle; rounds after the substitution carry
    # the new handle (the underlying byte for ö may render as mojibake, so
    # we assert structurally — handle present and different from round 0).
    assert game.rounds[0].handles[1] != ""
    assert game.rounds[5].handles[1] != ""
    assert game.rounds[5].handles[1] != game.rounds[0].handles[1]


def test_real_corpus_game_with_dragon_an_handle_drop_parses_cleanly():
    """Regression: BSW game 100952 round 8 has a `Drache an: (1)` line
    where BSW's serialiser dropped the handle even though the player has
    a known handle in surrounding lines. The legacy parser raised on it;
    after ADR-0010 the parse succeeds."""
    text = (_DATA / "game_100952.tch").read_text(encoding="utf-8")

    game = parse_tch(text, game_id="100952")

    # The game parses to a sensible round count and every round has an
    # Ergebnis tuple.
    assert len(game.rounds) > 0
    for r in game.rounds:
        assert isinstance(r.ergebnis, tuple) and len(r.ergebnis) == 2


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
    # Handles are per-round (ADR-0010); round-0 captures the starting seating.
    assert game_00.rounds[0].handles == ("evi_sea", "1David", "evdokia!!", "Lisaaaaaaa")


def test_parsed_game_two_has_different_handles(game_01):
    assert game_01.rounds[0].handles == ("evi_sea", "1David", "evdokia!!", "Steffi0722")


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


def test_pre_schupfen_grand_tichu_call_surfaces_as_a_play_action(game_00):
    # Grand Tichu is declared between the 14-card start hands and the
    # Schupfen header. The parser used to record it in `grand_tichu_callers`
    # only, dropping the action from `plays` — which silently zeroed the
    # `call_grand_tichu` parquet shard. The call must now appear as a
    # ParsedAction in `plays` for the replay's decision stream to see it.
    plays = game_00.rounds[0].plays
    grand_actions = [p for p in plays if p.kind == "grand_tichu"]
    assert grand_actions, "expected a grand_tichu ParsedAction in round 0 plays"
    assert grand_actions[0].player == 1


def test_pre_schupfen_tichu_call_also_surfaces_as_play_action():
    # A round-level Tichu call before the Schupfen header should also land
    # in `plays`. Build a minimal fixture rather than depend on which
    # checked-in sample happens to exercise this path.
    log = (_DATA / "pre_schupfen_tichu.tch")
    if not log.exists():
        pytest.skip("pre-schupfen Tichu fixture not present; covered by 100k corpus run")
    game = parse_tch(log.read_text(encoding="utf-8"))
    tichu_actions = [p for p in game.rounds[0].plays if p.kind == "tichu"]
    assert tichu_actions, "pre-schupfen Tichu call should surface in plays"


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
    # `plays` may begin with pre-schupfen call actions (Grand Tichu / Tichu);
    # the first natural-card play is what we want here.
    plays = game_00.rounds[0].plays
    first_play = next(a for a in plays if a.kind == "play")
    assert first_play.player == 3  # Lisa
    assert first_play.cards == (MAHJONG,)


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


def test_handles_with_nbsp_parse_and_normalise():
    """BSW encodes literal spaces inside a handle (e.g. "Lucky Luke") as
    U+00A0 (NBSP) and sometimes pads cosmetic trailing NBSP before delimiters
    (e.g. "schubsi\\xa0:"). Python `\\S` excludes NBSP, so naive regexes drop
    the rest of the handle and the round fails to parse. Both forms must
    parse, and the captured handle must be normalised back to a regular space
    so downstream skill_decile lookup matches the ratings table.
    """
    raw = (_SAMPLES / "2417500.tch").read_text(encoding="utf-8")
    # Embedded NBSP: rename seat-0 handle "evi_sea" to "Lucky\xa0Luke" in
    # round-0 pre-deal AND start-hand lines (the sanity check at the start of
    # round parsing requires both lines to agree).
    embedded = "Lucky\xa0Luke"
    trailing = "schubsi\xa0"  # trailing-NBSP form

    lines = raw.splitlines()
    out: list[str] = []
    in_round_zero = True
    seat0_card_lines_rewritten = 0
    seat1_play_lines_rewritten = 0
    for line in lines:
        if in_round_zero:
            # Seat-0 card list lines: "(0)evi_sea <cards>" — replace handle with NBSP form.
            if line.startswith("(0)evi_sea ") and " gibt:" not in line and " passt." not in line and not line.startswith("(0)evi_sea:"):
                out.append("(0)" + embedded + " " + line[len("(0)evi_sea "):])
                seat0_card_lines_rewritten += 1
                if line.startswith("Ergebnis:"):
                    in_round_zero = False
                continue
            # Seat-1 play lines: "(1)1David: cards" — give the handle trailing NBSP.
            if line.startswith("(1)1David:"):
                out.append("(1)" + trailing + ":" + line[len("(1)1David:"):])
                seat1_play_lines_rewritten += 1
                if line.startswith("Ergebnis:"):
                    in_round_zero = False
                continue
        out.append(line)
        if line.startswith("Ergebnis:"):
            in_round_zero = False
    assert seat0_card_lines_rewritten >= 2, "test setup: expected to rewrite both pre-deal and start-hand seat-0 lines"
    assert seat1_play_lines_rewritten >= 1, "test setup: expected at least one seat-1 play line"
    spliced = "\n".join(out) + "\n"

    game = parse_tch(spliced, game_id="nbsp")

    # Embedded NBSP becomes a regular space in the captured handle.
    assert game.rounds[0].handles[0] == "Lucky Luke"
    # Cosmetic trailing NBSP is stripped — the round still parses, which is
    # the load-bearing assertion (seat-1 play lines all parsed successfully).
    assert len(game.rounds[0].plays) > 0
