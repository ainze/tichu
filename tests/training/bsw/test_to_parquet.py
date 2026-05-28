"""Verify Parquet shards are produced with the expected schema."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tichu_training.bsw.parser import parse_tch
from tichu_training.bsw.to_parquet import stream_to_parquet


_SAMPLES = Path(__file__).resolve().parents[3] / "sample"
_EXPECTED_DECISION_TYPES = {
    "play", "schupfen", "call_tichu", "call_grand_tichu", "wish", "dragon_assignment"
}
_EXPECTED_COLUMNS = {
    "decision_type", "game_id", "round_id", "timestamp", "player_handle",
    "action_taken", "state", "legal_actions_mask", "round_outcome",
    "round_won", "game_won", "featurizer_version", "action_space_version",
    "skill_decile", "sample_weight",
}


@pytest.fixture
def games():
    return [
        parse_tch((_SAMPLES / "2417500.tch").read_text(encoding="utf-8"), game_id="2417500"),
        parse_tch((_SAMPLES / "2417501.tch").read_text(encoding="utf-8"), game_id="2417501"),
    ]


def test_creates_one_parquet_per_decision_type(tmp_path, games):
    stream_to_parquet(games, tmp_path)
    written = {p.stem for p in tmp_path.glob("*.parquet")}
    assert written == {f"{t}_00000" for t in _EXPECTED_DECISION_TYPES}


def test_each_shard_has_the_required_schema(tmp_path, games):
    stream_to_parquet(games, tmp_path)
    for path in tmp_path.glob("*.parquet"):
        table = pq.read_table(path)
        assert set(table.column_names) == _EXPECTED_COLUMNS, (
            f"{path.name} schema missing: {_EXPECTED_COLUMNS - set(table.column_names)}"
        )


def test_complete_game_stamps_game_won_team_relative(tmp_path):
    """Per ADR-0013-era game-outcome target: every row of a Complete Game
    carries `game_won` from the perspective of the row's acting player's
    team. Sample 2417500: cumulative ergebnis = (590, 1110) so team-1 wins.
    Seats: evi_sea (0, team 0), 1David (1, team 1), evdokia!! (2, team 0),
    Lisaaaaaaa (3, team 1)."""
    game = parse_tch(
        (_SAMPLES / "2417500.tch").read_text(encoding="utf-8"), game_id="2417500",
    )
    stream_to_parquet([game], tmp_path)
    table = pq.read_table(tmp_path / "play_00000.parquet")
    by_handle: dict[str, set] = {}
    for h, g in zip(
        table.column("player_handle").to_pylist(),
        table.column("game_won").to_pylist(),
    ):
        by_handle.setdefault(h, set()).add(g)
    # team-0 lost, team-1 won; each handle's value is constant across its rows.
    assert by_handle["evi_sea"] == {False}
    assert by_handle["evdokia!!"] == {False}
    assert by_handle["1David"] == {True}
    assert by_handle["Lisaaaaaaa"] == {True}


def test_incomplete_session_stamps_null_game_won(tmp_path):
    """A ParsedGame whose summed ergebnis never crosses 1000 is an
    Incomplete Session — game_won stays NULL on every row so the AWR
    game-outcome value-baseline fit can filter it out without lying about
    who won. Sample 2417500 truncated to 4 rounds: totals (505, 795)."""
    from dataclasses import replace
    full = parse_tch(
        (_SAMPLES / "2417500.tch").read_text(encoding="utf-8"), game_id="2417500",
    )
    truncated = replace(full, rounds=full.rounds[:4])
    stream_to_parquet([truncated], tmp_path)
    table = pq.read_table(tmp_path / "play_00000.parquet")
    values = table.column("game_won").to_pylist()
    assert values, "expected at least one play row"
    assert all(v is None for v in values), (
        f"expected all NULL for Incomplete Session, got distinct values: {set(values)}"
    )


def test_stream_stats_counts_complete_and_incomplete(tmp_path):
    """Parse-time diagnostic: surface how many games crossed 1000 vs how
    many were abandoned mid-game. Lets the A/B comparison report the
    fraction of corpus the game-outcome target actually fits on."""
    from dataclasses import replace
    full_a = parse_tch(
        (_SAMPLES / "2417500.tch").read_text(encoding="utf-8"), game_id="2417500",
    )
    full_b = parse_tch(
        (_SAMPLES / "2417501.tch").read_text(encoding="utf-8"), game_id="2417501",
    )
    incomplete = replace(full_a, game_id="incomplete", rounds=full_a.rounds[:4])
    stats = stream_to_parquet([full_a, full_b, incomplete], tmp_path)
    assert stats.complete_games == 2
    assert stats.incomplete_sessions == 1


def test_schema_includes_game_won_column(tmp_path, games):
    """v3 schema: game_won is the team-relative game-level outcome label
    used by AWR when value_target='game'. See ADR-0013 for why the schema
    change is signalled by directory name, not by a featurizer_version bump."""
    stream_to_parquet(games, tmp_path)
    for path in tmp_path.glob("*.parquet"):
        table = pq.read_table(path)
        assert "game_won" in table.column_names, (
            f"{path.name} missing game_won column"
        )
        # bool_ allows nulls in Arrow; the Incomplete Session case relies on it.
        assert table.schema.field("game_won").type == pa.bool_()


def test_play_shard_contains_play_and_pass_rows(tmp_path, games):
    stream_to_parquet(games, tmp_path)
    table = pq.read_table(tmp_path / "play_00000.parquet")
    assert table.num_rows > 0
    decision_types = set(table.column("decision_type").to_pylist())
    assert decision_types == {"play"}
    actions = set(table.column("action_taken").to_pylist())
    assert any(a == "pass" for a in actions), "pass actions should be in the play shard"
    assert any(a.startswith("play:") for a in actions)


def test_schupfen_shard_contains_schupfen_rows(tmp_path, games):
    stream_to_parquet(games, tmp_path)
    table = pq.read_table(tmp_path / "schupfen_00000.parquet")
    # Per-round filtering (ADR-0009): each matching round contributes 4
    # schupfen submissions. Total is 4 × matched_rounds; at least one round
    # must match for there to be any schupfen rows at all.
    assert table.num_rows > 0
    assert table.num_rows % 4 == 0
    assert table.num_rows <= 4 * 10 * 2
    actions = table.column("action_taken").to_pylist()
    assert all(a.startswith("schupfen:") for a in actions)


def test_player_handle_column_uses_per_round_handles_for_substituted_seat(tmp_path):
    """When a seat's handle changes between rounds (BSW player substitution),
    each parquet row records the handle that was at the seat *during that
    round*. The previous game-level snapshot would have mis-attributed all
    rounds' decisions to the round-0 handle. See ADR-0010."""
    raw = (_SAMPLES / "2417500.tch").read_text(encoding="utf-8")
    # Substitute seat 1 from "1David" to "NewPlayer" starting round 1.
    lines = raw.splitlines()
    out: list[str] = []
    in_round_zero = True
    for line in lines:
        out.append(line if in_round_zero else line.replace("1David", "NewPlayer"))
        if line.startswith("Ergebnis:") and in_round_zero:
            in_round_zero = False
    spliced = "\n".join(out) + "\n"
    game = parse_tch(spliced, game_id="sub")

    stream_to_parquet([game], tmp_path)
    table = pq.read_table(tmp_path / "play_00000.parquet")

    handles = set(table.column("player_handle").to_pylist())
    # Both identities of seat 1 must appear in the play shard — the previous
    # `game.handles`-based attribution would only have emitted "1David".
    assert "1David" in handles, "round-0 seat-1 handle missing"
    assert "NewPlayer" in handles, (
        "seat-1 substitution lost — to_parquet is still reading game-level handles"
    )


def test_player_handles_match_seat_assignments(tmp_path, games):
    stream_to_parquet(games, tmp_path)
    table = pq.read_table(tmp_path / "play_00000.parquet")
    handles = set(table.column("player_handle").to_pylist())
    # Game 00 seats: evi_sea, 1David, evdokia!!, Lisaaaaaaa.
    # Game 01 seats: evi_sea, 1David, evdokia!!, Steffi0722.
    expected = {"evi_sea", "1David", "evdokia!!", "Lisaaaaaaa", "Steffi0722"}
    assert handles == expected


def test_state_and_legal_actions_mask_are_null(tmp_path, games):
    """v1 defers featurization to load time; columns exist but stay null."""
    stream_to_parquet(games, tmp_path)
    table = pq.read_table(tmp_path / "play_00000.parquet")
    assert all(v is None for v in table.column("state").to_pylist())
    assert all(v is None for v in table.column("legal_actions_mask").to_pylist())


def test_version_columns_stamped_from_constants(tmp_path, games):
    from tichu_training.action_space import ACTION_SPACE_VERSION
    from tichu_training.featurizer import FEATURIZER_VERSION
    stream_to_parquet(games, tmp_path)
    table = pq.read_table(tmp_path / "play_00000.parquet")
    fv = set(table.column("featurizer_version").to_pylist())
    av = set(table.column("action_space_version").to_pylist())
    assert fv == {FEATURIZER_VERSION}
    assert av == {ACTION_SPACE_VERSION}


def test_skill_decile_is_null_without_ratings(tmp_path, games):
    stream_to_parquet(games, tmp_path)
    table = pq.read_table(tmp_path / "play_00000.parquet")
    assert all(v is None for v in table.column("skill_decile").to_pylist())


def test_sample_weight_full_for_post_2015_games(tmp_path, games):
    # Sample games (2417500, 2417501) are well past the default cutoff (1855844).
    stream_to_parquet(games, tmp_path)
    table = pq.read_table(tmp_path / "play_00000.parquet")
    weights = table.column("sample_weight").to_pylist()
    assert weights and all(w == 1.0 for w in weights)


def test_sample_weight_downweighted_below_cutoff(tmp_path, games):
    stream_to_parquet(games, tmp_path, recency_cutoff_game_id=99999999, recency_weight=0.25)
    table = pq.read_table(tmp_path / "play_00000.parquet")
    weights = table.column("sample_weight").to_pylist()
    assert weights and all(w == 0.25 for w in weights)


def test_skill_decile_joined_from_ratings(tmp_path, games):
    """Build a tiny ratings table and verify the join."""
    import pyarrow as pa
    ratings_path = tmp_path / "ratings.parquet"
    pq.write_table(
        pa.table({
            "player_handle": ["evi_sea", "1David"],
            "skill_decile": pa.array([7, 3], type=pa.int32()),
        }),
        ratings_path,
    )
    stream_to_parquet(games, tmp_path, ratings_path=ratings_path)
    table = pq.read_table(tmp_path / "play_00000.parquet")
    by_handle = {h: d for h, d in zip(
        table.column("player_handle").to_pylist(),
        table.column("skill_decile").to_pylist(),
    )}
    assert by_handle["evi_sea"] == 7
    assert by_handle["1David"] == 3
    # Unknown handle should map to null.
    assert by_handle.get("evdokia!!") is None


def test_parquet_files_stay_readable_when_iterator_raises_mid_stream(tmp_path, games):
    """try/finally invariant: a source that yields then raises must still leave
    valid, footer-written Parquet on disk (no ParquetWriter left dangling)."""
    from tichu_training.bsw.to_parquet import stream_to_parquet as _stream

    def raising_iter():
        yield games[0]
        raise RuntimeError("simulated mid-stream failure")

    with pytest.raises(RuntimeError, match="simulated mid-stream failure"):
        _stream(raising_iter(), tmp_path, rows_per_flush=1)

    # Every parquet shard that was created must be readable end-to-end.
    shards = list(tmp_path.glob("*.parquet"))
    assert shards, "expected at least one shard to have been opened"
    for shard in shards:
        pq.read_table(shard)  # raises if footer is missing


def test_row_counts_returned_match_disk(tmp_path, games):
    stats = stream_to_parquet(games, tmp_path)
    for decision_type, n in stats.row_counts.items():
        if n == 0:
            continue
        table = pq.read_table(tmp_path / f"{decision_type}_00000.parquet")
        assert table.num_rows == n


def test_stream_raw_parses_inside_workers_and_records_parse_failures(tmp_path):
    """`stream_raw_to_parquet` parses .tch text in workers; parse failures are
    surfaced in StreamStats.parse_failures (not raised) so the dispatch loop
    keeps draining the iterator and the progress bar still advances."""
    from tichu_training.bsw.to_parquet import stream_raw_to_parquet

    good_text = (_SAMPLES / "2417500.tch").read_text(encoding="utf-8")
    pairs = [
        ("2417500", good_text),
        ("garbage", "this is not a .tch file at all"),
    ]
    callbacks = 0
    def _on_done(_stats):
        nonlocal callbacks
        callbacks += 1

    stats = stream_raw_to_parquet(pairs, tmp_path, workers=1, on_game_done=_on_done)
    assert callbacks == 2
    assert stats.parse_failures == ["garbage"]
    assert stats.games_fully_matched + stats.games_with_failed_rounds == 1
    # Real game still produced rows.
    table = pq.read_table(tmp_path / "play_00000.parquet")
    assert table.num_rows > 0


def test_multi_worker_matches_single_worker(tmp_path, games):
    """workers>1 must produce the same row counts and fire on_game_done per
    game (not per chunk). Previously `pool.map(chunksize=16)` batched results
    so on_game_done was never invoked for runs smaller than one chunk."""
    single_dir = tmp_path / "single"
    multi_dir = tmp_path / "multi"

    callback_count = 0
    def _on_done(_stats):
        nonlocal callback_count
        callback_count += 1

    single_stats = stream_to_parquet(games, single_dir, workers=1, on_game_done=_on_done)
    assert callback_count == len(games)

    callback_count = 0
    multi_stats = stream_to_parquet(games, multi_dir, workers=2, on_game_done=_on_done)
    assert callback_count == len(games)

    assert single_stats.row_counts == multi_stats.row_counts
    assert single_stats.games_fully_matched == multi_stats.games_fully_matched
    assert single_stats.rounds_matched == multi_stats.rounds_matched
