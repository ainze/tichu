"""Position-slice helper for eval_matrix: n_offset lets parallel jobs cover
disjoint Pool ranges (behavioral mode is serial, so we fan out by slice).
An off-by-one here would overlap ranges and bias a pooled dial."""

from tichu_training.cli.eval_matrix import _slice_positions


def test_offset_and_cap_select_a_disjoint_window():
    xs = list(range(100))
    assert _slice_positions(xs, 10, 30) == list(range(30, 40))
    assert _slice_positions(xs, 10, 0) == list(range(0, 10))


def test_adjacent_offsets_are_disjoint_and_contiguous():
    xs = list(range(100))
    a = _slice_positions(xs, 10, 0)
    b = _slice_positions(xs, 10, 10)
    assert set(a).isdisjoint(b)
    assert a + b == list(range(20))


def test_no_cap_takes_the_tail_from_offset():
    xs = list(range(100))
    assert _slice_positions(xs, None, 30) == list(range(30, 100))
    assert _slice_positions(xs, None, 0) == xs
