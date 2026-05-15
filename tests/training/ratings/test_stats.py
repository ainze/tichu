"""Spearman rank correlation implementation."""

import pytest

from tichu_training.ratings.stats import spearman_rho


def test_known_value_no_ties():
    # rank_x = [1,2,3,4], rank_y = [2,1,4,3], d^2 sum = 4 → rho = 0.6
    assert spearman_rho([1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]) == pytest.approx(0.6)


def test_perfect_positive():
    assert spearman_rho([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)


def test_perfect_negative():
    assert spearman_rho([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)


def test_average_ranks_on_ties():
    # Two values tied at the top of x; identical ties at top of y → rho = 1.0
    assert spearman_rho([1, 2, 2, 4], [10, 20, 20, 40]) == pytest.approx(1.0)


def test_returns_none_on_insufficient_input():
    assert spearman_rho([], []) is None
    assert spearman_rho([1.0], [2.0]) is None
