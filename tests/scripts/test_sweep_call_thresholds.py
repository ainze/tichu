"""build_variants must emit one single-threshold change per swept value, hold the other axis
at baseline, and skip values that equal the baseline. (The torch/tournament orchestration in
main is exercised by running the script; the pure variant logic is unit-tested here.)
"""

from scripts.sweep_call_thresholds import build_variants


def test_default_grid_is_eight_single_axis_variants():
    v = build_variants([0.3, 0.4, 0.6, 0.7], [0.3, 0.4, 0.6, 0.7], 0.5, 0.5)
    assert len(v) == 8
    tichu_variants = [x for x in v if x[0].startswith("tichu")]
    grand_variants = [x for x in v if x[0].startswith("grand")]
    assert len(tichu_variants) == 4 and len(grand_variants) == 4
    # tichu variants hold grand at baseline; grand variants hold tichu at baseline.
    assert all(g == 0.5 for _, t, g in tichu_variants)
    assert all(t == 0.5 for _, t, g in grand_variants)


def test_baseline_value_is_skipped():
    v = build_variants([0.5, 0.6], [0.5], 0.5, 0.5)
    assert v == [("tichu0.60", 0.6, 0.5)]


def test_labels_and_threshold_assignment():
    v = build_variants([0.3], [0.7], 0.5, 0.5)
    assert ("tichu0.30", 0.3, 0.5) in v
    assert ("grand0.70", 0.5, 0.7) in v
