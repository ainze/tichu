"""Behavioral Drift Benchmark — the HTML report rendered from the tidy panel."""

import re

import numpy as np
import pandas as pd

from tichu_eval.drift_report import render_report


def _rows(metric, family, cell, *, bc, subject, delta, lo, hi, bh=False, suppressed=False,
          conditional=True):
    base = {"metric": metric, "family": family, "cell": cell, "conditional": conditional,
            "suppressed": suppressed, "events": 1, "opportunities": 500,
            "exposure": 10.0, "exposure_lo": 9.0, "exposure_hi": 11.0, "p": 0.01}
    nan = np.nan
    return [
        {**base, "arm": "bc", "rate": nan if suppressed else bc, "rate_lo": nan, "rate_hi": nan,
         "bh_survives": False},
        {**base, "arm": "subject", "rate": nan if suppressed else subject, "rate_lo": nan,
         "rate_hi": nan, "bh_survives": False},
        {**base, "arm": "delta", "rate": nan if suppressed else delta,
         "rate_lo": nan if suppressed else lo, "rate_hi": nan if suppressed else hi,
         "bh_survives": bh},
    ]


PANEL = pd.DataFrame(
    _rows("dog_lead", "D. Play — leading", "partner called",
          bc=0.71, subject=0.48, delta=-0.23, lo=-0.27, hi=-0.19, bh=True)
    + _rows("pass_despite_beat", "E. Play — following", "partner",
            bc=0.60, subject=0.65, delta=0.05, lo=0.04, hi=0.06, bh=True)
    + _rows("pass_despite_beat", "E. Play — following", "opponent",
            bc=0.30, subject=0.31, delta=0.01, lo=-0.02, hi=0.04)
    + _rows("wish_fulfilled", "F. Wish", "all", bc=0, subject=0, delta=0, lo=0, hi=0,
            suppressed=True)
    + _rows("slam", "A. Round outcome", "for", bc=0.05, subject=0.06, delta=0.01,
            lo=0.0, hi=0.02, conditional=False)
)
PROV = {"subject": "iter_15360 <champion>", "bc": "v7 warm-start", "deals": 20000}


def _summary(html):
    return html.split('id="drift-summary"')[1].split("</section>")[0]


def test_drift_summary_lists_only_bh_survivors_largest_effect_first():
    summary = _summary(render_report(PANEL, PROV))
    assert "dog_lead" in summary and "pass_despite_beat" in summary
    assert "opponent" not in summary                      # not a BH survivor
    # Ranked by |Δ|/SE, with SE read off the 95% CI width.
    se = lambda lo, hi: (hi - lo) / 3.92
    order = sorted([("dog_lead", 0.23 / se(-0.27, -0.19)), ("pass_despite_beat", 0.05 / se(0.04, 0.06))],
                   key=lambda t: -t[1])
    assert summary.index(order[0][0]) < summary.index(order[1][0])


def test_families_appear_in_catalogue_order():
    html = render_report(PANEL, PROV)
    heads = re.findall(r"<h2[^>]*>([A-G])\. ", html)
    assert heads == sorted(heads) and set(heads) == {"A", "D", "E", "F"}


def test_suppressed_cells_read_n_too_small():
    html = render_report(PANEL, PROV)
    wish = html.split('id="m-wish_fulfilled"')[1].split("</table>")[0]
    assert "n too small" in wish


def test_unconditional_metrics_show_no_exposure():
    html = render_report(PANEL, PROV)
    slam = html.split('id="m-slam"')[1].split("</table>")[0]
    assert "Exposure" not in slam
    dog = html.split('id="m-dog_lead"')[1].split("</table>")[0]
    assert "Exposure" in dog


def test_provenance_is_escaped():
    html = render_report(PANEL, PROV)
    assert "iter_15360 &lt;champion&gt;" in html and "<champion>" not in html


def _ref(metric, family, cell, arm, rate, suppressed=False, conditional=True):
    return {"metric": metric, "family": family, "cell": cell, "arm": arm, "conditional": conditional,
            "suppressed": suppressed, "events": 1, "opportunities": 500, "exposure": 9.0,
            "exposure_lo": 8.0, "exposure_hi": 10.0, "rate": np.nan if suppressed else rate,
            "rate_lo": np.nan if suppressed else rate - 0.01,
            "rate_hi": np.nan if suppressed else rate + 0.01, "p": np.nan, "bh_survives": False}


WITH_HUMANS = pd.concat([PANEL, pd.DataFrame([
    _ref("dog_lead", "D. Play — leading", "partner called", "humans (all)", 0.62),
    _ref("dog_lead", "D. Play — leading", "partner called", "humans (top 10%)", 0.66),
    _ref("wish_fulfilled", "F. Wish", "all", "humans (all)", 0, suppressed=True),
])], ignore_index=True)


def test_reference_columns_sit_before_bc_in_every_table():
    html = render_report(WITH_HUMANS, PROV)
    dog = html.split('id="m-dog_lead"')[1].split("</table>")[0]
    head = dog.split("</thead>")[0]
    assert head.index("humans (all)") < head.index("humans (top 10%)") < head.index(">BC<")
    assert "62.0%" in dog and "66.0%" in dog


def test_reference_columns_show_a_dash_when_missing_or_suppressed():
    html = render_report(WITH_HUMANS, PROV)
    pass_ = html.split('id="m-pass_despite_beat"')[1].split("</table>")[0]
    assert "humans (all)" in pass_ and "–" in pass_          # no human row for this metric


def test_drift_summary_carries_the_reference_rates():
    summary = _summary(render_report(WITH_HUMANS, PROV))
    assert "humans (top 10%)" in summary and "66.0%" in summary


def test_no_references_no_extra_columns():
    html = render_report(PANEL, PROV)
    assert "humans" not in html


def test_numeric_headers_align_with_their_right_aligned_values():
    # Every header over a numeric column must be right-aligned like its values;
    # only the label columns (Metric, Cell) stay left.
    html = render_report(WITH_HUMANS, PROV)
    for table in re.findall(r"<table>(.*?)</table>", html, re.S):
        head, body = table.split("</thead>")
        ths = re.findall(r"<th([^>]*)>([^<]*)</th>", head)
        first_row = re.findall(r"<td([^>]*)>", body.split("</tr>")[0])
        if any("colspan" in c for c in first_row):
            continue
        for (th_attrs, label), td_attrs in zip(ths, first_row):
            assert ("num" in th_attrs) == ("num" in td_attrs), (label, th_attrs, td_attrs)
