"""Behavioral Drift Benchmark — the HTML report.

Rendered purely from the tidy panel (`drift_metrics.summarise`), so the CSV stays
the source of truth and a report can be rebuilt from it at any time. Layout:

* a provenance header (Checkpoints, Pool, deal count, git SHA) and the standing
  caveats — descriptive not causal, Synthetic Games valid only for score-blind nets;
* the **Drift Summary**: every Δ that survives Benjamini–Hochberg, largest
  |Δ| / SE first — where the Subject moved furthest from its BC;
* one section per Decision family, one table per metric, one row per cell:
  Exposure (conditional metrics only) and the Conditional Rate for BC and
  Subject, the paired Δ with its 95% CI, and a paired bar for eyeballing.

Self-contained (inline CSS, no scripts) so it opens from disk anywhere.
"""

from __future__ import annotations

import html
import math

import pandas as pd

from tichu_eval.drift_metrics import METRICS

# Cells whose value is a mean (points, rounds, bombs), not a proportion.
_MEAN_METRICS = {"round_points", "call_bonus", "bombs_per_round"}
_MEAN_GAME_CELLS = {"rounds_per_game", "margin"}
_GAME_DESCRIPTION = ("Synthetic Games (Round results chained to 1000): rounds per Game, "
                     "Subject win rate, final margin.")


_AGENT_ARMS = ("bc", "subject", "delta")


def render_report(panel: pd.DataFrame, provenance: dict, title: str = "Behavioral Drift") -> str:
    catalogue = {m.name: m for m in METRICS}
    refs = [a for a in dict.fromkeys(panel.arm) if a not in _AGENT_ARMS]
    sections = []
    for family, fam in panel.groupby("family", sort=True):
        blocks = [_metric_block(name, fam[fam.metric == name], catalogue.get(name), refs)
                  for name in dict.fromkeys(fam[fam.arm.isin(_AGENT_ARMS)].metric)]
        sections.append(f'<section class="family"><h2>{_e(family)}</h2>{"".join(blocks)}</section>')
    return _PAGE.format(
        title=_e(title),
        css=_CSS,
        provenance="".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in provenance.items()),
        summary=_summary(panel, refs),
        sections="".join(sections),
        reference_note=_REFERENCE_NOTE if refs else "",
    )


# --- Drift Summary -------------------------------------------------------------

def _summary(panel: pd.DataFrame, refs: list[str]) -> str:
    d = panel[(panel.arm == "delta") & panel.bh_survives.astype(bool)].copy()
    if d.empty:
        body = '<p class="muted">No Δ survives Benjamini–Hochberg at 5%.</p>'
    else:
        se = (d.rate_hi - d.rate_lo) / 3.92
        d["z"] = (d.rate.abs() / se.where(se > 0)).fillna(math.inf)
        d = d.sort_values("z", ascending=False)
        rows = []
        for r in d.itertuples():
            bc, subj = (_arm_value(panel, r.metric, r.cell, a) for a in ("bc", "subject"))
            mean = _is_mean(r.metric, r.cell)
            ref_tds = "".join(f"<td class=num>{_fmt(_arm_value(panel, r.metric, r.cell, a), mean)}</td>"
                              for a in refs)
            rows.append(
                f"<tr><td><a href=\"#m-{_e(r.metric)}\">{_e(r.metric)}</a></td><td>{_e(r.cell)}</td>"
                f"{ref_tds}<td class=num>{_fmt(bc, mean)}</td><td class=num>{_fmt(subj, mean)}</td>"
                f"<td class=num>{_delta(r.rate, r.rate_lo, r.rate_hi, mean)}</td>"
                f"<td class=num>{_z(r.z)}</td></tr>")
        body = ("<table><thead><tr><th>Metric</th><th>Cell</th>"
                + "".join(f"<th class=num>{_e(a)}</th>" for a in refs)
                + "<th class=num>BC</th><th class=num>Subject</th>"
                "<th class=num>Δ [95% CI]</th><th class=num>|Δ|/SE</th></tr></thead><tbody>"
                + "".join(rows) + "</tbody></table>")
    return f'<section id="drift-summary"><h2>Drift Summary</h2>{body}</section>'


# --- Per-metric tables ---------------------------------------------------------

def _metric_block(name: str, rows: pd.DataFrame, m, refs: list[str]) -> str:
    description = m.description if m is not None else _GAME_DESCRIPTION
    agent = rows[rows.arm.isin(_AGENT_ARMS)]
    conditional = bool(agent.conditional.iloc[0])
    head = ["Cell"]
    if conditional:
        head += ["Exposure BC", "Exposure Subject", "Δ Exposure"]
    head += [*refs, "BC", "Subject", "Δ [95% CI]", ""]
    body = []
    for cell in dict.fromkeys(agent.cell):
        c = rows[rows.cell == cell].set_index("arm")
        bc, subj, d = c.loc["bc"], c.loc["subject"], c.loc["delta"]
        mean = _is_mean(name, cell)
        tds = [f"<td>{_e(cell)}{' <span class=star title=\"survives BH\">★</span>' if d.bh_survives else ''}</td>"]
        if conditional:
            tds += [f"<td class=num>{_num(bc.exposure)}</td>", f"<td class=num>{_num(subj.exposure)}</td>",
                    f"<td class=num>{_delta(d.exposure, d.exposure_lo, d.exposure_hi, True)}</td>"]
        tds += [_ref_td(c.loc[a] if a in c.index else None, mean) for a in refs]
        if bool(d.suppressed):
            tds += ['<td class="muted" colspan=4>n too small</td>']
        else:
            tds += [f"<td class=num>{_fmt(bc.rate, mean)}</td>", f"<td class=num>{_fmt(subj.rate, mean)}</td>",
                    f"<td class=num>{_delta(d.rate, d.rate_lo, d.rate_hi, mean)}</td>",
                    f"<td>{'' if mean else _bars(bc.rate, subj.rate)}</td>"]
        body.append("<tr>" + "".join(tds) + "</tr>")
    # Label column left; every value column right-aligned like its numbers (the
    # trailing bar column carries no header and no numbers).
    ths = "".join(f"<th{' class=num' if i and h else ''}>{_e(h)}</th>" for i, h in enumerate(head))
    table = ("<table><thead><tr>" + ths + "</tr></thead><tbody>"
             + "".join(body) + "</tbody></table>")
    return (f'<article class="metric" id="m-{_e(name)}"><h3>{_e(name)}</h3>'
            f'<p class="muted">{_e(description)}</p>{table}</article>')


# --- Formatting ----------------------------------------------------------------

def _ref_td(row, mean: bool) -> str:
    """A reference (human) rate, its CI on hover; a dash when absent or suppressed."""
    if row is None or bool(row.suppressed) or math.isnan(row.rate):
        return '<td class="num ref">–</td>'
    ci = f"95% CI [{_fmt(row.rate_lo, mean)}, {_fmt(row.rate_hi, mean)}]"
    return f'<td class="num ref" title="{_e(ci)}">{_fmt(row.rate, mean)}</td>'


def _is_mean(metric: str, cell: str) -> bool:
    return metric in _MEAN_METRICS or (metric == "game" and cell in _MEAN_GAME_CELLS)


def _arm_value(panel, metric, cell, arm):
    r = panel[(panel.metric == metric) & (panel.cell == cell) & (panel.arm == arm)]
    return r.rate.iloc[0] if len(r) else math.nan


def _fmt(v, mean: bool) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:.2f}" if mean else f"{100 * v:.1f}%"


def _num(v) -> str:
    return "–" if v is None or math.isnan(v) else f"{v:.1f}"


def _delta(v, lo, hi, mean: bool) -> str:
    if v is None or math.isnan(v):
        return "–"
    if mean:
        return f"{v:+.2f} <span class=ci>[{lo:+.2f}, {hi:+.2f}]</span>"
    return f"{100 * v:+.1f}pt <span class=ci>[{100 * lo:+.1f}, {100 * hi:+.1f}]</span>"


def _z(z: float) -> str:
    return "∞" if math.isinf(z) else f"{z:.1f}"


def _bars(bc: float, subj: float) -> str:
    w = lambda v: 0 if math.isnan(v) else max(0.0, min(100.0, 100 * v))
    return (f'<div class=bars><span class="bar bc" style="width:{w(bc):.1f}%"></span>'
            f'<span class="bar subj" style="width:{w(subj):.1f}%"></span></div>')


def _e(v) -> str:
    return html.escape(str(v))


_CSS = """
:root{--bg:#fbfaf8;--fg:#1d1c1a;--muted:#6b6862;--line:#e4e1db;--bc:#9c978d;--subj:#c2643c;--head:#f2f0ec;--link:#9a4a26}
@media (prefers-color-scheme:dark){:root{--bg:#1c1b19;--fg:#ecebe7;--muted:#a19d95;--line:#35332f;--bc:#77736b;--subj:#e08a62;--head:#252420;--link:#eba27f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1320px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:36px 0 8px;padding-bottom:4px;border-bottom:1px solid var(--line)}h3{font-size:15px;margin:22px 0 2px}
dl{display:grid;grid-template-columns:max-content 1fr;gap:2px 12px;margin:12px 0}dt{color:var(--muted)}dd{margin:0}
.caveats{color:var(--muted);font-size:13px}.muted{color:var(--muted);margin:2px 0 6px}
.metric{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{background:var(--head);font-weight:600;font-size:12px}td.num,th.num{text-align:right}
a{color:var(--link)}.ci{color:var(--muted);font-size:12px}
.star{color:var(--subj)}td.ref{color:var(--muted)}.bars{width:90px}.bar{display:block;height:5px;margin:2px 0;border-radius:2px}
.bar.bc{background:var(--bc)}.bar.subj{background:var(--subj)}
"""

_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{css}</style></head>
<body><main><h1>{title}</h1><dl>{provenance}</dl>
<p class="caveats">Fixed-Opponent design: the Subject team and the BC team each face the same BC opponents on the same
deals; only Subject seats are measured. Rates are ratios of sums; CIs are deal-cluster bootstrap 95%; Δ is paired.
★ = the Δ survives Benjamini–Hochberg at 5% across the whole panel. Descriptive, not causal. Synthetic Games are exact
only for score-blind Checkpoints.</p>{reference_note}
{summary}{sections}</main></body></html>"""

_REFERENCE_NOTE = """<p class="caveats">Human columns are an unpaired reference from replayed BSW games (grey; 95% CI on
hover, bootstrapped by Game): different deals, human opponents, no Δ and no BH. Same Decision definitions as the agents,
except that out-of-turn Bomb interrupts are not logged, Trick share is unavailable, and humans see the game score.
Rounds per Game is the real Game length.</p>"""
