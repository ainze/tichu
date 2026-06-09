r"""mine_divergences — find where co-training drifted AWAY from BC+human consensus (2026-06-08).

Move-prediction is usually an aggregate accuracy number. This mines the per-decision diffs
behind it, triangulating three policies on every held-out human PLAY decision:

  master (BC) chose M,  cotrain chose C,  the human played H.

Buckets:
  * RED   = M == H and C != H   -> cotrain LEFT a move both its BC base AND a strong human
            endorse. The high-precision "candidate mistake" set.
  * GREEN = C == H and M != H   -> cotrain learned a human-confirmed improvement (no action).
  * GREY  = M != C, neither == H -> both differ from the human (less informative).
  * AGREE = M == C               -> co-training did not change this decision.

It clusters the RED set by (lead/follow, disagreement kind, caller context) and prints a
ranked table, plus a per-decision parquet for drill-down. Each cluster is a HYPOTHESIS, not
a verdict: a divergence means cotrain disagrees with BC+human, which could be cotrain
correctly improving on EV OR a real regression. The forced-action probe (scripts/probe_heuristics.py
pattern) is what adjudicates EV — this just says where to point it.

CAVEAT: the corpus is not skill-filtered here (BSW skill lives in a separate TrueSkill sweep,
not the parsed records), so "human" spans all deciles. The RED bucket's `M == H` leg is the
mitigant: BC is corpus-trained, so a move BC *also* makes is typical play, not a weak-human
fluke. Both agents are conditioned on skill_decile=9 (like check_cotrain). Calls are NOT
covered (decisions_from_game skips Tichu/Grand calls — they are out-of-band should_call).

    py -m scripts.mine_divergences --config configs/cotrain_wish_v5.yaml ^
        --export-dir C:\workbench\tichu\data\runs\cotrain_wish_v5\export\iter_06225 ^
        --held-out C:\workbench\tichu\data\archive.zst --max-games 200
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Torch-free helpers (engine only) — kept importable for unit tests without checkpoints.
from tichu_engine.cards import DRAGON, PHOENIX
from tichu_engine.combinations import (
    FourOfAKindBomb,
    FullHouse,
    Pair,
    PairStep,
    Single,
    Straight,
    StraightFlushBomb,
    Triple,
)
from tichu_engine.engine import _trick_points
from tichu_engine.legality import Pass, _cards_in

_BOMBS = (FourOfAKindBomb, StraightFlushBomb)


def action_kind(a) -> str:
    for cls, name in (
        (Pass, "PASS"), (Single, "SINGLE"), (Pair, "PAIR"), (Triple, "TRIPLE"),
        (FullHouse, "FULLHOUSE"), (Straight, "STRAIGHT"), (PairStep, "PAIRSTEP"),
        (FourOfAKindBomb, "BOMB4"), (StraightFlushBomb, "BOMBSF"),
    ):
        if isinstance(a, cls):
            return name
    return type(a).__name__


def is_premium(a) -> bool:
    """True iff the action spends a Dragon or Phoenix."""
    if isinstance(a, Pass):
        return False
    try:
        return any(c is DRAGON or c is PHOENIX for c in _cards_in(a))
    except Exception:  # noqa: BLE001 — defensive over odd action shapes
        return False


def _rank(a):
    return None if isinstance(a, Pass) else getattr(a, "rank", None)


def disagreement_kind(human, cotrain) -> str:
    """Label HOW cotrain's move differs from the human's, most-salient axis first."""
    hp, cp = isinstance(human, Pass), isinstance(cotrain, Pass)
    if hp and not cp:
        return "cotrain_contests"      # human ceded; cotrain plays over it
    if cp and not hp:
        return "cotrain_cedes"         # human played; cotrain passes
    cb, hb = isinstance(cotrain, _BOMBS), isinstance(human, _BOMBS)
    if cb and not hb:
        return "cotrain_bombs"
    if hb and not cb:
        return "cotrain_unbombs"
    cprem, hprem = is_premium(cotrain), is_premium(human)
    if cprem and not hprem:
        return "cotrain_premium"       # cotrain spends Dragon/Phoenix where human didn't
    if hprem and not cprem:
        return "human_premium"         # human spent premium, cotrain conserved
    hk, ck = action_kind(human), action_kind(cotrain)
    if hk == ck:
        hr, cr = _rank(human), _rank(cotrain)
        if hr is not None and cr is not None and hr != cr:
            return "cotrain_higher" if cr > hr else "cotrain_lower"
        return "same_shape"
    return "diff_shape"


def caller_ctx(pv) -> str:
    pub = pv.public
    callers = pub.tichu_callers | pub.grand_tichu_callers
    me = pv.player
    if me in callers:
        return "self_caller"
    if (me + 2) % 4 in callers:
        return "partner_caller"
    if callers:
        return "opp_caller"
    return "no_caller"


def following(pv) -> bool:
    return pv.public.trick.top_combination is not None


def points_bucket(pv) -> str:
    pts = _trick_points(pv.public.trick)
    return "0" if pts == 0 else ("1-9" if pts < 10 else "10+")


def handsize_bucket(pv) -> str:
    n = len(pv.hand)
    return "1-3" if n <= 3 else ("4-7" if n <= 7 else "8+")


def cluster_key(pv, human, cotrain) -> tuple[str, str, str]:
    return (
        "follow" if following(pv) else "lead",
        disagreement_kind(human, cotrain),
        caller_ctx(pv),
    )


# ------------------------------------------------------------------------------
# Driver (torch + corpus loaded lazily so the helpers above stay test-importable).
# ------------------------------------------------------------------------------

def _severity(cotrain_agent, pv, human, cotrain_action):
    """Under cotrain's own policy: prob(cotrain's move) - prob(human's move). Higher =
    cotrain more confidently disfavoured the human move. None if either is unscored."""
    scores = getattr(cotrain_agent, "play_action_scores", None)
    if scores is None:
        return None
    try:
        probs = dict(scores(pv))
    except Exception:  # noqa: BLE001
        return None
    ph, pc = probs.get(human), probs.get(cotrain_action)
    if ph is None or pc is None:
        return None
    return pc - ph


def _hand_str(pv) -> str:
    return " ".join(sorted(repr(c) for c in pv.hand))


def _load_games_robust(held_out, max_games):
    """Load up to `max_games` parseable games from a .tch dir OR a .zst archive,
    SKIPPING any that fail to parse (the BSW archive has truncated logs). Returns
    (games, n_skipped). Over-samples the archive 2x so skips don't starve the count."""
    from tichu_training.bsw.parser import parse_tch

    path = Path(held_out)
    good: list = []
    skipped = 0
    if path.suffix == ".zst":
        from tichu_training.bsw.archive import iter_archive, list_game_ids

        ids = list_game_ids(path)
        if max_games and len(ids) > max_games:
            step = max(1, len(ids) // (max_games * 2))  # 2x headroom for skips
            ids = ids[::step]
        for gid, text in iter_archive(path, game_ids=set(ids)):
            if max_games and len(good) >= max_games:
                break
            try:
                good.append(parse_tch(text, game_id=gid))
            except Exception:  # noqa: BLE001 — corrupt/truncated game, skip it
                skipped += 1
        return good, skipped
    for p in sorted(path.glob("*.tch")):
        if max_games and len(good) >= max_games:
            break
        try:
            good.append(parse_tch(p.read_text(encoding="utf-8"), game_id=p.stem))
        except Exception:  # noqa: BLE001
            skipped += 1
    return good, skipped


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Mine cotrain-vs-BC-vs-human play divergences")
    p.add_argument("--config", required=True, help="Co-training config YAML (eval.master + skill).")
    p.add_argument("--export-dir", required=True,
                   help="Cotrain snapshot export dir (policy.pt / schupfen.pt / tichu_call.pt / "
                        "grand_tichu_call.pt).")
    p.add_argument("--held-out", default=None,
                   help="Dir of .tch games OR a .zst archive. Default: <repo>/data/archive.zst.")
    p.add_argument("--max-games", type=int, default=200, help="Cap games loaded (default 200).")
    p.add_argument("--max-decisions", type=int, default=None, help="Cap play decisions scored.")
    p.add_argument("--skill-decile", type=int, default=9, help="Skill row both agents condition on.")
    p.add_argument("--top", type=int, default=25, help="RED clusters to print (default 25).")
    p.add_argument("--out", default=None, help="Parquet path for per-decision RED records.")
    args = p.parse_args(argv)

    import yaml

    from tichu_engine.legality import legal_actions_for
    from tichu_eval.move_prediction import decisions_from_game
    from tichu_ml.registry import build_agent
    import tichu_inference.ml_agent  # noqa: F401 — registers the `ml` factory

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    master_kwargs = dict(config["eval"]["master"])
    d = Path(args.export_dir)
    cotrain_kwargs = {
        "checkpoint_path": str(d / "policy.pt"),
        "schupfen_path": str(d / "schupfen.pt"),
        "tichu_call_path": str(d / "tichu_call.pt"),
        "grand_call_path": str(d / "grand_tichu_call.pt"),
    }
    sd = int(args.skill_decile)
    master = build_agent("ml", skill_decile=sd, **master_kwargs)
    cotrain = build_agent("ml", skill_decile=sd, **cotrain_kwargs)

    held_out = args.held_out or str(Path("data") / "archive.zst")
    games, parse_skipped = _load_games_robust(held_out, args.max_games)
    if parse_skipped:
        print(f"  (skipped {parse_skipped} unparseable games)", flush=True)

    counts = {"AGREE": 0, "RED": 0, "GREEN": 0, "GREY": 0}
    clusters: dict[tuple, list] = defaultdict(list)
    red_rows: list[dict] = []
    n_play = 0

    print(f"mine-divergences: master(BC) vs cotrain vs human, {len(games)} games, "
          f"skill_decile {sd}", flush=True)
    print(f"  subject: {d}", flush=True)

    replay_skipped = 0
    for game in games:
        if args.max_decisions is not None and n_play >= args.max_decisions:
            break
        gid = getattr(game, "game_id", None)
        try:
            decs = list(decisions_from_game(game))
        except Exception:  # noqa: BLE001 — one malformed game shouldn't kill the run
            replay_skipped += 1
            continue
        for dec in decs:
            if dec.decision_type != "play":
                continue
            pv = dec.private_state
            if not legal_actions_for(pv):
                continue
            if args.max_decisions is not None and n_play >= args.max_decisions:
                break
            n_play += 1

            human = dec.human_action
            m = master.act(pv)
            c = cotrain.act(pv)
            if m == c:
                counts["AGREE"] += 1
                continue
            if m == human and c != human:
                bucket = "RED"
            elif c == human and m != human:
                counts["GREEN"] += 1
                continue
            else:
                counts["GREY"] += 1
                continue

            counts["RED"] += 1
            key = cluster_key(pv, human, c)
            sev = _severity(cotrain, pv, human, c)
            clusters[key].append(sev)
            red_rows.append({
                "game_id": str(gid) if gid is not None else "",
                "phase": key[0], "disagreement": key[1], "caller_ctx": key[2],
                "points": points_bucket(pv), "handsize": handsize_bucket(pv),
                "human_kind": action_kind(human), "cotrain_kind": action_kind(c),
                "master_kind": action_kind(m),
                "severity": float(sev) if sev is not None else float("nan"),
                "human": repr(human), "cotrain": repr(c), "hand": _hand_str(pv),
            })

    total_changed = counts["RED"] + counts["GREEN"] + counts["GREY"]
    if replay_skipped:
        print(f"  (skipped {replay_skipped} games that failed to replay)", flush=True)
    print(f"\nplay decisions: {n_play}  |  cotrain changed {total_changed} "
          f"({total_changed / max(n_play,1):.1%})", flush=True)
    print(f"  RED  (cotrain left BC+human consensus): {counts['RED']}", flush=True)
    print(f"  GREEN(cotrain -> human, BC did not):    {counts['GREEN']}", flush=True)
    print(f"  GREY (both differ from human):          {counts['GREY']}", flush=True)
    print(f"  AGREE(cotrain == BC):                   {counts['AGREE']}", flush=True)

    def _mean_sev(sevs):
        vals = [s for s in sevs if s is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    ranked = sorted(clusters.items(), key=lambda kv: -len(kv[1]))[: args.top]
    red = max(counts["RED"], 1)
    print(f"\nTop {len(ranked)} RED clusters (candidate-mistake hypotheses), by count:", flush=True)
    print(f"  {'phase':<6} {'disagreement':<18} {'caller_ctx':<14} {'n':>5} {'share':>6} {'mean_sev':>9}",
          flush=True)
    for (phase, dis, ctx), sevs in ranked:
        print(f"  {phase:<6} {dis:<18} {ctx:<14} {len(sevs):>5} {len(sevs)/red:>5.1%} "
              f"{_mean_sev(sevs):>9.3f}", flush=True)

    if args.out and red_rows:
        import pyarrow as pa
        import pyarrow.parquet as pq
        cols = list(red_rows[0].keys())
        table = pa.table({c: pa.array([r[c] for r in red_rows]) for c in cols})
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_path)
        print(f"\nwrote {len(red_rows)} per-decision RED records -> {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
