r"""ADR-0039 pre-check: residual lift of H1/H2 on the decile-9 contested slice.

Answers the go/no-go question *before* any FEATURIZER_VERSION bump: does adding
the trick point-stakes block {H1 trick_point_value, H2 current_trick_winner} to
the v6 Feature Vector measurably improve prediction of the human's grab-vs-cede
choice on contested tricks — signal the v6 net demonstrably cannot already see?

Method (channel 1, "better BC prior"):
  * Slice: decile-9 *contested* play Decisions (non-empty Trick, non-zero point
    stakes, a legal non-Pass action) — `is_contested_trick_decision`.
  * Label: grab (human played a beat) vs cede (human Passed).
  * Baseline features: the v6 591-dim Feature Vector.
  * Treatment: v6 + `stakes_block` (5 dims). Diagnostic arm: v6 + H1 scalar only.
  * Predictor: a small MLP, **seed-averaged** (init noise can dwarf a small
    effect) with a **by-game** train/test split (round-unique leakage would
    inflate the delta — the ADR-0033 lesson).
  * Report: held-out NLL/AUC per arm + the treatment-vs-baseline NLL delta
    against a pre-registered bar. H1-alone is a diagnostic, never the go/no-go.

This writes nothing to disk beyond the JSON report — no bundle, no version bump.

Inputs are the BSW archive + the ratings parquet only — features are computed
by replay + `featurize` on the fly, so no shard read and no manifest build (a
full-corpus manifest would defeat the "cheap" promise). PYTHONPATH must point at
*this* worktree's src so the local v6 featurizer + `trick_stakes` are picked up.

Usage (PowerShell; data lives in the main checkout, script in the worktree):
  $env:PYTHONPATH = "C:\workbench\tichu\.claude\worktrees\youthful-hypatia-feeadf\src"
  python scripts/precheck_trick_stakes.py `
      --archive C:\workbench\tichu\data\archive.zst `
      --ratings C:\workbench\tichu\data\ratings_full.parquet `
      --decile 9 --limit-games 20000 --out precheck_stakes.json
  # plumbing smoke (no corpus needed):
  python scripts/precheck_trick_stakes.py --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass

import numpy as np

log = logging.getLogger("precheck_trick_stakes")


@dataclass
class Collected:
    X: np.ndarray      # (N, 591) v6 features, float32
    S: np.ndarray      # (N, 5)   stakes block (H1/H2), float32
    R: np.ndarray      # (N, 17)  remaining-by-rank block (H3), float32
    y: np.ndarray      # (N,)     1=grab, 0=cede, int8
    games: np.ndarray  # (N,)     game index, int64 (for by-game split)


def save_cache(path: str, d: Collected) -> None:
    np.savez_compressed(path, X=d.X, S=d.S, R=d.R, y=d.y, games=d.games)


def load_cache(path: str) -> Collected:
    z = np.load(path)
    return Collected(X=z["X"], S=z["S"], R=z["R"], y=z["y"], games=z["games"])


# --------------------------------------------------------------------------- #
# Data collection — mirrors ParquetBCDataset.__iter__, tapping the stakes block
# and the grab/cede label on the contested slice.
# --------------------------------------------------------------------------- #
def _load_skill_lookup(ratings: str | None) -> dict[str, int]:
    if not ratings:
        return {}
    import pyarrow.parquet as pq

    t = pq.read_table(ratings, columns=["player_handle", "skill_decile"])
    handles = t.column("player_handle").to_pylist()
    deciles = t.column("skill_decile").to_pylist()
    return {h: int(d) for h, d in zip(handles, deciles) if h and d is not None}


def collect_rows(
    archive: str, ratings: str | None, decile: int, limit_games: int | None,
) -> Collected:
    """Stream the archive directly (no manifest — the full-corpus manifest build
    would defeat the "cheap pre-check" promise). Features come from replay +
    `featurize`, never the shards, so no shard read and no version pin is needed.
    Rounds that fail to replay are skipped; the small residue of replay-but-
    Ergebnis-mismatch rounds (which the ADR-0008 manifest would exclude) is
    negligible for a relative delta measured on identical rows across arms."""
    from tichu_engine.legality import Pass as EnginePass
    from tichu_training.bsw.archive import iter_archive
    from tichu_training.bsw.parser import parse_tch
    from tichu_training.bsw.replay import replay_round
    from tichu_training.featurizer import featurize
    from tichu_training.trick_stakes import (
        is_contested_trick_decision, stakes_block,
    )
    from tichu_training.card_counting import remaining_by_rank_block

    skill_lookup = _load_skill_lookup(ratings)
    neutral = -1  # sentinel: never equals a real decile, so unknown handles drop

    Xs: list[np.ndarray] = []
    Ss: list[np.ndarray] = []
    Rs: list[np.ndarray] = []
    ys: list[int] = []
    gs: list[int] = []
    n_games = 0
    for stem, text in iter_archive(archive, game_ids=None):
        n_games += 1
        if limit_games is not None and n_games > limit_games:
            break
        try:
            game = parse_tch(text, game_id=stem)
        except Exception:  # noqa: BLE001
            continue
        for parsed_round in game.rounds:
            replay = replay_round(parsed_round)
            if replay.final_state is None:
                continue
            for (parsed_action, concrete), pre_state, cached in zip(
                replay.decisions, replay.pre_decision_states, replay.legal_actions_at,
            ):
                if pre_state is None or cached is None:
                    continue
                if parsed_action.kind not in ("play", "pass"):
                    continue
                player = parsed_action.player
                if not 0 <= player < 4:
                    continue
                handle = parsed_round.handles[player]
                if skill_lookup.get(handle, neutral) != decile:
                    continue
                private = pre_state.private_view(player)
                if not is_contested_trick_decision(private, cached):
                    continue
                Xs.append(featurize(private))
                Ss.append(stakes_block(private))
                Rs.append(remaining_by_rank_block(private))
                ys.append(0 if isinstance(concrete, EnginePass) else 1)
                gs.append(n_games)
        if n_games % 2000 == 0:
            log.info("scanned %d games, %d contested rows so far", n_games, len(ys))

    if not ys:
        raise SystemExit(
            "no decile-%d contested rows collected — check --decile / --ratings / "
            "--archive" % decile
        )
    return Collected(
        X=np.asarray(Xs, dtype=np.float32),
        S=np.asarray(Ss, dtype=np.float32),
        R=np.asarray(Rs, dtype=np.float32),
        y=np.asarray(ys, dtype=np.int8),
        games=np.asarray(gs, dtype=np.int64),
    )


def by_game_split(games: np.ndarray, test_frac: float, seed: int):
    """Hold out whole games (not rows) so no round leaks across the split."""
    uniq = np.unique(games)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * test_frac))
    test_games = set(uniq[:n_test].tolist())
    is_test = np.array([g in test_games for g in games])
    return ~is_test, is_test


def by_game_split3(games: np.ndarray, val_frac: float, test_frac: float, seed: int):
    """Three-way by-game split (train / val / test). Val drives early stopping so
    a wider probe cannot simply overfit — otherwise capacity comparisons conflate
    representational power with train-set memorisation."""
    uniq = np.unique(games)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test_g = set(uniq[:n_test].tolist())
    val_g = set(uniq[n_test:n_test + n_val].tolist())
    is_test = np.array([g in test_g for g in games])
    is_val = np.array([g in val_g for g in games])
    return ~(is_test | is_val), is_val, is_test


# --------------------------------------------------------------------------- #
# Predictor — small MLP, one arm per input matrix. Early-stopped on a held-out
# validation split so the capacity sweep is not confounded by overfitting.
# --------------------------------------------------------------------------- #
def fit_eval_one(Xtr, ytr, Xva, yva, Xte, yte, hidden: int, epochs: int, seed: int,
                 patience: int = 4) -> dict:
    import torch

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = torch.nn.BCEWithLogitsLoss()
    xtr = torch.as_tensor(Xtr, device=dev)
    ytt = torch.as_tensor(ytr.astype(np.float32), device=dev)
    xva = torch.as_tensor(Xva, device=dev)
    yva_t = torch.as_tensor(yva.astype(np.float32), device=dev)
    xte = torch.as_tensor(Xte, device=dev)
    n = xtr.shape[0]
    bs = 4096
    best_val = float("inf")
    best_state = None
    bad = 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(xtr[idx]).squeeze(1), ytt[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val_nll = float(lossf(net(xva).squeeze(1), yva_t))
        if val_nll < best_val - 1e-5:
            best_val = val_nll
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(xte).squeeze(1)).clamp(1e-6, 1 - 1e-6).cpu().numpy()
    yte = yte.astype(np.float64)
    nll = float(-(yte * np.log(p) + (1 - yte) * np.log(1 - p)).mean())
    acc = float(((p > 0.5).astype(np.float64) == yte).mean())
    auc = _auc(yte, p)
    return {"nll": nll, "acc": acc, "auc": auc}


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run_arms(data: Collected, *, hidden, epochs, n_seeds, test_frac, seed0) -> dict:
    arms = {
        "baseline_v6": data.X,
        "v6+H1": np.concatenate([data.X, data.S[:, :1]], axis=1),
        "v6+H2": np.concatenate([data.X, data.S[:, 1:]], axis=1),
        "v6+H1H2": np.concatenate([data.X, data.S], axis=1),
        "v6+H3": np.concatenate([data.X, data.R], axis=1),
        "v6+H1H2+H3": np.concatenate([data.X, data.S, data.R], axis=1),
    }
    results: dict[str, dict] = {}
    for name, feats in arms.items():
        per_seed = []
        for s in range(n_seeds):
            tr, va, te = by_game_split3(data.games, 0.1, test_frac, seed0 + s)
            per_seed.append(fit_eval_one(
                feats[tr], data.y[tr], feats[va], data.y[va],
                feats[te], data.y[te], hidden, epochs, seed0 + s,
            ))
        results[name] = {
            k: {"mean": float(np.mean([r[k] for r in per_seed])),
                "std": float(np.std([r[k] for r in per_seed]))}
            for k in ("nll", "acc", "auc")
        }
    base = results["baseline_v6"]["nll"]["mean"]
    for name in ("v6+H1", "v6+H2", "v6+H1H2", "v6+H3", "v6+H1H2+H3"):
        d = results[name]
        d["nll_delta_vs_baseline"] = d["nll"]["mean"] - base
        d["nll_rel_improvement"] = (base - d["nll"]["mean"]) / base if base else 0.0
    return results


def _synthetic(n=8000, d=591, seed=0) -> Collected:
    """Plumbing smoke: H2 slot carries real signal, H1 does not — a correct run
    should show v6+H1H2 improving NLL and v6+H1 barely moving."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    S = np.zeros((n, 5), np.float32)
    S[:, 0] = rng.standard_normal(n)                 # H1 scalar: pure noise here
    winner = rng.integers(0, 4, n)
    S[np.arange(n), 1 + winner] = 1.0                # H2 one-hot: the real signal
    R = (rng.random((n, 17)) < 0.5).astype(np.float32)  # H3: noise (derivable-inert)
    logit = 0.3 * X[:, 0] + 1.5 * (winner == 2)      # partner-winning -> cede less
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
    games = rng.integers(0, n // 8, n).astype(np.int64)
    return Collected(X=X, S=S, R=R, y=y, games=games)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", help="path to the BSW zstd archive (.zst)")
    ap.add_argument("--ratings", help="ratings parquet (player_handle -> skill_decile)")
    ap.add_argument("--decile", type=int, default=9)
    ap.add_argument("--limit-games", type=int)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bar-rel-nll", type=float, default=0.01,
                    help="pre-registered bar: min relative held-out NLL reduction "
                         "of v6+H1H2 vs baseline to greenlight the v7 bump")
    ap.add_argument("--out", default="precheck_stakes.json")
    ap.add_argument("--save-cache", help="save collected arrays (.npz) so future "
                    "arm/bar/seed tweaks skip the ~90-min replay")
    ap.add_argument("--load-cache", help="load collected arrays (.npz) and skip "
                    "collection entirely")
    ap.add_argument("--smoke", action="store_true",
                    help="run on synthetic data to validate the torch plumbing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.smoke:
        data = _synthetic()
        log.info("SMOKE: synthetic %d rows", len(data.y))
    elif args.load_cache:
        data = load_cache(args.load_cache)
        log.info("loaded cache %s: N=%d rows", args.load_cache, len(data.y))
    else:
        if not args.archive:
            ap.error("--archive is required (or use --smoke / --load-cache)")
        if not args.ratings:
            ap.error("--ratings is required to select the decile-%d slice" % args.decile)
        data = collect_rows(
            args.archive, args.ratings, args.decile, args.limit_games,
        )
    if args.save_cache and not args.load_cache:
        save_cache(args.save_cache, data)
        log.info("saved cache -> %s", args.save_cache)
    grab_rate = float(data.y.mean())
    log.info("collected N=%d contested rows, grab-rate=%.3f", len(data.y), grab_rate)

    results = run_arms(
        data, hidden=args.hidden, epochs=args.epochs, n_seeds=args.n_seeds,
        test_frac=args.test_frac, seed0=args.seed,
    )
    rel = results["v6+H1H2"]["nll_rel_improvement"]
    verdict = "GREENLIGHT v7" if rel >= args.bar_rel_nll else "STOP — below bar"
    report = {
        "n_rows": int(len(data.y)),
        "grab_rate": grab_rate,
        "decile": args.decile,
        "bar_rel_nll": args.bar_rel_nll,
        "arms": results,
        "treatment_rel_nll_improvement": rel,
        "verdict": verdict,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    log.info("verdict: %s (rel NLL improvement %.4f vs bar %.4f)",
             verdict, rel, args.bar_rel_nll)
    print(f"N={report['n_rows']}  grab_rate={grab_rate:.3f}  "
          f"bar={args.bar_rel_nll:.3%}")
    for name in ("v6+H1", "v6+H2", "v6+H1H2", "v6+H3", "v6+H1H2+H3"):
        print(f"  {name:12s} relNLL={results[name]['nll_rel_improvement']:+.4%}  "
              f"AUC={results[name]['auc']['mean']:.4f}")
    print(f"verdict (on v6+H1H2): {verdict}")


if __name__ == "__main__":
    main()
