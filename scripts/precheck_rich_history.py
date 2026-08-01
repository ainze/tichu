r"""ADR-0041 follow-up pre-check: does the recovered play-history block improve
the **policy's** move prediction — i.e. does it belong in `featurize()` rather
than in a standalone Belief Model?

The belief route was measured first and came back thin: on 2,000 champion
self-play Rounds the rich block moved hand-prediction from +0.024 to +0.027
top-1 and holder NLL from ~0 to -0.013 vs the card-counting floor. But belief is
an information *bottleneck* — 824 input dims compressed to 168 occupancy
probabilities — and ADR-0028 already warned that feeding its output back "can
only re-package information the policy already had". The policy can consume the
same channels directly and train them against the objective that matters.

Method mirrors `precheck_trick_stakes.py` (ADR-0039), which is the protocol that
turned that hypothesis's apparent +1.16% into a null:
  * Slice: decile-9 **non-forced** Play Decisions (|legal| > 1). Forced rows carry
    zero gradient and inflate top-1 (~39% of play rows).
  * Label: the human's actual play, as an Action-Space index — masked multiclass,
    not a binary proxy. This is move prediction, the featurizer's own bar.
  * Split: **by game**, three-way, val drives early stopping. Round-unique
    leakage is the ADR-0033 lesson and it inflated every un-stopped belief run.
  * **Capacity placebo**: the same 233 extra dims, row-shuffled. Same width, same
    marginals, no row-wise information. Without it a width effect reads as signal.
  * Attribution arms: wish-void alone (the only *certainty* in the block) and
    decline-context alone (the channel v6 most obviously drops).

Pre-registered bar (from the legal-mask input pre-check, which greenlit at
+3.04% rel NLL with a -2.24% placebo): treatment must clear **+1% relative NLL**
on non-forced rows AND beat the placebo by a clear margin. Anything less is the
v7 outcome — sub-1% once early-stopped — and does not justify a version bump.

  $env:PYTHONPATH = "C:\workbench\tichu\.claude\worktrees\tichu-perfect-info-policy-b30cbd\src"
  python scripts/precheck_rich_history.py `
      --archive C:\workbench\tichu\data\archive.zst `
      --ratings C:\workbench\tichu\data\ratings_full.parquet `
      --decile 9 --limit-games 20000 `
      --cache C:\workbench\tichu\data\precheck_rich_cache.npz `
      --out precheck_rich_history.json
  python scripts/precheck_rich_history.py --smoke   # plumbing, no corpus
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("precheck_rich_history")


@dataclass
class Collected:
    X: np.ndarray       # (N, 591)  v6 Feature Vector
    H: np.ndarray       # (N, 233)  RichHistory block
    mask: np.ndarray    # (N, 1809) legal-action mask (bool)
    y: np.ndarray       # (N,)      Action-Space index the human played
    games: np.ndarray   # (N,)      game index, for the by-game split


def save_cache(path: str, d: Collected) -> None:
    np.savez_compressed(
        path, X=d.X, H=d.H, mask=np.packbits(d.mask, axis=1), y=d.y, games=d.games,
        n_actions=np.array([d.mask.shape[1]], dtype=np.int64),
    )


def load_cache(path: str) -> Collected:
    with np.load(path) as z:
        n_actions = int(z["n_actions"][0])
        return Collected(
            X=z["X"], H=z["H"],
            mask=np.unpackbits(z["mask"], axis=1, count=n_actions).astype(bool),
            y=z["y"], games=z["games"],
        )


def collect_rows(
    archive: str, ratings: str | None, decile: int, limit_games: int | None,
) -> Collected:
    """Stream the archive, replaying each Round and accumulating `RichHistory`
    alongside — the block must reflect Decisions BEFORE the one being featurised,
    exactly as the self-play emit does."""
    from tichu_training.action_space import legal_mask, play_intent_index
    from tichu_training.belief.rich_history import RichHistory
    from tichu_training.bsw.archive import iter_archive
    from tichu_training.bsw.parser import parse_tch
    from tichu_training.bsw.replay import replay_round
    from tichu_training.featurizer import featurize

    skill_lookup = _load_skill_lookup(ratings)
    neutral = -1

    Xs, Hs, Ms, ys, gs = [], [], [], [], []
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
            history = RichHistory()
            for (parsed_action, concrete), pre_state, cached in zip(
                replay.decisions, replay.pre_decision_states, replay.legal_actions_at,
            ):
                if pre_state is None:
                    continue
                if parsed_action.kind not in ("play", "pass"):
                    continue
                player = parsed_action.player
                if not 0 <= player < 4:
                    continue
                keep = skill_lookup.get(parsed_round.handles[player], neutral) == decile
                if keep:
                    try:
                        target = play_intent_index(concrete)
                        m = legal_mask("play", pre_state, player, cached_actions=cached)
                    except Exception:  # noqa: BLE001
                        keep = False
                    else:
                        # Non-forced only: a single legal action carries no gradient
                        # and inflates top-1 (~39% of play rows).
                        if int(m.sum()) > 1 and m[target]:
                            Xs.append(featurize(pre_state.private_view(player)))
                            Hs.append(history.block(player))
                            Ms.append(m)
                            ys.append(target)
                            gs.append(n_games)
                history.update(player, concrete, pre_state)
        if n_games % 2000 == 0:
            log.info("scanned %d games, %d non-forced rows", n_games, len(ys))

    if not ys:
        raise SystemExit(f"no decile-{decile} non-forced rows collected")
    return Collected(
        X=np.asarray(Xs, dtype=np.float32),
        H=np.asarray(Hs, dtype=np.float32),
        mask=np.asarray(Ms, dtype=bool),
        y=np.asarray(ys, dtype=np.int64),
        games=np.asarray(gs, dtype=np.int64),
    )


def _load_skill_lookup(ratings: str | None) -> dict[str, int]:
    if not ratings:
        return {}
    import pyarrow.parquet as pq

    t = pq.read_table(ratings, columns=["player_handle", "skill_decile"])
    return {
        h: int(d)
        for h, d in zip(
            t.column("player_handle").to_pylist(),
            t.column("skill_decile").to_pylist(),
        )
        if h and d is not None
    }


def by_game_split3(games: np.ndarray, val_frac: float, test_frac: float, seed: int):
    """Whole games move together, so no Round leaks across the split; val drives
    early stopping so a wider arm cannot win by memorising."""
    uniq = np.unique(games)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_frac))
    n_test = max(1, int(len(uniq) * test_frac))
    val_g = set(uniq[:n_val].tolist())
    test_g = set(uniq[n_val:n_val + n_test].tolist())
    is_val = np.array([g in val_g for g in games])
    is_test = np.array([g in test_g for g in games])
    return ~(is_val | is_test), is_val, is_test


def fit_eval_one(feats, data, tr, va, te, *, hidden, epochs, seed, patience=4):
    """Masked multiclass move prediction. Illegal logits are driven to -inf, so
    the metric is the human's action against the legal set — the same quantity
    the BC play head is scored on."""
    import torch

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_actions = data.mask.shape[1]
    net = torch.nn.Sequential(
        torch.nn.Linear(feats.shape[1], hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, n_actions),
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

    def tensors(idx):
        return (
            torch.as_tensor(feats[idx], device=dev),
            torch.as_tensor(data.mask[idx], device=dev),
            torch.as_tensor(data.y[idx], device=dev),
        )

    xtr, mtr, ytr = tensors(tr)
    xva, mva, yva = tensors(va)
    xte, mte, yte = tensors(te)
    lossf = torch.nn.CrossEntropyLoss()

    def masked_logits(x, m):
        return net(x).masked_fill(~m, float("-inf"))

    best_val, best_state, bad = float("inf"), None, 0
    bs = 4096
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(xtr.shape[0], device=dev)
        for i in range(0, xtr.shape[0], bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            lossf(masked_logits(xtr[idx], mtr[idx]), ytr[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(lossf(masked_logits(xva, mva), yva))
        if val < best_val - 1e-5:
            best_val, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        logits = masked_logits(xte, mte)
        nll = float(lossf(logits, yte))
        top1 = float((logits.argmax(dim=1) == yte).float().mean())
    return {"nll": nll, "top1": top1}


def run_arms(data: Collected, *, hidden, epochs, n_seeds, test_frac, seed0) -> dict:
    from tichu_training.belief.rich_history import PER_OPPONENT

    # Channel offsets inside one opponent's slice, for the attribution arms.
    wish_lo = 6 * 3 + 6 + 2 + 1 + 1
    wish_hi = wish_lo + 13
    ctx = np.concatenate([
        np.arange(o * PER_OPPONENT, o * PER_OPPONENT + 18) for o in range(3)
    ])
    wish = np.concatenate([
        np.arange(o * PER_OPPONENT + wish_lo, o * PER_OPPONENT + wish_hi)
        for o in range(3)
    ])

    rng = np.random.default_rng(seed0)
    shuffled = data.H[rng.permutation(len(data.H))]

    # Built one at a time and freed: at corpus scale each arm is a full
    # (N, 591+k) float32 copy, and holding five at once runs to several GB.
    builders = {
        "baseline_v6": lambda: data.X,
        "v6+rich": lambda: np.concatenate([data.X, data.H], axis=1),
        "v6+rich_PLACEBO": lambda: np.concatenate([data.X, shuffled], axis=1),
        "v6+wish_void": lambda: np.concatenate([data.X, data.H[:, wish]], axis=1),
        "v6+decline_ctx": lambda: np.concatenate([data.X, data.H[:, ctx]], axis=1),
    }
    results: dict[str, dict] = {}
    for name, build in builders.items():
        feats = build()
        per_seed = []
        for s in range(n_seeds):
            tr, va, te = by_game_split3(data.games, 0.1, test_frac, seed0 + s)
            per_seed.append(fit_eval_one(
                feats, data, tr, va, te,
                hidden=hidden, epochs=epochs, seed=seed0 + s,
            ))
        results[name] = {
            k: {"mean": float(np.mean([r[k] for r in per_seed])),
                "std": float(np.std([r[k] for r in per_seed]))}
            for k in ("nll", "top1")
        }
        log.info("%-18s nll=%.4f top1=%.4f", name,
                 results[name]["nll"]["mean"], results[name]["top1"]["mean"])
        del feats

    base_nll = results["baseline_v6"]["nll"]["mean"]
    base_top1 = results["baseline_v6"]["top1"]["mean"]
    for name, d in results.items():
        if name == "baseline_v6":
            continue
        d["nll_rel_improvement"] = (base_nll - d["nll"]["mean"]) / base_nll
        d["top1_delta"] = d["top1"]["mean"] - base_top1
    return results


def _synthetic(n=6000, d=591, h=233, n_actions=40, seed=0) -> Collected:
    """Plumbing smoke: one history column really drives the label, so a correct
    run shows v6+rich beating baseline and the PLACEBO arm not."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    H = rng.standard_normal((n, h)).astype(np.float32)
    mask = rng.random((n, n_actions)) < 0.5
    mask[:, 0] = True  # always at least one legal action
    logit = 2.0 * H[:, 0]
    y = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), 0, 1)
    y = np.where(mask[np.arange(n), y], y, 0).astype(np.int64)
    games = rng.integers(0, n // 8, n).astype(np.int64)
    return Collected(X=X, H=H, mask=mask, y=y, games=games)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive")
    p.add_argument("--ratings")
    p.add_argument("--decile", type=int, default=9)
    p.add_argument("--limit-games", type=int, default=20000)
    p.add_argument("--cache")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--out", default="precheck_rich_history.json")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.smoke:
        data = _synthetic()
    else:
        import os

        if args.cache and os.path.exists(args.cache):
            data = load_cache(args.cache)
            log.info("loaded %d cached rows from %s", len(data.y), args.cache)
        else:
            if not args.archive:
                raise SystemExit("--archive is required (or use --smoke)")
            data = collect_rows(
                args.archive, args.ratings, args.decile, args.limit_games,
            )
            if args.cache:
                save_cache(args.cache, data)
                log.info("cached %d rows -> %s", len(data.y), args.cache)

    log.info("rows=%d  v6=%d  rich=%d  actions=%d",
             len(data.y), data.X.shape[1], data.H.shape[1], data.mask.shape[1])
    results = run_arms(
        data, hidden=args.hidden, epochs=args.epochs, n_seeds=args.n_seeds,
        test_frac=args.test_frac, seed0=args.seed0,
    )
    print(json.dumps(results, indent=2))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    log.info("wrote %s", args.out)

    rich = results["v6+rich"]
    placebo = results["v6+rich_PLACEBO"]
    log.info(
        "\nVERDICT INPUTS: rich rel-NLL %+.4f  placebo rel-NLL %+.4f  "
        "(bar: rich >= +0.01 AND clearly above placebo)",
        rich["nll_rel_improvement"], placebo["nll_rel_improvement"],
    )


if __name__ == "__main__":
    main()
