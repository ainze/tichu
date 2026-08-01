r"""Pre-check: does putting the LEGAL-MOVE SET into the input Feature Vector
improve the play policy — signal the v6 net demonstrably cannot already see?

Context. The legal mask is already computed on every Decision and applied as an
*output* mask (`bc.loss.masked_cross_entropy`, `ppo.policy.sample_masked`). It
never enters the trunk: `BCModel.forward` concatenates only `features +
skill_emb`. This probe asks whether feeding it (or a compact summary of it) to
the trunk buys anything.

Prior to beat: ADR-0017 found a 1809-dim one-hot over *this same* Action Space,
used as an input section, was ~89% dead weight and was replaced by 50 compact
dims with no measurable loss.

Method (mirrors the ADR-0039 pre-check, `scripts/precheck_trick_stakes.py`):
  * Slice: play Decisions by a chosen skill decile (ADR-0024 master tier = 9).
  * Objective: the REAL BC objective — masked cross-entropy over the 1809-wide
    play head against the human's Intent, with the true legal mask applied at
    the output in EVERY arm. So the delta measures purely the representational
    shortcut, not the mask's decision-relevant content (which every arm has).
  * Arms:
      A  baseline_v6        591 dims
      B  v6+raw_mask        591 + 1809
      B' v6+shuffled_mask   591 + 1809, mask rows permuted across samples
                            (placebo: isolates capacity from information)
      C  v6+summary         591 + 30   (mobility / per-kind beat-range / bombs)
      D  v6+n_legal         591 + 3    (n_legal, log n_legal, is_forced)
  * Split: by GAME (never by row — round-unique leakage inflates deltas, the
    ADR-0033 lesson). Three-way train/val/test; val drives early stopping.
  * Report: held-out masked NLL + top-1, seed-averaged, reported BOTH overall
    and on the NON-FORCED subset (|legal| > 1). Forced rows are ~39% of play
    Decisions and are free for every arm — they dilute any real effect.

Writes nothing but the JSON report (and an optional collection cache).

Usage (PowerShell; data lives in the main checkout, script in the worktree):
  $env:PYTHONPATH = "C:\workbench\tichu\.claude\worktrees\sweet-meitner-455316\src"
  python scripts/precheck_legal_moves.py `
      --archive C:\workbench\tichu\data\archive.zst `
      --ratings C:\workbench\tichu\data\ratings_full.parquet `
      --decile 9 --limit-games 12000 --max-rows 300000 `
      --save-cache C:\workbench\tichu\data\precheck_legal_cache.npz `
      --out precheck_legal.json
  # plumbing smoke (no corpus needed):
  python scripts/precheck_legal_moves.py --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("precheck_legal_moves")

MASK_DIM = 1809  # HEAD_LOGIT_DIMS["play"]; asserted against action_space at run.


@dataclass
class Collected:
    X: np.ndarray      # (N, 591) v6 features, float32
    M: np.ndarray      # (N, 227) bit-packed legal mask, uint8
    y: np.ndarray      # (N,)     play-head target index, int32
    games: np.ndarray  # (N,)     game index, int64 (for by-game split)
    v: np.ndarray      # (N,)     seat-relative round outcome (critic target), f32


def save_cache(path: str, d: Collected) -> None:
    np.savez_compressed(path, X=d.X, M=d.M, y=d.y, games=d.games, v=d.v)


def load_cache(path: str) -> Collected:
    z = np.load(path)
    return Collected(X=z["X"], M=z["M"], y=z["y"], games=z["games"], v=z["v"])


# --------------------------------------------------------------------------- #
# Action-space taxonomy: index -> (kind, primary_rank, is_bomb).
# Built once; drives the compact summary block.
# --------------------------------------------------------------------------- #
def build_action_taxonomy():
    """Return (kind, prank, n_kinds) arrays over the 1809 action slots.

    kind: 0..7 play combination kinds in canonical order, 8 = Pass, 9 = other
          (calls / schupfen / wish / dragon slots that live in the same space
          but are never legal for a play Decision).
    prank: primary rank 1..14 (0 when not applicable).
    """
    from tichu_training.action_space import (
        ACTION_SPACE_SIZE, CANONICAL_ACTIONS, Pass, PlayFourBomb, PlayFullHouse,
        PlayPair, PlayPairStep, PlaySingle, PlayStraight, PlayStraightFlushBomb,
        PlayTriple,
    )

    assert ACTION_SPACE_SIZE == MASK_DIM, (ACTION_SPACE_SIZE, MASK_DIM)
    kind = np.full(ACTION_SPACE_SIZE, 9, dtype=np.int8)
    prank = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
    _SPECIAL_RANK = {"mahjong": 1, "dog": 0, "dragon": 14, "phoenix": 1}
    for i, act in enumerate(CANONICAL_ACTIONS):
        if isinstance(act, PlaySingle):
            kind[i] = 0
            if act.rank is not None:
                prank[i] = act.rank
            elif act.phoenix_as_rank is not None:
                prank[i] = act.phoenix_as_rank
            elif act.special is not None:
                prank[i] = _SPECIAL_RANK.get(act.special, 0)
        elif isinstance(act, PlayPair):
            kind[i], prank[i] = 1, act.rank
        elif isinstance(act, PlayTriple):
            kind[i], prank[i] = 2, act.rank
        elif isinstance(act, PlayFullHouse):
            kind[i], prank[i] = 3, act.triple_rank
        elif isinstance(act, PlayPairStep):
            kind[i], prank[i] = 4, act.start_rank
        elif isinstance(act, PlayStraight):
            kind[i], prank[i] = 5, act.start_rank
        elif isinstance(act, PlayFourBomb):
            kind[i], prank[i] = 6, act.rank
        elif isinstance(act, PlayStraightFlushBomb):
            kind[i], prank[i] = 7, act.start_rank
        elif isinstance(act, Pass):
            kind[i] = 8
    return kind, prank


N_KINDS = 8  # play combination kinds
SUMMARY_DIM = 30


def summary_block(mask: np.ndarray, kind: np.ndarray, prank: np.ndarray) -> np.ndarray:
    """Compact legality summary, (B, 30) float32, from a (B, 1809) bool mask.

    Layout:
      0        n_legal / 20, clipped to 1
      1        log1p(n_legal) / 4
      2        is_forced (n_legal == 1)
      3        can_pass
      4..11    per-kind legal count / 8, clipped to 1   (8 dims)
      12..19   per-kind max legal primary rank / 14     (8 dims)
      20..27   per-kind min legal primary rank / 14     (8 dims)
      28       any four-of-a-kind bomb legal
      29       any straight-flush bomb legal
    """
    m = mask.astype(np.float32)
    B = m.shape[0]
    out = np.zeros((B, SUMMARY_DIM), dtype=np.float32)
    n_legal = m.sum(axis=1)
    out[:, 0] = np.clip(n_legal / 20.0, 0.0, 1.0)
    out[:, 1] = np.log1p(n_legal) / 4.0
    out[:, 2] = (n_legal == 1).astype(np.float32)
    pass_idx = int(np.flatnonzero(kind == 8)[0])
    out[:, 3] = m[:, pass_idx]
    pr = prank.astype(np.float32)
    for k in range(N_KINDS):
        cols = np.flatnonzero(kind == k)
        sub = m[:, cols]                       # (B, |cols|)
        cnt = sub.sum(axis=1)
        out[:, 4 + k] = np.clip(cnt / 8.0, 0.0, 1.0)
        ranks = sub * pr[cols][None, :]
        out[:, 12 + k] = ranks.max(axis=1) / 14.0
        # min over LEGAL entries only: push illegal to +inf-equivalent (99).
        lo = np.where(sub > 0, pr[cols][None, :], 99.0).min(axis=1)
        out[:, 20 + k] = np.where(lo < 99.0, lo / 14.0, 0.0)
    out[:, 28] = (out[:, 4 + 6] > 0).astype(np.float32)
    out[:, 29] = (out[:, 4 + 7] > 0).astype(np.float32)
    return out


def nlegal_block(mask: np.ndarray) -> np.ndarray:
    """(B, 3): n_legal/20 clipped, log1p(n_legal)/4, is_forced."""
    n = mask.sum(axis=1).astype(np.float32)
    return np.stack([
        np.clip(n / 20.0, 0.0, 1.0), np.log1p(n) / 4.0, (n == 1).astype(np.float32),
    ], axis=1)


# --------------------------------------------------------------------------- #
# Collection — streams the archive, replays, featurises on the fly. No manifest
# build (that would defeat the "cheap pre-check" promise); the small residue of
# replay-but-mismatch rounds is irrelevant for a relative delta measured on
# identical rows across arms.
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
    archive: str, ratings: str | None, decile: int,
    limit_games: int | None, max_rows: int | None,
) -> Collected:
    from tichu_training.action_space import bc_target_for_concrete, legal_mask
    from tichu_training.bsw.archive import iter_archive
    from tichu_training.bsw.parser import parse_tch
    from tichu_training.bsw.replay import replay_round
    from tichu_training.featurizer import featurize

    skill_lookup = _load_skill_lookup(ratings)
    neutral = -1  # sentinel: never equals a real decile, so unknown handles drop

    Xs: list[np.ndarray] = []
    Ms: list[np.ndarray] = []
    ys: list[int] = []
    gs: list[int] = []
    vs: list[float] = []
    n_games = 0
    for stem, text in iter_archive(archive, game_ids=None):
        n_games += 1
        if limit_games is not None and n_games > limit_games:
            break
        if max_rows is not None and len(ys) >= max_rows:
            break
        try:
            game = parse_tch(text, game_id=stem)
        except Exception:  # noqa: BLE001
            continue
        for parsed_round in game.rounds:
            replay = replay_round(parsed_round)
            if replay.final_state is None:
                continue
            # Team-0-minus-team-1 Ergebnis, made seat-relative per row below —
            # the same quantity `awr.targets` calls the "round" value target.
            team_delta = float(parsed_round.ergebnis[0] - parsed_round.ergebnis[1])
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
                if decile >= 0:
                    handle = parsed_round.handles[player]
                    if skill_lookup.get(handle, neutral) != decile:
                        continue
                # Only the current player's own Decision: bomb-interrupt rows
                # live in a different legality universe and are a rounding error.
                if pre_state.public.current_player != player:
                    continue
                try:
                    target = bc_target_for_concrete("play", concrete)
                except (KeyError, ValueError):
                    continue
                mask = legal_mask("play", pre_state, player, cached_actions=cached)
                if not mask[target]:
                    continue  # defensive: target must be inside the legal set
                private = pre_state.private_view(player)
                Xs.append(featurize(private))
                Ms.append(np.packbits(mask))
                ys.append(int(target))
                gs.append(n_games)
                vs.append(team_delta if player % 2 == 0 else -team_delta)
        if n_games % 500 == 0:
            log.info("scanned %d games, %d rows so far", n_games, len(ys))

    if not ys:
        raise SystemExit(
            "no rows collected — check --decile / --ratings / --archive"
        )
    return Collected(
        X=np.asarray(Xs, dtype=np.float32),
        M=np.asarray(Ms, dtype=np.uint8),
        y=np.asarray(ys, dtype=np.int32),
        games=np.asarray(gs, dtype=np.int64),
        v=np.asarray(vs, dtype=np.float32),
    )


def by_game_split3(games: np.ndarray, val_frac: float, test_frac: float, seed: int):
    """Three-way by-GAME split. Val drives early stopping so a wider arm cannot
    win by memorisation — otherwise the comparison conflates representational
    power with overfitting (the ADR-0039 lesson)."""
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
# Predictor — one MLP per arm onto the real 1809-wide play head, with the true
# legal mask applied at the output in every arm.
# --------------------------------------------------------------------------- #
def fit_eval_one(
    Xtr, Mtr, ytr, Xva, Mva, yva, Xte, Mte, yte,
    hidden: int, epochs: int, seed: int, patience: int = 3, bs: int = 2048,
) -> dict:
    import torch
    from torch.nn.functional import log_softmax

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, MASK_DIM),
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

    def _nll(X, M, y, train: bool):
        """Masked NLL over the arm's rows. Batched host->device transfer so a
        2400-dim arm at 300k rows still fits in 12 GB."""
        total, count = 0.0, 0
        top1 = 0
        n = X.shape[0]
        order = np.random.default_rng(seed).permutation(n) if train else np.arange(n)
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            xb = torch.as_tensor(X[idx], device=dev)
            mb = torch.as_tensor(M[idx], device=dev)
            yb = torch.as_tensor(y[idx].astype(np.int64), device=dev)
            logits = net(xb).masked_fill(~mb, -1e9)
            lp = log_softmax(logits, dim=-1)
            loss = -lp.gather(1, yb.unsqueeze(1)).squeeze(1).mean()
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                top1 += int((lp.argmax(dim=-1) == yb).sum())
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        return total / count, (top1 / count if not train else 0.0)

    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        _nll(Xtr, Mtr, ytr, train=True)
        net.eval()
        with torch.no_grad():
            val_nll, _ = _nll(Xva, Mva, yva, train=False)
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
        nll, acc = _nll(Xte, Mte, yte, train=False)
        # Non-forced subset: |legal| > 1. Forced rows are free for every arm.
        nf = Mte.sum(axis=1) > 1
        if nf.any():
            nll_nf, acc_nf = _nll(Xte[nf], Mte[nf], yte[nf], train=False)
        else:
            nll_nf, acc_nf = float("nan"), float("nan")
    return {
        "nll": nll, "top1": acc,
        "nll_nonforced": nll_nf, "top1_nonforced": acc_nf,
        "val_nll": best_val,
    }


def run_arms(data: Collected, *, hidden, epochs, n_seeds, test_frac, seed0,
             arms_filter=None) -> dict:
    kind, prank = build_action_taxonomy()
    mask = np.unpackbits(data.M, axis=1)[:, :MASK_DIM].astype(bool)
    log.info("unpacked mask: %s, mean |legal| = %.2f, forced-frac = %.3f",
             mask.shape, mask.sum(axis=1).mean(), float((mask.sum(axis=1) == 1).mean()))

    summ = summary_block(mask, kind, prank)
    nleg = nlegal_block(mask)
    maskf = mask.astype(np.float32)
    rng = np.random.default_rng(seed0 + 9999)
    shuf = maskf[rng.permutation(len(maskf))]

    # Arms are built lazily and freed after use: a 2400-dim arm at 300k rows is
    # 2.9 GB, and holding all five at once would not fit alongside the corpus.
    arms = {
        "A_baseline_v6": lambda: data.X,
        "B_raw_mask": lambda: np.concatenate([data.X, maskf], axis=1),
        "Bp_shuffled_mask": lambda: np.concatenate([data.X, shuf], axis=1),
        "C_summary30": lambda: np.concatenate([data.X, summ], axis=1),
        "D_nlegal3": lambda: np.concatenate([data.X, nleg], axis=1),
    }
    if arms_filter:
        arms = {k: v for k, v in arms.items() if k in arms_filter}

    results: dict[str, dict] = {}
    for name, make_feats in arms.items():
        feats = make_feats()
        per_seed = []
        for s in range(n_seeds):
            tr, va, te = by_game_split3(data.games, 0.1, test_frac, seed0 + s)
            per_seed.append(fit_eval_one(
                feats[tr], mask[tr], data.y[tr],
                feats[va], mask[va], data.y[va],
                feats[te], mask[te], data.y[te],
                hidden, epochs, seed0 + s,
            ))
            log.info("  %s seed %d: %s", name, s, per_seed[-1])
        results[name] = {
            k: {"mean": float(np.mean([r[k] for r in per_seed])),
                "std": float(np.std([r[k] for r in per_seed]))}
            for k in ("nll", "top1", "nll_nonforced", "top1_nonforced")
        }
        results[name]["in_dim"] = int(feats.shape[1])
        del feats
    base = results.get("A_baseline_v6")
    if base:
        for name, d in results.items():
            if name == "A_baseline_v6":
                continue
            for key in ("nll", "nll_nonforced"):
                b = base[key]["mean"]
                d[f"{key}_delta"] = d[key]["mean"] - b
                d[f"{key}_rel"] = (b - d[key]["mean"]) / b if b else 0.0
    return results


# --------------------------------------------------------------------------- #
# H3 — the CRITIC arm. `ValueBaseline.forward(features)` takes the same 591-dim
# vector and, unlike the play head, has NO output mask at all. If the legal set
# carries anything the trunk cannot already compute, V(s) is where it lands —
# and the critic residual is the standing bottleneck (advantage-SNR probe).
# --------------------------------------------------------------------------- #
def fit_value_one(Xtr, vtr, Xva, vva, Xte, vte, hidden, epochs, seed,
                  patience: int = 3, bs: int = 4096) -> dict:
    import torch

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, 1),
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    # Standardise the target so MSE is on a comparable scale across arms.
    mu, sd = float(vtr.mean()), float(vtr.std()) or 1.0

    def _pass(X, v, train):
        tot, cnt = 0.0, 0
        n = X.shape[0]
        order = np.random.default_rng(seed).permutation(n) if train else np.arange(n)
        preds = []
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            xb = torch.as_tensor(X[idx], device=dev)
            vb = torch.as_tensor((v[idx] - mu) / sd, device=dev)
            out = net(xb).squeeze(1)
            loss = torch.nn.functional.mse_loss(out, vb)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                preds.append(out.detach().cpu().numpy())
            tot += float(loss.detach()) * len(idx)
            cnt += len(idx)
        return tot / cnt, (np.concatenate(preds) if preds else None)

    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        _pass(Xtr, vtr, True)
        net.eval()
        with torch.no_grad():
            val_mse, _ = _pass(Xva, vva, False)
        if val_mse < best - 1e-5:
            best, bad = val_mse, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        _, pred = _pass(Xte, vte, False)
    yz = (vte - mu) / sd
    ss_res = float(((yz - pred) ** 2).sum())
    ss_tot = float(((yz - yz.mean()) ** 2).sum())
    return {"r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
            "mse": ss_res / len(yz), "val_mse": best}


def run_value_arms(data: Collected, *, hidden, epochs, n_seeds, test_frac, seed0):
    kind, prank = build_action_taxonomy()
    mask = np.unpackbits(data.M, axis=1)[:, :MASK_DIM].astype(bool)
    rng = np.random.default_rng(seed0 + 9999)
    shuf = mask.astype(np.float32)[rng.permutation(len(mask))]
    arms = {
        "A_baseline_v6": lambda: data.X,
        "B_raw_mask": lambda: np.concatenate([data.X, mask.astype(np.float32)], axis=1),
        # The capacity placebo matters here too: on the policy side it came out
        # NEGATIVE, so a value lift must clear it, not just clear baseline.
        "Bp_shuffled_mask": lambda: np.concatenate([data.X, shuf], axis=1),
        "C_summary30": lambda: np.concatenate(
            [data.X, summary_block(mask, kind, prank)], axis=1),
    }
    results = {}
    for name, make in arms.items():
        feats = make()
        per_seed = []
        for s in range(n_seeds):
            tr, va, te = by_game_split3(data.games, 0.1, test_frac, seed0 + s)
            per_seed.append(fit_value_one(
                feats[tr], data.v[tr], feats[va], data.v[va],
                feats[te], data.v[te], hidden, epochs, seed0 + s,
            ))
            log.info("  value %s seed %d: %s", name, s, per_seed[-1])
        results[name] = {
            k: {"mean": float(np.mean([r[k] for r in per_seed])),
                "std": float(np.std([r[k] for r in per_seed]))}
            for k in ("r2", "mse")
        }
        del feats
    base = results["A_baseline_v6"]["r2"]["mean"]
    for name, d in results.items():
        d["r2_delta"] = d["r2"]["mean"] - base
    return results


# --------------------------------------------------------------------------- #
# Redundancy check — is the mask a deterministic function of the v6 features?
# If identical Feature Vectors always carry identical masks, the mask adds ZERO
# information and can only ever be a computational shortcut.
# --------------------------------------------------------------------------- #
def redundancy_check(data: Collected) -> dict:
    mask = np.unpackbits(data.M, axis=1)[:, :MASK_DIM]
    xb = np.ascontiguousarray(data.X).view(np.uint8).reshape(len(data.X), -1)
    xk = [h.tobytes() for h in xb]
    mk = [h.tobytes() for h in np.ascontiguousarray(mask)]
    seen: dict[bytes, bytes] = {}
    dup = conflict = 0
    for a, b in zip(xk, mk):
        if a in seen:
            dup += 1
            if seen[a] != b:
                conflict += 1
        else:
            seen[a] = b
    return {
        "n_rows": len(xk), "n_distinct_features": len(seen),
        "n_duplicate_feature_vectors": dup,
        "n_mask_conflicts": conflict,
        "mask_is_function_of_features": conflict == 0,
    }


def _synthetic(n=12000, d=591, seed=0) -> Collected:
    """POSITIVE CONTROL: proves the harness can detect a real mask-input effect.

    The control must be *context-dependent*, because output masking is applied
    in every arm: any label rule of the form "argmax over legals of a per-action
    score" is already learnable WITHOUT the mask as input — the net emits a
    score per action and the output mask restricts it. The mask can only add
    information when an action's value depends on WHAT ELSE is legal.

    So: label = the LOWEST legal index when |legal| is even, the HIGHEST when
    odd. Parity of |legal| is invisible to a per-action scorer, so the baseline
    caps near 50% top-1 while any arm that sees the mask (esp. D_nlegal3, which
    gets n_legal directly) should approach 100%.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    mask = np.zeros((n, MASK_DIM), dtype=bool)
    ys = np.zeros(n, dtype=np.int32)
    for i in range(n):
        k = int(rng.integers(2, 9))
        # Draw from a narrow index band so the "index ramp" the output layer
        # must learn is a handful of dims, not 1809 — the control tests whether
        # the harness sees a mask effect, not how fast a 1809-wide ramp fits.
        cols = np.sort(rng.choice(40, size=k, replace=False))
        mask[i, cols] = True
        ys[i] = cols[0] if k % 2 == 0 else cols[-1]
    return Collected(
        X=X, M=np.packbits(mask, axis=1), y=ys,
        games=rng.integers(0, n // 8, n).astype(np.int64),
        v=rng.standard_normal(n).astype(np.float32),
    )


# --------------------------------------------------------------------------- #
# Scale sweep — the single biggest threat to this probe's external validity.
# A hand-computed shortcut feature helps MOST when the model is under-
# parameterised and data-starved; production is a 1024x4 residual trunk on
# 1.25B rows. If the raw-mask lift DECAYS as width and rows grow, the trunk is
# already learning legality internally and the feature buys nothing at scale.
# If it stays FLAT, it survives. This measures the trend, not a point estimate.
# --------------------------------------------------------------------------- #
def run_sweep(data: Collected, *, widths, fracs, epochs, n_seeds, test_frac, seed0):
    mask = np.unpackbits(data.M, axis=1)[:, :MASK_DIM].astype(bool)
    full = {"A": data.X, "B": np.concatenate([data.X, mask.astype(np.float32)], axis=1)}
    rows = []
    for frac in fracs:
        rng = np.random.default_rng(seed0 + 4242)
        uniq = np.unique(data.games)
        keep = set(rng.permutation(uniq)[:max(2, int(len(uniq) * frac))].tolist())
        sel = np.array([g in keep for g in data.games])
        for hidden in widths:
            per_arm = {}
            for arm in ("A", "B"):
                accs = []
                for s in range(n_seeds):
                    tr, va, te = by_game_split3(data.games[sel], 0.1, test_frac, seed0 + s)
                    f = full[arm][sel]
                    accs.append(fit_eval_one(
                        f[tr], mask[sel][tr], data.y[sel][tr],
                        f[va], mask[sel][va], data.y[sel][va],
                        f[te], mask[sel][te], data.y[sel][te],
                        hidden, epochs, seed0 + s,
                    ))
                per_arm[arm] = {
                    k: float(np.mean([a[k] for a in accs]))
                    for k in ("nll_nonforced", "top1_nonforced")
                }
            b = per_arm["A"]["nll_nonforced"]
            rel = (b - per_arm["B"]["nll_nonforced"]) / b
            row = {
                "n_rows": int(sel.sum()), "frac": frac, "hidden": hidden,
                "A_nll_nf": per_arm["A"]["nll_nonforced"],
                "B_nll_nf": per_arm["B"]["nll_nonforced"],
                "A_top1_nf": per_arm["A"]["top1_nonforced"],
                "B_top1_nf": per_arm["B"]["top1_nonforced"],
                "rel_nll_nf": rel,
                "top1_gain_pts": 100 * (per_arm["B"]["top1_nonforced"]
                                        - per_arm["A"]["top1_nonforced"]),
            }
            rows.append(row)
            log.info("SWEEP %s", row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive")
    ap.add_argument("--ratings")
    ap.add_argument("--decile", type=int, default=9, help="-1 = all deciles")
    ap.add_argument("--limit-games", type=int)
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", help="comma-separated arm subset")
    ap.add_argument("--bar-rel-nll", type=float, default=0.01,
                    help="pre-registered bar: min relative held-out NON-FORCED "
                         "NLL reduction of the best mask arm to greenlight a bump")
    ap.add_argument("--out", default="precheck_legal.json")
    ap.add_argument("--save-cache")
    ap.add_argument("--load-cache")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--skip-value", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="run the capacity x data scale sweep instead of the arms")
    ap.add_argument("--sweep-widths", default="128,512,1024")
    ap.add_argument("--sweep-fracs", default="0.125,0.25,0.5,1.0")
    ap.add_argument("--smoke", action="store_true")
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
        data = collect_rows(
            args.archive, args.ratings, args.decile,
            args.limit_games, args.max_rows,
        )
    if args.save_cache and not args.load_cache:
        save_cache(args.save_cache, data)
        log.info("saved cache -> %s", args.save_cache)
    if args.collect_only:
        log.info("collect-only: N=%d rows, done", len(data.y))
        return

    if args.sweep:
        rows = run_sweep(
            data,
            widths=[int(w) for w in args.sweep_widths.split(",")],
            fracs=[float(f) for f in args.sweep_fracs.split(",")],
            epochs=args.epochs, n_seeds=args.n_seeds,
            test_frac=args.test_frac, seed0=args.seed,
        )
        with open(args.out, "w") as f:
            json.dump({"sweep": rows}, f, indent=2)
        print(f"{'rows':>8} {'hidden':>7} {'A_nll':>8} {'B_nll':>8} "
              f"{'relNLL':>9} {'top1_gain':>10}")
        for r in rows:
            print(f"{r['n_rows']:>8} {r['hidden']:>7} {r['A_nll_nf']:>8.4f} "
                  f"{r['B_nll_nf']:>8.4f} {r['rel_nll_nf']:>+8.3%} "
                  f"{r['top1_gain_pts']:>+9.2f}pt")
        return

    redundancy = redundancy_check(data)
    log.info("redundancy: %s", redundancy)

    results = run_arms(
        data, hidden=args.hidden, epochs=args.epochs, n_seeds=args.n_seeds,
        test_frac=args.test_frac, seed0=args.seed,
        arms_filter=set(args.arms.split(",")) if args.arms else None,
    )
    value_results = run_value_arms(
        data, hidden=args.hidden, epochs=args.epochs, n_seeds=args.n_seeds,
        test_frac=args.test_frac, seed0=args.seed,
    ) if not args.skip_value else {}
    mask_arms = [k for k in ("B_raw_mask", "C_summary30", "D_nlegal3") if k in results]
    best = max((results[k].get("nll_nonforced_rel", 0.0) for k in mask_arms),
               default=0.0)
    placebo = results.get("Bp_shuffled_mask", {}).get("nll_nonforced_rel", 0.0)
    verdict = (
        "GREENLIGHT" if best >= args.bar_rel_nll and best > placebo + 0.002
        else "STOP — below bar / not separable from the capacity placebo"
    )
    report = {
        "n_rows": int(len(data.y)),
        "decile": args.decile,
        "bar_rel_nll": args.bar_rel_nll,
        "redundancy": redundancy,
        "value_arms": value_results,
        "arms": results,
        "best_mask_arm_rel_nll_nonforced": best,
        "placebo_rel_nll_nonforced": placebo,
        "verdict": verdict,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"N={report['n_rows']}  bar={args.bar_rel_nll:.2%}")
    for name, d in results.items():
        print(f"  {name:18s} dim={d['in_dim']:5d} "
              f"NLL={d['nll']['mean']:.4f} top1={d['top1']['mean']:.4f} | "
              f"nonforced NLL={d['nll_nonforced']['mean']:.4f} "
              f"top1={d['top1_nonforced']['mean']:.4f} "
              f"relNLL_nf={d.get('nll_nonforced_rel', 0.0):+.4%}")
    if value_results:
        print("value head (H3 — critic has NO output mask):")
        for name, d in value_results.items():
            print(f"  {name:18s} R2={d['r2']['mean']:.4f} "
                  f"(+/-{d['r2']['std']:.4f})  dR2={d['r2_delta']:+.4f}")
    print(f"redundancy: mask_is_function_of_features="
          f"{redundancy['mask_is_function_of_features']} "
          f"(conflicts {redundancy['n_mask_conflicts']} / "
          f"{redundancy['n_duplicate_feature_vectors']} duplicate feature rows)")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
