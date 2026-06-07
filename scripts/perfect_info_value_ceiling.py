r"""Perfect-Info Value-Ceiling Test — the cheap gate for ADR-0033.

Question: does a critic that sees ALL FOUR HANDS at training time materially cut
the policy-gradient advantage variance vs the observable-only critic that made
PPO Refine v1 flat (ADR-0029)? If yes, the asymmetric Perfect-Info Critic (PTIE)
build is justified; if no, ESCAPE is dead and we route to ACCEPT.

Generates master self-play `(observable-224, perfect-info-392, z, round_id,
call_live)` once, fits two `ValueBaseline`s on the SAME round-level split, and
reports the residual-variance ratio

    rho = (1 - R^2_perfect) / (1 - R^2_observable)

three ways — overall / call-live states / the rest (`perfect_info.split_rho`).
Pre-committed gate (ADR-0033):
    rho <= 0.5  -> BUILD the PTIE-PPO loop
    rho >= 0.8  -> KILL ESCAPE, route to ACCEPT
    else        -> run a short asymmetric-critic PPO smoke to break the tie

A guardrail observable-only data-scale sub-sweep confirms R^2_observable has
PLATEAUED w.r.t. data before the delta is trusted (the prior self-play sweep was
starved at ~85k samples).

Usage (PowerShell):
  $env:PYTHONPATH = (Resolve-Path src).Path
  py -3.14 scripts\perfect_info_value_ceiling.py --rounds 4000
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from tichu_eval.full_position_pool import load_full_position_pool
from tichu_eval.play_full import play_full_round
from tichu_inference.ml_agent import MLAgent
from tichu_training.awr.value_baseline import ValueBaseline, fit_value_baseline
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
from tichu_training.perfect_info import (
    PERFECT_INFO_DIM,
    featurize_perfect_info,
    is_call_live,
    r2_score,
    split_rho,
)

EXPORT = r"C:\workbench\tichu\data\export\bc_full_corpus_v5_memmap_unshuffled"
# n=20000 pool (~1.1M self-play samples at ~55/round) so observable R^2 can reach
# the plateau the guardrail checks for; the n=2000 pool caps at ~110k (below it).
POOL = r"C:\workbench\tichu\data\full_position_pool_s0_n20000.parquet"


def _master() -> MLAgent:
    return MLAgent(EXPORT + r"\policy.pt", skill_decile=9,
                   schupfen_path=EXPORT + r"\schupfen.pt",
                   tichu_call_path=EXPORT + r"\tichu_call.pt",
                   grand_call_path=EXPORT + r"\grand_tichu_call.pt")


def collect(master, positions):
    """Master self-play. Per Play Decision record the observable-224 and
    perfect-info-392 features, the call-live flag, and (at round end) the
    team-relative round_outcome/100 as the value target."""
    obs, pi, call_live, rounds, z = [], [], [], [], []
    for rid, pos in enumerate(positions):
        rows = []

        def state_observer(seat, game_state, action, _rows=rows):
            full = featurize_perfect_info(game_state, seat)
            _rows.append((seat, full, is_call_live(game_state.public, seat)))

        result = play_full_round((master, master, master, master), pos.state,
                                 pos.grand_prefixes, state_observer=state_observer)
        total = result.total
        for seat, full, cl in rows:
            t = seat % 2
            obs.append(full[:FEATURIZER_OUTPUT_DIM])
            pi.append(full)
            call_live.append(cl)
            rounds.append(rid)
            z.append((total[t] - total[1 - t]) / 100.0)
    return (np.asarray(obs, dtype=np.float32),
            np.asarray(pi, dtype=np.float32),
            np.asarray(z, dtype=np.float32),
            np.asarray(rounds, dtype=np.int32),
            np.asarray(call_live, dtype=bool))


def _fit(xtr, ytr, *, input_dim, hidden, epochs, lr):
    model = ValueBaseline(input_dim, hidden=hidden)
    fit_value_baseline(model, xtr, ytr, batch_size=4096, epochs=epochs, lr=lr,
                       show_progress=False)
    model.eval()
    return model


def _pred(model, x):
    with torch.no_grad():
        return model(torch.from_numpy(x)).cpu().numpy()


def _verdict(rho: float) -> str:
    if rho <= 0.5:
        return "BUILD - perfect info halves advantage variance; PTIE-PPO is justified."
    if rho >= 0.8:
        return "KILL ESCAPE -> ACCEPT - perfect info barely helps; the symmetric critic was not the confound."
    return "AMBER - run a short asymmetric-critic PPO smoke to break the tie."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=20000,
                    help="self-play rounds (capped at the pool size; master is deterministic)")
    ap.add_argument("--pool", default=POOL, help="starting-position pool parquet")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--cache", default=r"C:\workbench\tichu\data\tmp\perfect_info_ceiling.npz")
    args = ap.parse_args()

    cache = pathlib.Path(args.cache)
    if cache.exists():
        d = np.load(cache)
        obs, pi, z, rounds, call_live = (d["obs"], d["pi"], d["z"],
                                         d["rounds"], d["call_live"])
        print(f"loaded cached data: {len(z):,} samples")
    else:
        print(f"generating master self-play: {args.rounds} rounds...")
        master = _master()
        positions = load_full_position_pool(pathlib.Path(args.pool))[: args.rounds]
        t0 = time.perf_counter()
        obs, pi, z, rounds, call_live = collect(master, positions)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, obs=obs, pi=pi, z=z, rounds=rounds, call_live=call_live)
        print(f"collected {len(z):,} samples in {time.perf_counter()-t0:.0f}s "
              f"(Var(z)={z.var():.3f}, call-live {call_live.mean():.1%})")

    assert obs.shape[1] == FEATURIZER_OUTPUT_DIM and pi.shape[1] == PERFECT_INFO_DIM

    # Round-level 80/20 split (shared-z, correlated states of a round must not leak).
    uniq = np.unique(rounds)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    test_rounds = set(uniq[: max(1, len(uniq) // 5)].tolist())
    is_test = np.array([r in test_rounds for r in rounds])
    tr, te = ~is_test, is_test
    print(f"split: {tr.sum():,} train / {te.sum():,} test "
          f"({len(uniq)-len(test_rounds)}/{len(test_rounds)} rounds)\n")

    # --- Guardrail: confirm R^2_observable has plateaued w.r.t. data (cheap epochs). ---
    print("observable-only data-scale guardrail (test R^2 should flatten):")
    tr_rounds = np.array(sorted(set(rounds[tr].tolist())))
    for frac in (0.25, 0.5, 1.0):
        keep = set(tr_rounds[: max(1, int(len(tr_rounds) * frac))].tolist())
        m = np.array([r in keep for r in rounds]) & tr
        g = _fit(obs[m], z[m], input_dim=FEATURIZER_OUTPUT_DIM,
                 hidden=args.hidden, epochs=15, lr=args.lr)
        print(f"  data={int(frac*100):>3}% ({m.sum():>8,})  test R^2={r2_score(_pred(g, obs[te]), z[te]):.3f}")

    # --- The gate: observable-224 vs perfect-info-392, trained to convergence. ---
    # Report TRAIN R^2 too: a fair comparison requires the perfect-info critic
    # (a strict superset of the observable features) to be fit at least as well
    # as observable; perfect < observable signals under-fit, not "info useless".
    m_obs = _fit(obs[tr], z[tr], input_dim=FEATURIZER_OUTPUT_DIM,
                 hidden=args.hidden, epochs=args.epochs, lr=args.lr)
    m_pi = _fit(pi[tr], z[tr], input_dim=PERFECT_INFO_DIM,
                hidden=args.hidden, epochs=args.epochs, lr=args.lr)
    pred_obs, pred_pi = _pred(m_obs, obs[te]), _pred(m_pi, pi[te])
    print(f"\nfit check (epochs={args.epochs}, hidden={args.hidden}):")
    print(f"  observable   train R^2={r2_score(_pred(m_obs, obs[tr]), z[tr]):.3f}"
          f"  test R^2={r2_score(pred_obs, z[te]):.3f}")
    print(f"  perfect-info train R^2={r2_score(_pred(m_pi, pi[tr]), z[tr]):.3f}"
          f"  test R^2={r2_score(pred_pi, z[te]):.3f}")

    report = split_rho(z[te], pred_obs, pred_pi, call_live[te])
    print(f"\n{'split':<12}{'n':>9}{'R^2 obs':>10}{'R^2 perf':>10}{'rho':>9}")
    print("-" * 50)
    for key in ("overall", "call_live", "rest"):
        m = report[key]
        print(f"{key:<12}{m['n']:>9,}{m['r2_observable']:>10.3f}"
              f"{m['r2_perfect']:>10.3f}{m['rho']:>9.3f}")

    print(f"\nVERDICT (overall rho={report['overall']['rho']:.3f}): {_verdict(report['overall']['rho'])}")


if __name__ == "__main__":
    main()
