r"""Search + learning generation loop (ADR-0031 Phase 1).

One generation: self-play with the current `(policy_g, critic_g)` under PIMC + root
Dirichlet noise + τ-sampling (SelfPlaySearchAgent) → collect `(features, π, z)` →
retrain the play head toward π (masked KL) and the value toward z (fit_value_baseline),
warm-started from generation g → export `policy_{g+1}.pt` + `critic_{g+1}.bin`. Repeat.

Per ADR-0031 Decision A the self-play runs through `play_full_round` (full call stack +
call-bonus outcome). Collection is SERIAL here; the Decision-K multiprocessing fan-out is
the throughput step (add `--workers` later) — at the smoke / small-Phase-1 scale serial is
fine. policy_0's trainable checkpoint is the BC final `step_000001.bin` (verified byte-equal
to the master export); critic_0 is the master-self-play critic.

Usage (PowerShell):
  $env:PYTHONPATH = (Resolve-Path src).Path
  py -3.14 scripts\run_search_learning.py --generations 3 --rounds 50 --worlds 4 --sims 50
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import pathlib
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from tichu_eval.behavioral import run_behavioral_profiles
from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_export.torchscript import export_torchscript
from tichu_inference.ml_agent import MLAgent
from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.awr.value_baseline import ValueBaseline, fit_value_baseline
from tichu_training.bc.heads import BCModel
from tichu_training.bc.training import load_checkpoint, save_checkpoint
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM, FEATURIZER_VERSION
from tichu_training.search.critic import CriticValue, save_critic
from tichu_training.search.loop import collect_round, train_play_head
from tichu_training.search.selfplay import SelfPlaySearchAgent

ARCH = dict(feature_dim=FEATURIZER_OUTPUT_DIM, trunk_hidden=1024, trunk_depth=4,
            trunk_out_dim=512, skill_dim=64, head_hidden=256)
VALUE_HIDDEN = 512

EXPORT = r"C:\workbench\tichu\data\export\bc_full_corpus_v5_memmap_unshuffled"
SCHUPFEN = EXPORT + r"\schupfen.pt"
TICHU = EXPORT + r"\tichu_call.pt"
GRAND = EXPORT + r"\grand_tichu_call.pt"
POLICY0_CKPT = (r"C:\workbench\tichu\data\runs\bc_full_corpus_v5_memmap_unshuffled"
                r"\checkpoints\step_000001.bin")
CRITIC0 = r"C:\workbench\tichu\data\export\search_critic_master_selfplay_v5\critic.bin"


def _load_value_baseline(path) -> ValueBaseline:
    """Warm-start a trainable ValueBaseline from a save_critic blob."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    baseline = ValueBaseline(blob["feature_dim"], hidden=blob["hidden"])
    baseline.load_state_dict(blob["state_dict"])
    return baseline


def _collect_shard(payload):
    """One worker: build a frozen master + critic leaf (one load serves all four seats,
    inference is stateless), self-play this shard's positions, return its Samples. Pinned
    to one torch thread so N spawn workers don't oversubscribe (MCTS is Python-bound)."""
    torch.set_num_threads(1)
    policy_pt, critic_bin, positions, hp, base_seed = payload
    master = MLAgent(policy_pt, skill_decile=9, schupfen_path=SCHUPFEN,
                     tichu_call_path=TICHU, grand_call_path=GRAND)
    leaf = CriticValue(_load_value_baseline(critic_bin))
    agents = [
        SelfPlaySearchAgent.from_policy(master, leaf_fn=leaf, seed=base_seed + seat, **hp)
        for seat in range(4)
    ]
    out = []
    for pos in positions:
        out.extend(collect_round(agents, pos))
    return out


def _collect(policy_pt, critic_bin, positions, hp, workers):
    """Self-play collection, fanned out over `workers` spawn processes on round-robin
    position shards (Decision K). workers=1 runs in-process (smoke / tests)."""
    shards = [positions[i::workers] for i in range(workers)]
    payloads = [(policy_pt, critic_bin, shards[i], hp, 1000 * i)
                for i in range(workers) if shards[i]]
    if workers == 1:
        return _collect_shard(payloads[0])
    with mp.get_context("spawn").Pool(workers) as pool:
        results = pool.map(_collect_shard, payloads)
    return [s for shard in results for s in shard]


def _eval_passivity(policy_pt, positions) -> dict:
    """Cheap raw-policy (no-search) behavioral dial — the per-generation headline readout
    (Decision J). Builds an MLAgent from the exported policy and profiles it in self-play."""
    def builder():
        return MLAgent(policy_pt, skill_decile=9, schupfen_path=SCHUPFEN,
                       tichu_call_path=TICHU, grand_call_path=GRAND)
    return run_behavioral_profiles({"p": builder}, positions)["p"].as_row()


def _train_and_export(samples, in_ckpt, out_dir, *, play_epochs, play_lr,
                      value_epochs, value_lr):
    out_dir.mkdir(parents=True, exist_ok=True)
    # --- play head: warm-start from the previous generation, KL toward visit π ---
    model = BCModel(**ARCH)
    load_checkpoint(in_ckpt, model)
    hist = train_play_head(model, samples, epochs=play_epochs, lr=play_lr)
    ckpt_out = out_dir / "trainable.bin"
    save_checkpoint(model, torch.optim.Adam(model.parameters()), step=0, path=ckpt_out)
    model.eval()
    export_torchscript(
        model, example_inputs=(torch.randn(1, FEATURIZER_OUTPUT_DIM),
                               torch.tensor([0], dtype=torch.long)),
        featurizer_version=FEATURIZER_VERSION, action_space_version=ACTION_SPACE_VERSION,
        output_path=out_dir / "policy.pt",
    )
    # --- value: warm-start from the previous critic, MSE toward MC outcome z ---
    feats = np.stack([s.features for s in samples]).astype(np.float32)
    z = np.array([s.z for s in samples], dtype=np.float32)
    baseline = ValueBaseline(FEATURIZER_OUTPUT_DIM, hidden=VALUE_HIDDEN)
    mse = fit_value_baseline(baseline, feats, z, batch_size=4096,
                             epochs=value_epochs, lr=value_lr, show_progress=False)
    save_critic(baseline, out_dir / "critic.bin", feature_dim=FEATURIZER_OUTPUT_DIM,
                hidden=VALUE_HIDDEN, note=f"search+learning gen, {len(z)} samples, mse={mse:.4f}")
    return hist, mse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--c-puct", type=float, default=1.4)
    ap.add_argument("--root-alpha", type=float, default=1.0)
    ap.add_argument("--root-eps", type=float, default=0.25)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--play-epochs", type=int, default=4)
    ap.add_argument("--play-lr", type=float, default=1e-4)
    ap.add_argument("--value-epochs", type=int, default=8)
    ap.add_argument("--value-lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-rounds", type=int, default=120,
                    help="fixed raw-policy passivity eval set, disjoint from training")
    ap.add_argument("--pos-seed", type=int, default=0)
    ap.add_argument("--out-dir", default=r"C:\workbench\tichu\data\runs\search_learning_v1")
    args = ap.parse_args()

    hp = dict(worlds=args.worlds, sims=args.sims, c_puct=args.c_puct,
              root_alpha=args.root_alpha, root_eps=args.root_eps,
              temperature=args.temperature)
    # Fixed eval set, seeded far from any training seed (pos_seed + g*rounds) so the
    # passivity dial is measured on deals the loop never trained on.
    eval_positions = generate_full_position_pool(seed=900_000, n=args.eval_rounds)

    out_root = pathlib.Path(args.out_dir)
    gen0 = out_root / "gen0"
    gen0.mkdir(parents=True, exist_ok=True)
    shutil.copy(EXPORT + r"\policy.pt", gen0 / "policy.pt")
    shutil.copy(CRITIC0, gen0 / "critic.bin")
    shutil.copy(POLICY0_CKPT, gen0 / "trainable.bin")

    metrics_csv = out_root / "metrics.csv"
    fields = ["gen", "n_samples", "train_rounds", "play_kl_first", "play_kl_last",
              "value_mse", "mean_z", "caller_passivity_rate", "caller_pass_opportunities",
              "caller_bomb_passivity_rate", "bomb_when_legal_rate", "seconds"]
    with metrics_csv.open("w", encoding="utf-8", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields).writeheader()

    def _log(row):
        with metrics_csv.open("a", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow({k: row.get(k, "") for k in fields})

    # Generation 0 = the master itself: record its passivity baseline for the curve.
    base = _eval_passivity(str(gen0 / "policy.pt"), eval_positions)
    print(f"[gen 0 master] caller_passivity={base['caller_passivity_rate']:.3f} "
          f"(opp {base['caller_pass_opportunities']})")
    _log({"gen": 0, "caller_passivity_rate": base["caller_passivity_rate"],
          "caller_pass_opportunities": base["caller_pass_opportunities"],
          "caller_bomb_passivity_rate": base.get("caller_bomb_passivity_rate", ""),
          "bomb_when_legal_rate": base.get("bomb_when_legal_rate", "")})

    policy_pt = str(gen0 / "policy.pt")
    critic_bin = str(gen0 / "critic.bin")
    trainable = str(gen0 / "trainable.bin")

    for g in range(1, args.generations + 1):
        t0 = time.perf_counter()
        positions = generate_full_position_pool(seed=args.pos_seed + (g - 1) * args.rounds,
                                                 n=args.rounds)
        samples = _collect(policy_pt, critic_bin, positions, hp, args.workers)
        gen_dir = out_root / f"gen{g}"
        hist, mse = _train_and_export(
            samples, trainable, gen_dir,
            play_epochs=args.play_epochs, play_lr=args.play_lr,
            value_epochs=args.value_epochs, value_lr=args.value_lr)
        policy_pt = str(gen_dir / "policy.pt")
        critic_bin = str(gen_dir / "critic.bin")
        trainable = str(gen_dir / "trainable.bin")

        prof = _eval_passivity(policy_pt, eval_positions)
        dt = time.perf_counter() - t0
        row = {"gen": g, "n_samples": len(samples), "train_rounds": args.rounds,
               "play_kl_first": round(hist[0], 4), "play_kl_last": round(hist[-1], 4),
               "value_mse": round(mse, 4),
               "mean_z": round(float(np.mean([s.z for s in samples])), 4),
               "caller_passivity_rate": prof["caller_passivity_rate"],
               "caller_pass_opportunities": prof["caller_pass_opportunities"],
               "caller_bomb_passivity_rate": prof.get("caller_bomb_passivity_rate", ""),
               "bomb_when_legal_rate": prof.get("bomb_when_legal_rate", ""),
               "seconds": round(dt)}
        _log(row)
        print(f"[gen {g}] {len(samples):,} samples | play KL {hist[0]:.3f}->{hist[-1]:.3f} | "
              f"value mse {mse:.4f} | mean z {row['mean_z']:+.3f} | "
              f"caller_passivity {prof['caller_passivity_rate']:.3f} "
              f"(opp {prof['caller_pass_opportunities']}) | {dt:.0f}s")

    print(f"done. metrics -> {metrics_csv}")


if __name__ == "__main__":
    main()
