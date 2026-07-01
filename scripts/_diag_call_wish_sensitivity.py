"""[DIAG] Counterfactual: how much does the Tichu-call probability move when we
flip the Mahjong-wish feature block, holding everything else fixed?

Method: gather real pre-call states (each seat's first non-Pass Play state, the
ADR-0018 call-featurise moment) from BSW, then for each state overwrite the
15-dim `mahjong_wish` feature block with: (a) no-wish, and (b) each rank 2..14.
Forward the served v6 tichu_call.pt for every variant; compare P(call).

Two views:
  * generic     — Δ over ALL ranks 2..14 (does the dim matter at all?)
  * forced-rank — Δ over only ranks the PLAYER HOLDS (a wish they'd be forced
                  to satisfy — the straight-breaking scenario).
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"C:\workbench\tichu\.claude\worktrees\epic-dubinsky-b29c6c")
sys.path.insert(0, str(ROOT / "src"))

from tichu_engine.cards import Card
from tichu_training.bsw.archive import iter_archive
from tichu_training.bsw.parser import parse_tch
from tichu_training.bsw.replay import replay_round
from tichu_training.featurizer import featurize, SECTION_OFFSETS, SECTION_DIMS, mask_self_tichu_call
from tichu_inference.ml_agent import _with_round_local_scores, _NEUTRAL_SKILL

ARCHIVE = Path(r"C:\workbench\tichu\data\archive.zst")
MODEL = Path(r"C:\workbench\tichu\data\export\bc_full_corpus_v6_memmap\tichu_call.pt")
MAX_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
MAX_STATES = 15000

WOFF = SECTION_OFFSETS["mahjong_wish"]
WDIM = SECTION_DIMS["mahjong_wish"]  # 15: slots 0..12 ranks 2..14, 13 no-wish, 14 N/A

module = torch.jit.load(str(MODEL))
module.eval()


def wish_block(rank):
    """15-dim one-hot block. rank=None -> 'no wish active' (slot 13)."""
    b = np.zeros(WDIM, dtype=np.float32)
    b[13 if rank is None else (rank - 2)] = 1.0
    return b


# 1) Collect real pre-call states.
states = []  # (base_features[D], held_ranks set)
for gi, (gid, text) in enumerate(iter_archive(ARCHIVE)):
    if gi >= MAX_GAMES or len(states) >= MAX_STATES:
        break
    try:
        game = parse_tch(text, game_id=gid)
    except Exception:
        continue
    for rnd in game.rounds:
        rep = replay_round(rnd)
        if rep.final_state is None:
            continue
        seen = set()
        for (pa, _c), ps in zip(rep.decisions, rep.pre_decision_states):
            if ps is None or pa.kind != "play" or pa.player in seen:
                continue
            seen.add(pa.player)
            seat = pa.player
            priv = ps.private_view(seat)
            feats = featurize(_with_round_local_scores(priv)).astype(np.float32)
            mask_self_tichu_call(feats, seat)
            held = {c.rank for c in priv.hand if isinstance(c, Card) and 2 <= c.rank <= 14}
            states.append((feats, held))
        if len(states) >= MAX_STATES:
            break

N = len(states)
print(f"states={N}  model={MODEL.name}  wish_block@[{WOFF}:{WOFF+WDIM}]")

base = np.stack([s[0] for s in states])  # [N, D]
held = [s[1] for s in states]
skill = torch.full((N,), _NEUTRAL_SKILL, dtype=torch.long)


def p_call_for(rank):
    X = base.copy()
    X[:, WOFF:WOFF + WDIM] = wish_block(rank)
    with torch.no_grad():
        logits = module(torch.from_numpy(X), skill).cpu().numpy()  # [N,2] = [skip,call]
    return 1.0 / (1.0 + np.exp(logits[:, 0] - logits[:, 1]))


p_none = p_call_for(None)
p_rank = {r: p_call_for(r) for r in range(2, 15)}  # [N] each

# 2) Generic sensitivity: Δ vs no-wish, averaged over all ranks.
deltas_all = np.stack([p_rank[r] - p_none for r in range(2, 15)], axis=1)  # [N,13]
print("\n=== mean P(call) by scenario ===")
print(f"  no-wish          {p_none.mean():.4f}")
for r in range(2, 15):
    print(f"  wish={r:<2}          {p_rank[r].mean():.4f}   (Δ {p_rank[r].mean()-p_none.mean():+.4f})")

print("\n=== generic sensitivity (all ranks) ===")
print(f"  mean |Δ P(call)|         {np.abs(deltas_all).mean():.4f}")
print(f"  mean signed Δ            {deltas_all.mean():+.4f}")
print(f"  per-state max |Δ| (mean) {np.abs(deltas_all).max(axis=1).mean():.4f}")
print(f"  per-state max |Δ| (p95)  {np.percentile(np.abs(deltas_all).max(axis=1),95):.4f}")
print(f"  per-state max |Δ| (max)  {np.abs(deltas_all).max():.4f}")

# 3) Forced-rank: Δ only for wishes the player HOLDS (must satisfy).
forced_max, forced_mean = [], []
for i, h in enumerate(held):
    if not h:
        continue
    d = [abs(p_rank[r][i] - p_none[i]) for r in h]
    forced_max.append(max(d))
    forced_mean.append(float(np.mean(d)))
forced_max = np.array(forced_max)
forced_mean = np.array(forced_mean)
print(f"\n=== forced-rank sensitivity (wish ∈ player's hand, n={len(forced_max)}) ===")
print(f"  mean |Δ| over held ranks     {forced_mean.mean():.4f}")
print(f"  per-state max |Δ| (mean)     {forced_max.mean():.4f}")
print(f"  per-state max |Δ| (p95)      {np.percentile(forced_max,95):.4f}")
print(f"  per-state max |Δ| (max)      {forced_max.max():.4f}")
print(f"  states where a held wish moves P(call) by >0.05: "
      f"{100*(forced_max>0.05).mean():.1f}%  | >0.10: {100*(forced_max>0.10).mean():.1f}%")

# 4) Among 'live' call decisions (base P in [0.05,0.95]) — where it can matter.
live = (p_none >= 0.05) & (p_none <= 0.95)
print(f"\n=== among live decisions (base P(call)∈[0.05,0.95], n={live.sum()}) ===")
if live.sum():
    dlive = np.abs(deltas_all)[live]
    print(f"  mean |Δ| (all ranks)         {dlive.mean():.4f}")
    print(f"  per-state max |Δ| (mean)     {dlive.max(axis=1).mean():.4f}")
