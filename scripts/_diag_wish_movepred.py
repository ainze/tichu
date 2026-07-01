"""[DIAG] Wish-head move-prediction: do old vs wishfix BC reproduce human
Mahjong-wish decisions? Breaks out DECLINE (rank=None) vs RANK agreement — the
old head structurally can't output decline, so its decline top1 should be ~0."""
import sys
from pathlib import Path

ROOT = Path(r"C:\workbench\tichu\.claude\worktrees\epic-dubinsky-b29c6c")
sys.path.insert(0, str(ROOT / "src"))

from tichu_engine.legality import legal_actions_for
from tichu_training.bsw.archive import iter_archive, list_game_ids
from tichu_training.bsw.parser import parse_tch
from tichu_eval.move_prediction import decisions_from_game
from tichu_training.cli.eval_matrix import _build_agent

ARCHIVE = Path(r"C:\workbench\tichu\data\archive.zst")
OLD = r"C:\workbench\tichu\data\export\bc_full_corpus_v6_memmap"
MAXG = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

# Wish decisions only use the policy's wish head; schupfen/tichu/grand are
# irrelevant here, so every agent shares the old v6 export's standalone nets and
# differs only in policy.pt (= the wish head under test).
POLICIES = {
    "bc_v6":         fr"{OLD}\policy.pt",
    "bc_v6_wishfix": r"C:\workbench\tichu\data\export\bc_full_corpus_v6_wishfix_memmap\policy.pt",
    "cotrain_10368": r"C:\workbench\tichu\data\runs\cotrain_v6_pbrs_resid_wish_gated_wishfix\export\iter_10368\policy.pt",
    "cpfix3328":     r"C:\workbench\tichu\data\runs\cotrain_v6_pbrs_resid_wish_gated_cpfix\export\iter_03328\policy.pt",
}


def _ml(policy_pt):
    return _build_agent(
        "ml",
        checkpoint_path=policy_pt,
        skill_decile=9,
        schupfen_path=fr"{OLD}\schupfen.pt",
        tichu_call_path=fr"{OLD}\tichu_call.pt",
        grand_call_path=fr"{OLD}\grand_tichu_call.pt",
    )


agents = {name: _ml(p) for name, p in POLICIES.items()}

# Collect wish decisions from a strided archive sample.
ids = list_game_ids(ARCHIVE)
step = max(1, len(ids) // MAXG)
ids = set(ids[::step][:MAXG])
wish = []
bad = 0
for gid, text in iter_archive(ARCHIVE, game_ids=ids):
    try:
        g = parse_tch(text, game_id=gid)
    except Exception:
        bad += 1
        continue
    for d in decisions_from_game(g):
        if d.decision_type == "wish" and legal_actions_for(d.private_state):
            wish.append(d)

human_declines = sum(1 for d in wish if d.human_action.rank is None)
print(f"games~{len(ids)} (skipped {bad})  wish decisions={len(wish)}  "
      f"human declines={human_declines} ({100*human_declines/len(wish):.1f}%)\n")

for name, agent in agents.items():
    tot = hit = 0
    dtot = dhit = rtot = rhit = 0
    pred_decline = 0
    for d in wish:
        chosen = agent.act(d.private_state)
        ok = chosen == d.human_action
        is_decl = d.human_action.rank is None
        tot += 1
        hit += ok
        if chosen.rank is None:
            pred_decline += 1
        if is_decl:
            dtot += 1
            dhit += ok
        else:
            rtot += 1
            rhit += ok
    print(f"{name:>14s}  wish top1={hit/tot:.3f} (n={tot})  |  "
          f"decline top1={dhit/dtot:.3f} (n={dtot})  |  "
          f"rank top1={rhit/rtot:.3f} (n={rtot})  |  "
          f"agent declines {100*pred_decline/tot:.1f}% of the time")
