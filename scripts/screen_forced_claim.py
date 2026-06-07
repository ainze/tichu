"""Stage-1 screen (ADR-0032): how often does a chain Guaranteed Out arise at a
ForcedClaimAgent trigger (leading caller / live slam) in master self-play, and how
often does forcing it diverge from the master? Throwaway diagnostic; reads forced_count.
"""
import sys

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_eval.play_full import play_full_round
from tichu_training.search.forced_claim import ForcedClaimAgent

CKPT = r"C:\workbench\tichu\data\export\bc_full_corpus_v5_memmap_unshuffled"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
RECLAIM = len(sys.argv) > 2 and sys.argv[2] == "reclaim"


def _agent():
    return ForcedClaimAgent(
        rf"{CKPT}\policy.pt",
        skill_decile=9,
        schupfen_path=rf"{CKPT}\schupfen.pt",
        tichu_call_path=rf"{CKPT}\tichu_call.pt",
        grand_call_path=rf"{CKPT}\grand_tichu_call.pt",
        reclaim=RECLAIM,
    )


RESULT = r"C:\workbench\tichu\data\runs\forced_claim\screen.txt"


def _write(line: str) -> None:
    import os
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w") as f:
        f.write(line + "\n")
        f.flush()


def main() -> None:
    _write("status=started torch-loading")
    agents = [_agent() for _ in range(4)]
    _write("status=agents-loaded running")
    positions = generate_full_position_pool(seed=0, n=N)
    rounds_with_force = 0
    for i, pos in enumerate(positions):
        before = sum(a.forced_count for a in agents)
        play_full_round(agents, pos.state, pos.grand_prefixes)
        if sum(a.forced_count for a in agents) > before:
            rounds_with_force += 1
        if (i + 1) % 20 == 0:
            tot = sum(a.forced_count for a in agents)
            _write(f"status=running done={i + 1}/{N} forced_moves={tot} rounds_with_force={rounds_with_force}")
    total = sum(a.forced_count for a in agents)
    _write(f"status=DONE positions={N} forced_moves={total} rounds_with_force={rounds_with_force}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        _write("status=ERROR\n" + traceback.format_exc())
        raise
