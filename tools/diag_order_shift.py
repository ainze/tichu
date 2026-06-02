r"""[DEBUG-ord1] Diagnose the loss regime-change in bc_full_corpus_v5_memmap.

Walks the play_meta memmap (emission order == training order, since the
trainer feeds iter_batches(preserve_order=True) with no shuffle) and bins
it along the row axis. For each bin it samples rows and reports the
distribution of skill_decile, round_outcome, game_won, sample_weight, and
target — plus a degeneracy check (all-zero feature rows) to rule out a
truncated/corrupt tail.

If the tail bins differ sharply from the body, the loss jump is a
non-i.i.d. ordering artifact, not optimizer instability or corruption.

Run:
  py -3.14 tools\diag_order_shift.py C:\workbench\tichu\data\materialised_full_v5\bc
"""
import json
import sys
from pathlib import Path

import numpy as np

META_DTYPE = np.dtype([
    ("target", "u2"),
    ("sample_weight", "f4"),
    ("skill_decile", "u1"),
    ("round_outcome", "f4"),
    ("game_won", "i1"),
])

N_BINS = 60
SAMPLE_PER_BIN = 100_000


def main() -> int:
    data_dir = Path(sys.argv[1])
    manifest = json.loads((data_dir / "manifest.json").read_text())
    n_play = manifest["counts"]["play"]
    meta = np.memmap(
        data_dir / "play_meta.dat", dtype=META_DTYPE, mode="r", shape=(n_play,),
    )
    # feat_bits for the degeneracy (all-zero indicator row) check.
    fb_info = manifest["files"]["play"]
    feat_bits_bytes = fb_info["shape_feat_bits"][1]
    feat_bits = np.memmap(
        data_dir / "play_feat_bits.dat", dtype=np.uint8, mode="r",
        shape=(n_play, feat_bits_bytes),
    )

    print(f"play rows: {n_play:,}   binning into {N_BINS} "
          f"(~{n_play // N_BINS:,} rows/bin), sampling {SAMPLE_PER_BIN:,}/bin\n")
    edges = np.linspace(0, n_play, N_BINS + 1, dtype=np.int64)

    hdr = (f"{'bin':>3} {'row_start':>14} {'frac':>5} "
           f"{'skill_mean':>10} {'rnd_out':>8} {'win_rate':>8} "
           f"{'samp_w':>7} {'top_tgt%':>8} {'n_tgt':>6} {'zero_feat%':>10}")
    print(hdr)
    print("-" * len(hdr))

    for b in range(N_BINS):
        lo, hi = int(edges[b]), int(edges[b + 1])
        span = hi - lo
        step = max(1, span // SAMPLE_PER_BIN)
        idx = np.arange(lo, hi, step)
        m = meta[idx]
        skill = m["skill_decile"].astype(np.int32)
        rnd = m["round_outcome"].astype(np.float64)
        won = m["game_won"].astype(np.int32)
        won_valid = won[won >= 0]
        win_rate = won_valid.mean() if won_valid.size else float("nan")
        sw = m["sample_weight"].astype(np.float64)
        tgt = m["target"].astype(np.int64)
        counts = np.bincount(tgt)
        top_frac = counts.max() / tgt.size
        n_distinct = int((counts > 0).sum())
        # degeneracy: indicator block all zero (would mean truncated/corrupt).
        fb = feat_bits[idx]
        zero_feat = float((fb == 0).all(axis=1).mean())

        print(f"{b:>3} {lo:>14,} {hi / n_play:>5.2f} "
              f"{skill.mean():>10.3f} {rnd.mean():>8.3f} {win_rate:>8.3f} "
              f"{sw.mean():>7.3f} {top_frac * 100:>7.2f}% {n_distinct:>6} "
              f"{zero_feat * 100:>9.2f}%")

    # Skill-decile histogram for body vs tail (last 24%).
    print("\nskill_decile histogram (fraction of sampled rows):")
    cut = int(0.756 * n_play)
    for label, lo, hi in (("body[0:75.6%]", 0, cut), ("tail[75.6%:100%]", cut, n_play)):
        span = hi - lo
        step = max(1, span // 500_000)
        s = meta[np.arange(lo, hi, step)]["skill_decile"].astype(np.int32)
        h = np.bincount(s, minlength=11)[:11] / s.size
        print(f"  {label:>18}: " + " ".join(f"{v:.3f}" for v in h))
    return 0


if __name__ == "__main__":
    sys.exit(main())
