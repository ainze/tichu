"""Human behavioral baseline by Skill Decile, straight from the parquet manifest.

Answers: is the `master` ML Agent SOFTER than the decile-9 humans it imitates,
or faithfully matching them? Produces the human-side counterpart to
`eval_matrix --mode behavioral`:

  * tichu_call_rate / grand_call_rate  — calls per player-round, by decile
  * bomb_per_round                     — bomb plays per player-round, by decile

Success rates are intentionally NOT reported: the manifest's `round_outcome` is
net round score (team0-team1), not "caller went out first" (the master metric),
so a side-by-side would mislead. A faithful success comparison needs the
replay's out-order and is a separate job. The RATE comparison is what answers
"is master soft?" — and rates need no outcome semantics.

Denominator note: call shards store only POSITIVE calls (action_taken is the
constant "tichu" / "grand_tichu"), so the opportunity denominator is the
schupfen shard — exactly one row per (player, round). Bomb plays are counted
from the play shard, which is huge (~1.45B rows): we sample every K-th row group
(evenly across the game_id-ordered file, so the recency-meta shift is covered)
and scale by 1/fraction. Bombs are detected structurally from the card list
(4-same-rank, or 5+ same-suit consecutive) since action_taken does not tag them.

Usage:
  py -3.14 scripts/human_call_bomb_baseline.py `
    --manifest C:\workbench\tichu\data\parquet_full_feat_v5 --play-rg-stride 8
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import pyarrow.parquet as pq

NEUTRAL = 10  # Neutral Skill Decile sentinel (see CONTEXT.md)


def _decile_counts(path: Path):
    """Per-decile row count for a shard (streaming, O(deciles) memory)."""
    counts = collections.Counter()
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=500_000, columns=["skill_decile"]):
        for d in b.column("skill_decile").to_pylist():
            counts[NEUTRAL if d is None else d] += 1
    return counts


def _is_bomb(action_taken: str) -> bool:
    """Structural bomb test on a 'play:card,card,...' string. Four-of-a-kind
    (4 same rank) or straight-flush (>=5 same suit, consecutive). Specials
    (no '-') can never be in a bomb, so their presence is an early reject."""
    if not action_taken.startswith("play:"):
        return False
    body = action_taken[5:]
    if body.count(",") < 3:          # < 4 cards -> cannot be a bomb
        return False
    ranks: list[int] = []
    suits = set()
    for c in body.split(","):
        if "-" not in c:             # a special card -> not a bomb
            return False
        suit, rank = c.rsplit("-", 1)
        try:
            ranks.append(int(rank))
        except ValueError:
            return False
        suits.add(suit)
    if len(ranks) == 4 and len(set(ranks)) == 1:
        return True                  # four-of-a-kind bomb
    if len(ranks) >= 5 and len(suits) == 1:
        rs = sorted(ranks)
        if all(rs[i] + 1 == rs[i + 1] for i in range(len(rs) - 1)):
            return True              # straight-flush bomb
    return False


def _bomb_counts_sampled(path: Path, stride: int):
    """Per-decile bomb-play count over every `stride`-th row group, plus the
    sampled play-row count per decile (for diagnostics). Returns (bombs, plays,
    fraction)."""
    bombs = collections.Counter()
    plays = collections.Counter()
    pf = pq.ParquetFile(path)
    n_rg = pf.num_row_groups
    sampled = list(range(0, n_rg, stride))
    for i, rg in enumerate(sampled, 1):
        t = pf.read_row_group(rg, columns=["action_taken", "skill_decile"])
        ats = t.column("action_taken").to_pylist()
        ds = t.column("skill_decile").to_pylist()
        for a, d in zip(ats, ds):
            d = NEUTRAL if d is None else d
            plays[d] += 1
            if _is_bomb(a):
                bombs[d] += 1
        if i % 20 == 0 or i == len(sampled):
            print(f"  play row-groups: {i}/{len(sampled)} sampled "
                  f"({rg}/{n_rg})", flush=True)
    fraction = len(sampled) / n_rg
    return bombs, plays, fraction


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--play-rg-stride", type=int, default=8,
                   help="sample every K-th play row group (default 8, ~12 pct)")
    args = p.parse_args()
    m = args.manifest

    print("scanning schupfen shard (player-round denominator) ...", flush=True)
    player_rounds = _decile_counts(m / "schupfen_00000.parquet")

    print("scanning call_tichu shard ...", flush=True)
    tichu = _decile_counts(m / "call_tichu_00000.parquet")
    print("scanning call_grand_tichu shard ...", flush=True)
    grand = _decile_counts(m / "call_grand_tichu_00000.parquet")

    print(f"sampling play shard (stride={args.play_rg_stride}) ...", flush=True)
    bombs, plays, frac = _bomb_counts_sampled(
        m / "play_00000.parquet", args.play_rg_stride)
    print(f"play sample fraction: {frac:.4f}\n", flush=True)

    def rate(num, den):
        return num / den if den else 0.0

    hdr = (f"{'decile':>7} {'player_rnds':>12} {'tichu_call':>11} "
           f"{'grand_call':>11} {'bomb/round':>11} {'bomb/1k_play':>13}")
    print(hdr)
    print("-" * len(hdr))
    for d in sorted(player_rounds):
        pr = player_rounds[d]
        bomb_full = bombs[d] / frac if frac else 0.0   # scale sample to full corpus
        bomb_per_1k_play = 1000 * rate(bombs[d], plays[d])
        line = (
            f"{('neutral' if d == NEUTRAL else d):>7} {pr:>12,} "
            f"{rate(tichu[d], pr):>11.4f} "
            f"{rate(grand[d], pr):>11.4f} "
            f"{rate(bomb_full, pr):>11.4f} {bomb_per_1k_play:>13.3f}"
        )
        print(line)
    print("\nmaster (decile-9 ML, self-play) for comparison:")
    print(f"{'master':>7} {'-':>12} {0.117:>11.4f} {0.106:>11.4f} {0.115:>11.4f} "
          f"{'-':>13}")
    print(f"{'neutral_ml':>7} {'-':>12} {0.112:>11.4f} {0.001:>11.4f} {0.129:>11.4f} "
          f"{'-':>13}")
    print("\nNote: call-rate denominator = schupfen rows (player-rounds; includes "
          "the ~6% grand-callers who can't then call tichu, so tichu_call is a "
          "hair low). bomb/round = bomb plays per player-round (play-shard sample, "
          f"frac={frac:.3f}). master/neutral rows are from eval_matrix --mode "
          "behavioral (self-play).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
