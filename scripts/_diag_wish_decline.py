"""[DIAG] Count Mahjong plays vs Wunsch lines in BSW data to measure the
decline ("wish nothing") rate. Throwaway."""
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(r"C:\workbench\tichu\.claude\worktrees\epic-dubinsky-b29c6c")
sys.path.insert(0, str(ROOT / "src"))

from tichu_engine.cards import MAHJONG
from tichu_training.bsw.archive import iter_archive
from tichu_training.bsw.parser import parse_tch

ARCHIVE = Path(r"C:\workbench\tichu\data\archive.zst")
MAX_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 4000

c = Counter()
mahjong_first_card = 0  # rounds where mahjong was played
for i, (gid, text) in enumerate(iter_archive(ARCHIVE)):
    if i >= MAX_GAMES:
        break
    try:
        game = parse_tch(text, game_id=gid)
    except Exception:
        c["parse_error"] += 1
        continue
    c["games_ok"] += 1
    for rnd in game.rounds:
        c["rounds"] += 1
        plays = rnd.plays
        has_mahjong = any(
            p.kind == "play" and p.cards and MAHJONG in p.cards for p in plays
        )
        has_wish = any(p.kind == "wish" for p in plays)
        if has_mahjong:
            c["rounds_with_mahjong_play"] += 1
            if has_wish:
                c["mahjong_with_wish"] += 1
            else:
                c["mahjong_no_wish_DECLINE"] += 1
        if has_wish and not has_mahjong:
            c["wish_without_mahjong_play_ANOMALY"] += 1

for k in sorted(c):
    print(f"{k:40s} {c[k]}")

mp = c["rounds_with_mahjong_play"]
if mp:
    print(f"\ndecline rate (no Wunsch | mahjong played) = "
          f"{c['mahjong_no_wish_DECLINE']}/{mp} = "
          f"{100*c['mahjong_no_wish_DECLINE']/mp:.2f}%")
