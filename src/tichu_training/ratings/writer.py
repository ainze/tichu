"""Write the ratings table to Parquet."""

from pathlib import Path
from typing import Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from tichu_training.ratings.calls import CallStats
from tichu_training.ratings.sweep import PlayerRating


_SCHEMA = pa.schema([
    ("player_handle", pa.string()),
    ("mu", pa.float64()),
    ("sigma", pa.float64()),
    ("n_games", pa.int32()),
    ("skill_decile", pa.int32()),
    ("tichu_calls", pa.int32()),
    ("tichu_success_rate", pa.float64()),
    ("grand_tichu_calls", pa.int32()),
    ("grand_tichu_success_rate", pa.float64()),
])


def write_ratings_parquet(
    ratings: Mapping[str, PlayerRating],
    output_path: Path,
    deciles: Mapping[str, int] | None = None,
    call_stats: Mapping[str, CallStats] | None = None,
) -> None:
    """Write one row per handle in sorted handle order.

    Missing entries in `deciles` or `call_stats` default to null/0 columns.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handles = sorted(ratings.keys())
    deciles = deciles or {}
    call_stats = call_stats or {}

    def _calls(handle: str) -> CallStats:
        return call_stats.get(handle) or CallStats(handle=handle)

    columns = {
        "player_handle": handles,
        "mu": [ratings[h].mu for h in handles],
        "sigma": [ratings[h].sigma for h in handles],
        "n_games": [ratings[h].n_games for h in handles],
        "skill_decile": [deciles.get(h) for h in handles],
        "tichu_calls": [_calls(h).tichu_calls for h in handles],
        "tichu_success_rate": [_calls(h).tichu_success_rate for h in handles],
        "grand_tichu_calls": [_calls(h).grand_tichu_calls for h in handles],
        "grand_tichu_success_rate": [_calls(h).grand_tichu_success_rate for h in handles],
    }
    table = pa.Table.from_pydict(columns, schema=_SCHEMA)
    pq.write_table(table, output_path)
