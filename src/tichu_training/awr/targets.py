"""Value-baseline target selection for AWR Refine.

The Value Baseline V(s) regresses against a per-row target. Two targets
are supported, selected by the `awr.value_target` config knob:

- `"round"` — `BCExample.round_outcome` (per-round team-relative score
  delta). The v1/v2 target; high variance from slams + grand tichus.
- `"game"` — `BCExample.game_won` cast to 0.0/1.0 (team-relative game
  outcome). Sparser per-game signal, lower variance, aligned with the
  actual objective. Rows from an Incomplete Session carry
  `game_won is None` and `target_value` returns None for them — the
  caller filters such rows out of the baseline fit. See ADR-0013 and
  the Value Target entry in CONTEXT.md.

Kept in its own module — and taking primitives rather than a BCExample —
so it stays torch-free and unit-testable. Callers pass the relevant
BCExample fields directly.
"""


VALUE_TARGETS: tuple[str, ...] = ("round", "game")


def target_value(
    *,
    round_outcome: float,
    game_won: bool | None,
    kind: str,
) -> float | None:
    """Return the regression target under the given `kind`, or None if
    the row should be filtered out of the fit.

    Raises ValueError on an unknown kind.
    """
    if kind == "round":
        return round_outcome
    if kind == "game":
        if game_won is None:
            return None
        return 1.0 if game_won else 0.0
    raise ValueError(
        f"unknown value_target {kind!r}; expected one of {VALUE_TARGETS}"
    )
