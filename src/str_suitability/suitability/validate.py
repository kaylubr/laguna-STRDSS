import pandas as pd

from str_suitability.config import SUITABILITY_INDICATOR_DIRECTIONS, SUITABILITY_INDICATORS
from str_suitability.spatial.grid import assert_one_row_per_cell

MINIMUM_INDICATOR_COUNT = 2


def validate_suitability_inputs(
    grid: pd.DataFrame,
    indicators=None,
    directions=None,
) -> dict[str, object]:
    indicators = SUITABILITY_INDICATORS if indicators is None else tuple(indicators)
    directions = (
        SUITABILITY_INDICATOR_DIRECTIONS if directions is None else dict(directions)
    )

    assert_one_row_per_cell(grid)
    assert set(indicators) == set(directions), (
        "the indicator set and its directions must name the same indicators"
    )
    assert len(indicators) >= MINIMUM_INDICATOR_COUNT, (
        f"a composite needs at least {MINIMUM_INDICATOR_COUNT} indicators, "
        f"found {len(indicators)}"
    )

    absent = [name for name in indicators if name not in grid.columns]
    assert not absent, f"missing suitability indicators: {absent}"

    null_counts = grid[list(indicators)].isna().sum()
    incomplete = {name: int(count) for name, count in null_counts.items() if count > 0}
    assert not incomplete, f"suitability indicators are incomplete: {incomplete}"

    return {
        "cells": int(len(grid)),
        "indicators": list(indicators),
        "directions": directions,
    }


def assert_within_unit_interval(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        assert values.notna().all(), f"normalized indicator {column} contains nulls"
        assert values.min() >= 0.0, f"normalized indicator {column} below 0: {values.min()}"
        assert values.max() <= 1.0, f"normalized indicator {column} above 1: {values.max()}"
