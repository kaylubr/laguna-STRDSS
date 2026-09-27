import numpy as np
import pandas as pd

from str_suitability.config import SUITABILITY_INDICATOR_DIRECTIONS
from str_suitability.suitability.validate import (
    assert_within_unit_interval,
    validate_suitability_inputs,
)

NORMALIZED_PREFIX = "normalized_"
POSITIVE_DIRECTION = "positive"
NEGATIVE_DIRECTION = "negative"


def normalized_column(indicator: str) -> str:
    return f"{NORMALIZED_PREFIX}{indicator}"


def indicator_from_normalized(column: str) -> str:
    return column.removeprefix(NORMALIZED_PREFIX)


def minimum_maximum_normalize(
    grid: pd.DataFrame, directions: dict[str, str] | None = None, transform=None
) -> pd.DataFrame:
    directions = (
        SUITABILITY_INDICATOR_DIRECTIONS if directions is None else dict(directions)
    )
    validate_suitability_inputs(grid, indicators=tuple(directions), directions=directions)

    normalized = pd.DataFrame(index=grid.index)
    for indicator in directions:
        values = pd.to_numeric(grid[indicator], errors="raise").astype(float)
        if transform is not None:
            values = transform(values)
        minimum = float(values.min())
        maximum = float(values.max())
        assert maximum > minimum, (
            f"indicator {indicator} takes the single value {minimum} across all cells, "
            "so min-max normalization has no span to divide by"
        )

        span = maximum - minimum
        direction = directions[indicator]
        if direction == NEGATIVE_DIRECTION:
            normalized[normalized_column(indicator)] = (maximum - values) / span
        elif direction == POSITIVE_DIRECTION:
            normalized[normalized_column(indicator)] = (values - minimum) / span
        else:
            raise ValueError(f"indicator {indicator} has an unknown direction: {direction}")

    assert_within_unit_interval(normalized, list(normalized.columns))
    return normalized
