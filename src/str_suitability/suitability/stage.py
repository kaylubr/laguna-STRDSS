import pandas as pd

from str_suitability.config import (
    BLEND_RATIO,
    EWM_WEIGHTING,
    HYBRID_WEIGHTING,
    POI_BLOC_INDICATORS,
    SUITABILITY_INDICATOR_DIRECTIONS,
)
from str_suitability.suitability.classify import (
    CLASS_COUNT,
    CLASS_LABELS,
    CLASSIFICATION_METHOD,
    SUITABILITY_CLASS_COLUMN,
    classify_suitability,
)
from str_suitability.suitability.composite_score import (
    COMPOSITE_SCORE_COLUMN,
    composite_suitability_score,
)
from str_suitability.suitability.entropy_weights import weights_from_normalized
from str_suitability.suitability.hybrid_weights import bloc_weight, equal_weights, hybrid_weights
from str_suitability.suitability.normalize import (
    indicator_from_normalized,
    minimum_maximum_normalize,
    normalized_column,
)


def _named(weights: pd.Series) -> dict[str, float]:
    return {
        indicator_from_normalized(column): float(weight) for column, weight in weights.items()
    }


def compute_suitability(
    grid: pd.DataFrame,
    directions: dict[str, str] | None = None,
    weighting: str = EWM_WEIGHTING,
    rfi_weights: pd.Series | None = None,
    transform=None,
) -> tuple[pd.DataFrame, dict]:
    directions = (
        SUITABILITY_INDICATOR_DIRECTIONS if directions is None else dict(directions)
    )
    normalized = minimum_maximum_normalize(grid, directions=directions, transform=transform)
    entropy = weights_from_normalized(normalized)

    if weighting == EWM_WEIGHTING:
        weights = entropy
    elif weighting == HYBRID_WEIGHTING:
        assert rfi_weights is not None, (
            "hybrid weighting needs the random-forest permutation importances"
        )
        aligned_rfi = pd.Series(
            rfi_weights.reindex(
                [indicator_from_normalized(column) for column in normalized.columns]
            ).to_numpy(),
            index=normalized.columns,
        )
        weights = hybrid_weights(entropy, aligned_rfi)
    else:
        raise ValueError(f"unknown weighting method: {weighting}")

    reference = equal_weights(normalized.columns)
    scores = composite_suitability_score(normalized, weights)
    labels, classification_report = classify_suitability(scores)

    result = grid.copy()
    for column in normalized.columns:
        result[column] = normalized[column]
    result[COMPOSITE_SCORE_COLUMN] = scores
    result[SUITABILITY_CLASS_COLUMN] = labels

    assert result[COMPOSITE_SCORE_COLUMN].notna().all(), (
        "every retained grid cell must carry a composite suitability score"
    )
    assert result[SUITABILITY_CLASS_COLUMN].notna().all(), (
        "every retained grid cell must carry a suitability class"
    )

    bloc_columns = [normalized_column(indicator) for indicator in POI_BLOC_INDICATORS]
    report = {
        "cells": int(len(result)),
        "indicators": list(directions),
        "class_count": CLASS_COUNT,
        "classification_method": CLASSIFICATION_METHOD,
        "weighting_method": weighting,
        "blend_ratio": float(BLEND_RATIO) if weighting == HYBRID_WEIGHTING else None,
        "weights": _named(weights),
        "weights_ewm": _named(entropy),
        "weights_equal": _named(reference),
        "poi_bloc_weight": bloc_weight(weights, bloc_columns),
        "poi_bloc_weight_ewm": bloc_weight(entropy, bloc_columns),
        "weight_sum": float(weights.sum()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "class_labels": list(CLASS_LABELS),
        "cells_with_score": int(result[COMPOSITE_SCORE_COLUMN].notna().sum()),
        "cells_with_class": int(result[SUITABILITY_CLASS_COLUMN].notna().sum()),
        **classification_report,
    }
    return result, report
