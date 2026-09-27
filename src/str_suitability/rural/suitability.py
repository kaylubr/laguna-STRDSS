import geopandas as gpd
import pandas as pd

from str_suitability.config import EWM_WEIGHTING, NEARBY_TRAINING_THRESHOLD_KM
from str_suitability.features.accessibility import distance_to_nearest_km
from str_suitability.rural.classify import (
    CELL_CLASS_COLUMN,
    CELL_ID_COLUMN,
    RURAL_CELL,
    UNCLASSIFIED_CELL,
    URBAN_CELL,
)
from str_suitability.suitability.classify import (
    CLASS_LABELS,
    SUITABILITY_CLASS_COLUMN,
    TRAFFIC_CLASS_LABELS,
)
from str_suitability.suitability.composite_score import COMPOSITE_SCORE_COLUMN
from str_suitability.suitability.stage import compute_suitability

RURAL_SCORE_COLUMN = "rural_composite_suitability_score"
RURAL_CLASS_COLUMN = "rural_suitability_class"
IN_RURAL_ANALYSIS_COLUMN = "in_rural_analysis"
NEARBY_DISTANCE_COLUMN = "distance_to_nearest_training_listing"
HAS_NEARBY_COLUMN = "has_nearby_training_data"

RURAL_CLASS_LABELS = (
    "Rural — Very Low",
    "Rural — Low",
    "Rural — Moderate",
    "Rural — High",
    "Rural — Very High",
)

OUTSIDE_RURAL_ANALYSIS_LABELS = {
    URBAN_CELL: "Urban / Outside Rural Analysis",
    UNCLASSIFIED_CELL: "Unclassified / Outside Rural Analysis",
}


def rural_class_lookup() -> dict[str, str]:
    assert len(RURAL_CLASS_LABELS) == len(CLASS_LABELS), (
        f"{len(RURAL_CLASS_LABELS)} rural class labels for {len(CLASS_LABELS)} suitability classes"
    )
    assert RURAL_CLASS_LABELS[0].endswith("Very Low"), (
        "the rural labels must run from the lowest class to the highest, as the suitability classes do"
    )
    assert RURAL_CLASS_LABELS[-1].endswith("Very High"), (
        "the rural labels must run from the lowest class to the highest, as the suitability classes do"
    )
    return dict(zip(CLASS_LABELS, RURAL_CLASS_LABELS))


def compute_rural_suitability(
    rural_cells: pd.DataFrame,
    directions: dict[str, str] | None = None,
    weighting: str = EWM_WEIGHTING,
    rfi_weights: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    assert len(rural_cells) > 0, "there are no rural cells to score"
    scored, report = compute_suitability(
        rural_cells, directions=directions, weighting=weighting, rfi_weights=rfi_weights
    )

    result = pd.DataFrame(
        {
            CELL_ID_COLUMN: scored[CELL_ID_COLUMN],
            RURAL_SCORE_COLUMN: scored[COMPOSITE_SCORE_COLUMN],
            RURAL_CLASS_COLUMN: scored[SUITABILITY_CLASS_COLUMN].map(rural_class_lookup()),
        }
    )
    assert result[RURAL_SCORE_COLUMN].notna().all(), "a rural cell has no rural suitability score"
    assert result[RURAL_CLASS_COLUMN].notna().all(), "a rural cell has no rural suitability class"
    assert result[CELL_ID_COLUMN].is_unique, "the rural suitability result repeats a cell"

    report.update(
        {
            "rural_cells_scored": int(len(result)),
            "rural_class_labels": list(RURAL_CLASS_LABELS),
            "outside_rural_analysis_labels": dict(OUTSIDE_RURAL_ANALYSIS_LABELS),
        }
    )
    return result, report


def attach_training_support(
    output: gpd.GeoDataFrame, observations: pd.DataFrame
) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    assert {"longitude", "latitude"} <= set(output.columns), (
        "the rural output carries no coordinates to measure support distance from"
    )
    assert len(observations) > 0, "there are no fitted listings to measure training support against"
    distances = distance_to_nearest_km(
        output["longitude"].to_numpy(),
        output["latitude"].to_numpy(),
        observations["longitude"].to_numpy(),
        observations["latitude"].to_numpy(),
    )
    result = output.copy()
    result[NEARBY_DISTANCE_COLUMN] = distances
    result[HAS_NEARBY_COLUMN] = distances <= NEARBY_TRAINING_THRESHOLD_KM
    report = {
        "threshold_km": float(NEARBY_TRAINING_THRESHOLD_KM),
        "cells_with_nearby_training_data": int(result[HAS_NEARBY_COLUMN].sum()),
        "median_distance_to_training_listing_km": float(pd.Series(distances).median()),
    }
    return gpd.GeoDataFrame(result, geometry="geometry", crs=output.crs), report


def assemble_rural_output(
    grid: gpd.GeoDataFrame,
    cell_classification: pd.DataFrame,
    rural_scored: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    output = grid.merge(
        cell_classification[[CELL_ID_COLUMN, CELL_CLASS_COLUMN]],
        on=CELL_ID_COLUMN,
        how="left",
        validate="one_to_one",
    ).merge(rural_scored, on=CELL_ID_COLUMN, how="left", validate="one_to_one")
    output = (
        output.set_index(CELL_ID_COLUMN)
        .reindex(grid[CELL_ID_COLUMN])
        .reset_index()
    )
    assert output[CELL_CLASS_COLUMN].notna().all(), "a cell left the rural classification on assembly"

    in_rural = output[CELL_CLASS_COLUMN] == RURAL_CELL
    output[IN_RURAL_ANALYSIS_COLUMN] = in_rural

    assert output.loc[in_rural, RURAL_SCORE_COLUMN].notna().all(), (
        "a rural cell carries no rural suitability score"
    )
    assert output.loc[~in_rural, RURAL_SCORE_COLUMN].isna().all(), (
        "a cell outside the rural domain carries a rural suitability score"
    )
    assert output.loc[~in_rural, RURAL_CLASS_COLUMN].isna().all(), (
        "a cell outside the rural domain was classified into a rural suitability class"
    )

    outside_labels = output[CELL_CLASS_COLUMN].map(OUTSIDE_RURAL_ANALYSIS_LABELS)
    output[RURAL_CLASS_COLUMN] = output[RURAL_CLASS_COLUMN].fillna(outside_labels)
    assert output[RURAL_CLASS_COLUMN].notna().all(), (
        "every cell must be either rural-classed or explicitly outside the rural analysis"
    )
    assert output[RURAL_CLASS_COLUMN].isin(
        [*RURAL_CLASS_LABELS, *TRAFFIC_CLASS_LABELS, *OUTSIDE_RURAL_ANALYSIS_LABELS.values()]
    ).all(), "a cell carries an unrecognised rural suitability label"
    assert output[CELL_ID_COLUMN].tolist() == grid[CELL_ID_COLUMN].tolist(), (
        "assembly changed the cell identifiers or their order"
    )

    report = {
        "cells": int(len(output)),
        "cells_in_rural_analysis": int(in_rural.sum()),
        "cells_outside_rural_analysis": int((~in_rural).sum()),
        "cells_urban": int((output[CELL_CLASS_COLUMN] == URBAN_CELL).sum()),
        "cells_unclassified": int((output[CELL_CLASS_COLUMN] == UNCLASSIFIED_CELL).sum()),
        "class_counts": {
            label: int((output[RURAL_CLASS_COLUMN] == label).sum())
            for label in [
                *TRAFFIC_CLASS_LABELS,
                *RURAL_CLASS_LABELS,
                *OUTSIDE_RURAL_ANALYSIS_LABELS.values(),
            ]
        },
    }
    return gpd.GeoDataFrame(output, geometry="geometry", crs=grid.crs), report
