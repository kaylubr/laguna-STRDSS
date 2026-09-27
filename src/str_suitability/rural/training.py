import pandas as pd

from str_suitability.pipeline import build_training_frame
from str_suitability.rural.classify import (
    CELL_CLASS_COLUMN,
    CELL_ID_COLUMN,
    LISTING_ID_COLUMN,
    RURAL,
    RURAL_CELL,
    URBAN_CELL,
    URBAN_RURAL_COLUMN,
    attach_listing_classification,
)

CONCENTRATION_CELL_COUNT = 10
TOP_MUNICIPALITY_COUNT = 3


def select_rural_targets(targets: pd.DataFrame, listing_classification: pd.DataFrame) -> pd.DataFrame:
    labelled = attach_listing_classification(targets, listing_classification)
    rural = labelled.loc[labelled[URBAN_RURAL_COLUMN] == RURAL]
    assert len(rural) > 0, "no listing is classified rural, so there is no rural training population"
    return rural


def build_rural_training_frame(
    targets: pd.DataFrame,
    grid: pd.DataFrame,
    listing_classification: pd.DataFrame,
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rural_targets = select_rural_targets(targets, listing_classification)
    merged, report = build_training_frame(rural_targets, grid)

    assert merged[LISTING_ID_COLUMN].is_unique, "the rural training frame repeats a listing"
    _assert_only_rural_listings(merged, listing_classification)
    fitted_rural = _fitted_rural_ids(rural_targets, observations)
    assert set(merged[LISTING_ID_COLUMN]) == fitted_rural, (
        "the rural training observations are not exactly the rural subset of the fitted training population"
    )

    report.update(
        {
            "all_rural_listings": int(len(rural_targets)),
            "active_rural_listings": int(rural_targets["in_training_population"].sum()),
            "fitted_rural_listings": int(len(merged)),
            **rural_concentration(merged),
        }
    )
    return merged, report


def _assert_only_rural_listings(
    merged: pd.DataFrame, listing_classification: pd.DataFrame
) -> None:
    labels = merged[[LISTING_ID_COLUMN]].merge(
        listing_classification[[LISTING_ID_COLUMN, URBAN_RURAL_COLUMN]],
        on=LISTING_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    assert labels[URBAN_RURAL_COLUMN].notna().all(), "a rural training listing has no PSA classification"
    assert (labels[URBAN_RURAL_COLUMN] == RURAL).all(), (
        "a listing that is not PSA-rural entered the rural training population"
    )


def _fitted_rural_ids(rural_targets: pd.DataFrame, observations: pd.DataFrame) -> set[str]:
    fitted = rural_targets.merge(
        observations[[LISTING_ID_COLUMN, CELL_ID_COLUMN]], on=LISTING_ID_COLUMN, how="inner"
    )
    return set(fitted[LISTING_ID_COLUMN])


def rural_concentration(merged: pd.DataFrame) -> dict[str, object]:
    per_municipality = merged["municipality"].value_counts()
    per_cell = merged[CELL_ID_COLUMN].value_counts()
    top_municipalities = per_municipality.head(TOP_MUNICIPALITY_COUNT)

    return {
        "municipalities_with_training_listings": int(per_municipality.size),
        "cells_with_training_listings": int(per_cell.size),
        "median_listings_per_municipality": float(per_municipality.median()),
        "max_listings_per_municipality": int(per_municipality.max()),
        "cells_holding_one_listing": int((per_cell == 1).sum()),
        "max_listings_in_a_cell": int(per_cell.max()),
        "share_in_busiest_cell": float(per_cell.iloc[0] / len(merged)),
        "share_in_busiest_cells": float(per_cell.head(CONCENTRATION_CELL_COUNT).sum() / len(merged)),
        "top_municipalities": {name: int(count) for name, count in top_municipalities.items()},
        "share_in_top_municipalities": float(top_municipalities.sum() / len(merged)),
        "listings_per_municipality": {name: int(count) for name, count in per_municipality.items()},
    }


def rural_training_cell_classes(
    merged: pd.DataFrame, cell_classification: pd.DataFrame, observations: pd.DataFrame
) -> dict[str, object]:
    cell_classes = cell_classification.set_index(CELL_ID_COLUMN)[CELL_CLASS_COLUMN]
    rural_cells = set(
        cell_classification.loc[cell_classification[CELL_CLASS_COLUMN] == RURAL_CELL, CELL_ID_COLUMN]
    )
    labels = merged[CELL_ID_COLUMN].map(cell_classes)
    assert labels.notna().all(), "a rural training listing sits in a cell with no rural classification"

    rural_with_rural_listings = set(
        merged.loc[labels == RURAL_CELL, CELL_ID_COLUMN]
    )
    rural_with_any_training_listing = set(observations[CELL_ID_COLUMN]) & rural_cells

    return {
        "listings_in_rural_cells": int((labels == RURAL_CELL).sum()),
        "listings_in_urban_cells": int((labels == URBAN_CELL).sum()),
        "rural_cells_with_fitted_rural_listings": int(len(rural_with_rural_listings)),
        "rural_cells_with_any_training_listing": int(len(rural_with_any_training_listing)),
        "rural_cells_without_any_training_listing": int(
            len(rural_cells) - len(rural_with_any_training_listing)
        ),
    }
