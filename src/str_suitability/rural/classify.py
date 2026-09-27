import numpy as np
import pandas as pd
import geopandas as gpd

from str_suitability.config import GEOGRAPHIC_CRS, RURAL_AREA_SHARE_THRESHOLD
from str_suitability.rural.psa import (
    MUNICIPALITY_NAME_COLUMN,
    POLYGON_NAME_COLUMN,
    PSGC_COLUMN,
    RURAL,
    URBAN,
    URBAN_RURAL_COLUMN,
)
from str_suitability.spatial.grid import assert_one_row_per_cell

CELL_ID_COLUMN = "cell_id"
CELL_CLASS_COLUMN = "cell_class"
RURAL_CELL = "rural"
URBAN_CELL = "urban"
UNCLASSIFIED_CELL = "unclassified"
CELL_CLASSES = (RURAL_CELL, URBAN_CELL, UNCLASSIFIED_CELL)

AREA_SHARE_TOLERANCE = 1e-9

LISTING_ID_COLUMN = "listing_id"
LISTING_CLASSIFICATION_COLUMNS = [
    LISTING_ID_COLUMN,
    "matched_barangay_psgc",
    "matched_barangay_name",
    "matched_municipality",
    URBAN_RURAL_COLUMN,
    "match_status",
]

CELL_CLASSIFICATION_COLUMNS = [
    CELL_ID_COLUMN,
    "rural_area_share",
    "urban_area_share",
    "dominant_barangay",
    "dominant_barangay_psgc",
    "dominant_barangay_urban_rural",
    CELL_CLASS_COLUMN,
]


def select_cells_with_class(
    grid: gpd.GeoDataFrame, cell_classification: pd.DataFrame, cell_class: str
) -> gpd.GeoDataFrame:
    assert cell_class in CELL_CLASSES, f"unknown cell class: {cell_class}"
    assert cell_classification[CELL_ID_COLUMN].is_unique, "cell classification repeats a cell"
    selected = cell_classification.loc[
        cell_classification[CELL_CLASS_COLUMN] == cell_class, CELL_ID_COLUMN
    ]
    assert len(selected) > 0, f"no cell is classified {cell_class}"

    subset = grid[grid[CELL_ID_COLUMN].isin(set(selected))].copy()
    assert len(subset) == len(selected), "the grid does not cover every cell of this class"
    return gpd.GeoDataFrame(subset, geometry="geometry", crs=grid.crs)


def classify_grid_cells(
    grid: gpd.GeoDataFrame,
    barangays: gpd.GeoDataFrame,
    threshold: float = RURAL_AREA_SHARE_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, object]]:
    assert_one_row_per_cell(grid)
    assert barangays[PSGC_COLUMN].is_unique, "barangay polygons repeat a PSGC code"
    assert URBAN_RURAL_COLUMN in barangays.columns, (
        f"barangay polygons carry no {URBAN_RURAL_COLUMN} field"
    )

    overlay = gpd.overlay(
        grid[[CELL_ID_COLUMN, "geometry"]],
        barangays[[PSGC_COLUMN, POLYGON_NAME_COLUMN, URBAN_RURAL_COLUMN, "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    overlay["overlap_area_m2"] = overlay.geometry.area
    assert (overlay["overlap_area_m2"] > 0).all(), "a grid cell intersects a barangay with no area"

    rural_area = _area_by_cell(overlay, RURAL)
    urban_area = _area_by_cell(overlay, URBAN)
    covered_area = overlay.groupby(CELL_ID_COLUMN)["overlap_area_m2"].sum()

    ids = grid[CELL_ID_COLUMN].to_numpy()
    covered = covered_area.reindex(ids).to_numpy()
    rural_share = _share(rural_area, covered, ids)
    urban_share = _share(urban_area, covered, ids)

    uncovered = np.isnan(covered)
    cell_class = np.where(
        uncovered,
        UNCLASSIFIED_CELL,
        np.where(rural_share >= threshold, RURAL_CELL, URBAN_CELL),
    )

    dominant = (
        overlay.sort_values(
            [CELL_ID_COLUMN, "overlap_area_m2", PSGC_COLUMN], ascending=[True, False, True]
        )
        .drop_duplicates(CELL_ID_COLUMN)
        .set_index(CELL_ID_COLUMN)
    )

    classification = pd.DataFrame(
        {
            CELL_ID_COLUMN: ids,
            "rural_area_share": rural_share,
            "urban_area_share": urban_share,
            "dominant_barangay": dominant[POLYGON_NAME_COLUMN].reindex(ids).to_numpy(),
            "dominant_barangay_psgc": dominant[PSGC_COLUMN].reindex(ids).to_numpy(),
            "dominant_barangay_urban_rural": dominant[URBAN_RURAL_COLUMN].reindex(ids).to_numpy(),
            CELL_CLASS_COLUMN: cell_class,
        }
    )
    report = _cell_classification_report(grid, classification, uncovered, threshold)
    return classification, report


def _area_by_cell(overlay: gpd.GeoDataFrame, urban_rural: str) -> pd.Series:
    return (
        overlay.loc[overlay[URBAN_RURAL_COLUMN] == urban_rural]
        .groupby(CELL_ID_COLUMN)["overlap_area_m2"]
        .sum()
    )


def _share(area: pd.Series, covered: np.ndarray, ids: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        share = area.reindex(ids).fillna(0.0).to_numpy() / covered
    return np.where(np.isnan(covered), np.nan, share)


def _cell_classification_report(
    grid: gpd.GeoDataFrame,
    classification: pd.DataFrame,
    uncovered: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    assert len(classification) == len(grid), "grid classification lost or gained a cell"
    assert classification[CELL_ID_COLUMN].tolist() == grid[CELL_ID_COLUMN].tolist(), (
        "grid classification changed the cell identifiers or their order"
    )
    assert classification[CELL_CLASS_COLUMN].isin(CELL_CLASSES).all(), (
        "every grid cell must receive exactly one of rural, urban or unclassified"
    )

    rural_share = classification["rural_area_share"]
    urban_share = classification["urban_area_share"]
    covered = rural_share.notna()
    assert covered.equals(urban_share.notna()), "rural and urban coverage disagree on which cells are covered"
    for name, share in (("rural", rural_share), ("urban", urban_share)):
        values = share[covered]
        assert values.between(0.0, 1.0).all(), f"{name} area share left the unit interval"
    assert ((rural_share[covered] + urban_share[covered]) <= 1.0 + AREA_SHARE_TOLERANCE).all(), (
        "rural and urban area shares sum above one within a cell"
    )

    counts = classification[CELL_CLASS_COLUMN].value_counts().to_dict()
    assert counts.get(UNCLASSIFIED_CELL, 0) == int(uncovered.sum()), (
        "unclassified cells must be exactly the cells no PSA barangay overlaps"
    )
    return {
        "cells": int(len(classification)),
        "rural_cells": int(counts.get(RURAL_CELL, 0)),
        "urban_cells": int(counts.get(URBAN_CELL, 0)),
        "unclassified_cells": int(counts.get(UNCLASSIFIED_CELL, 0)),
        "cells_with_dominant_urban_barangay": int(
            (classification["dominant_barangay_urban_rural"] == URBAN).sum()
        ),
        "area_share_threshold": float(threshold),
    }


def classify_listings_by_barangay(
    listings: pd.DataFrame, barangays: gpd.GeoDataFrame
) -> pd.DataFrame:
    assert LISTING_ID_COLUMN in listings.columns, "listings have no listing_id"
    assert {"longitude", "latitude"} <= set(listings.columns), "listings have no coordinates"
    points = gpd.GeoDataFrame(
        {LISTING_ID_COLUMN: listings[LISTING_ID_COLUMN].astype(str).to_numpy()},
        geometry=gpd.points_from_xy(listings["longitude"], listings["latitude"]),
        crs=GEOGRAPHIC_CRS,
    )
    joined = gpd.sjoin(
        points,
        barangays.to_crs(GEOGRAPHIC_CRS),
        how="left",
        predicate="within",
    )
    joined = joined.drop_duplicates(LISTING_ID_COLUMN, keep="first").set_index(LISTING_ID_COLUMN)
    joined = joined.reindex(points[LISTING_ID_COLUMN])

    urban_rural = joined[URBAN_RURAL_COLUMN]
    status = np.where(
        urban_rural.eq(RURAL),
        "matched_rural",
        np.where(urban_rural.eq(URBAN), "matched_urban", "unmatched"),
    )
    classified = pd.DataFrame(
        {
            LISTING_ID_COLUMN: points[LISTING_ID_COLUMN].to_numpy(),
            "matched_barangay_psgc": joined[PSGC_COLUMN].to_numpy(),
            "matched_barangay_name": joined[POLYGON_NAME_COLUMN].to_numpy(),
            "matched_municipality": joined[MUNICIPALITY_NAME_COLUMN].to_numpy()
            if MUNICIPALITY_NAME_COLUMN in joined.columns
            else None,
            URBAN_RURAL_COLUMN: urban_rural.to_numpy(),
            "match_status": status,
        }
    )
    assert classified[LISTING_ID_COLUMN].is_unique, "listing classification repeated a listing"
    assert len(classified) == len(listings), "listing classification dropped a listing"
    return classified


def load_listing_classification(path) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(
        path,
        dtype={
            LISTING_ID_COLUMN: str,
            "matched_barangay_psgc": str,
            "matched_barangay_name": str,
            "matched_municipality": str,
            URBAN_RURAL_COLUMN: str,
            "match_status": str,
        },
    )
    frame[LISTING_ID_COLUMN] = frame[LISTING_ID_COLUMN].astype(str)
    assert frame[LISTING_ID_COLUMN].is_unique, "the listing classification repeats a listing_id"

    labels = set(frame[URBAN_RURAL_COLUMN].dropna())
    assert labels <= {RURAL, URBAN}, (
        f"listing classification must be {RURAL}, {URBAN} or unmatched, found {sorted(labels - {RURAL, URBAN})}"
    )

    rural = int((frame[URBAN_RURAL_COLUMN] == RURAL).sum())
    urban = int((frame[URBAN_RURAL_COLUMN] == URBAN).sum())
    unmatched = int(frame[URBAN_RURAL_COLUMN].isna().sum())
    assert rural + urban + unmatched == len(frame), "listing classification does not partition the listings"

    report = {
        "listings": int(len(frame)),
        "rural_listings": rural,
        "urban_listings": urban,
        "unmatched_listings": unmatched,
    }
    return frame[LISTING_CLASSIFICATION_COLUMNS], report


def attach_listing_classification(
    listings: pd.DataFrame, classification: pd.DataFrame
) -> pd.DataFrame:
    assert URBAN_RURAL_COLUMN not in listings.columns, (
        f"listings already carry {URBAN_RURAL_COLUMN}; the PSA classification must remain the only source"
    )
    assert classification[LISTING_ID_COLUMN].is_unique, "the listing classification repeats a listing_id"

    attached = listings.merge(
        classification,
        on=LISTING_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    assert len(attached) == len(listings), "attaching the listing classification changed the row count"
    assert len(classification) == len(listings), (
        "the listing classification does not cover exactly the listings it is joined to"
    )
    return attached


def rural_population_levels(
    targets: pd.DataFrame, observations: pd.DataFrame
) -> dict[str, object]:
    assert URBAN_RURAL_COLUMN in targets.columns, (
        f"the target table carries no {URBAN_RURAL_COLUMN} labels to select rural listings from"
    )
    assert "in_training_population" in targets.columns, (
        "the target table carries no in_training_population flag"
    )
    assert CELL_ID_COLUMN in observations.columns, "the fitted observations carry no cell identifier"

    rural = targets.loc[targets[URBAN_RURAL_COLUMN] == RURAL]
    active = rural.loc[rural["in_training_population"]]
    fitted = active.merge(
        observations[[LISTING_ID_COLUMN, CELL_ID_COLUMN]], on=LISTING_ID_COLUMN, how="inner"
    )

    assert len(active) <= len(rural), "more active rural listings than rural listings"
    assert len(fitted) <= len(active), "more fitted rural listings than active rural listings"

    levels = {
        "all_rural": _level_summary(rural),
        "active_rural": _level_summary(active),
        "fitted_rural": {
            "listings": int(len(fitted)),
            "municipalities": int(fitted["municipality"].nunique()),
            "cells": int(fitted[CELL_ID_COLUMN].nunique()),
        },
    }
    return levels


def _level_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "listings": int(len(frame)),
        "municipalities": int(frame["municipality"].nunique()),
        "cells": None,
    }
