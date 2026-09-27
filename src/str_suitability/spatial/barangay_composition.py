import geopandas as gpd
import numpy as np
import pandas as pd

from str_suitability.config import PROJECTED_CRS
from str_suitability.rural.psa import RURAL, URBAN, URBAN_RURAL_COLUMN
from str_suitability.spatial.grid import assert_one_row_per_cell

CELL_ID_COLUMN = "cell_id"
BARANGAY_PSGC_COLUMN = "psgc"
GRID_AREA_SOURCE_COLUMN = "projected_area_m2"
GRID_AREA_COLUMN = "grid_area_m2"
INTERSECTION_AREA_COLUMN = "intersection_area_m2"
PERCENTAGE_COLUMN = "barangay_percentage"
COVERED_SHARE_COLUMN = "covered_area_share"
UNCLASSIFIED_LABEL = "unclassified"
SLIVER_AREA_M2 = 1.0
AREA_TOLERANCE = 1e-9

COVERAGE_COLUMNS = (
    CELL_ID_COLUMN,
    GRID_AREA_COLUMN,
    "barangay_covered_area_m2",
    "rural_area_m2",
    "urban_area_m2",
    "unclassified_area_m2",
    "uncovered_area_m2",
    "rural_coverage",
    "urban_coverage",
    "unclassified_coverage",
    "uncovered_coverage",
    "rural_share_of_covered",
    "urban_share_of_covered",
    "barangay_count",
    "dominant_barangay_psgc",
    "dominant_barangay_share",
)

SHARE_COLUMNS = (
    "rural_coverage",
    "urban_coverage",
    "unclassified_coverage",
    "uncovered_coverage",
)


def repair_invalid_geometries(
    layer: gpd.GeoDataFrame, label: str
) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    assert len(layer) > 0, f"{label} is empty"
    assert layer.geometry.notna().all(), f"{label} holds a null geometry"

    invalid = ~layer.geometry.is_valid
    count = int(invalid.sum())
    if count == 0:
        return layer, {"layer": label, "invalid_geometries": 0, "repaired_geometries": 0}

    repaired = layer.copy()
    repaired.loc[invalid, "geometry"] = repaired.loc[invalid, "geometry"].make_valid()
    assert repaired.geometry.is_valid.all(), (
        f"{label} still holds invalid geometries after make_valid"
    )
    return repaired, {"layer": label, "invalid_geometries": count, "repaired_geometries": count}


def barangay_cell_overlay(
    grid: gpd.GeoDataFrame, barangays: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    assert_one_row_per_cell(grid)
    assert grid.crs == PROJECTED_CRS, (
        f"the grid must be in {PROJECTED_CRS} to measure area, found {grid.crs}"
    )
    assert GRID_AREA_SOURCE_COLUMN in grid.columns, (
        f"the grid carries no {GRID_AREA_SOURCE_COLUMN} to divide by"
    )
    assert barangays.crs == PROJECTED_CRS, (
        f"the barangay polygons must be in {PROJECTED_CRS}, found {barangays.crs}"
    )
    assert BARANGAY_PSGC_COLUMN in barangays.columns, (
        f"the barangay polygons carry no {BARANGAY_PSGC_COLUMN} to identify them by"
    )
    assert URBAN_RURAL_COLUMN in barangays.columns, (
        f"the barangay polygons carry no {URBAN_RURAL_COLUMN} field"
    )

    grid, grid_repair = repair_invalid_geometries(grid, "grid")
    barangays, barangay_repair = repair_invalid_geometries(barangays, "barangays")

    carried = [column for column in barangays.columns if column != "geometry"]
    overlay = gpd.overlay(
        grid[[CELL_ID_COLUMN, GRID_AREA_SOURCE_COLUMN, "geometry"]],
        barangays[[*carried, "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    assert len(overlay) > 0, "no barangay intersects the grid"

    overlay = overlay.rename(columns={GRID_AREA_SOURCE_COLUMN: GRID_AREA_COLUMN})
    overlay[URBAN_RURAL_COLUMN] = overlay[URBAN_RURAL_COLUMN].fillna(UNCLASSIFIED_LABEL)
    overlay[INTERSECTION_AREA_COLUMN] = overlay.geometry.area
    overlay = overlay[overlay[INTERSECTION_AREA_COLUMN] > 0].reset_index(drop=True)
    assert (overlay[GRID_AREA_COLUMN] > 0).all(), "a grid cell carries a non-positive area"

    overlay[PERCENTAGE_COLUMN] = (
        100.0 * overlay[INTERSECTION_AREA_COLUMN] / overlay[GRID_AREA_COLUMN]
    )
    covered_in_cell = overlay.groupby(CELL_ID_COLUMN)[INTERSECTION_AREA_COLUMN].transform("sum")
    overlay[COVERED_SHARE_COLUMN] = overlay[INTERSECTION_AREA_COLUMN] / covered_in_cell

    assert (overlay[PERCENTAGE_COLUMN] <= 100.0 + AREA_TOLERANCE).all(), (
        "a barangay covers more than the cell it sits in"
    )

    totals = overlay.groupby(CELL_ID_COLUMN).agg(
        covered_area_m2=(INTERSECTION_AREA_COLUMN, "sum"),
        grid_area_m2=(GRID_AREA_COLUMN, "first"),
        barangays=(BARANGAY_PSGC_COLUMN, "size"),
    )
    overlapping = totals[totals["covered_area_m2"] > totals["grid_area_m2"] * (1.0 + AREA_TOLERANCE)]
    assert len(overlapping) == 0, (
        f"{len(overlapping)} cells are covered by barangay polygons that overlap each other"
    )

    report = {
        "cells": int(len(grid)),
        "intersections": int(len(overlay)),
        "cells_with_barangay": int(len(totals)),
        "cells_without_barangay": int(len(grid) - len(totals)),
        "cells_with_multiple_barangays": int((totals["barangays"] > 1).sum()),
        "max_barangays_in_a_cell": int(totals["barangays"].max()),
        "slivers_below_1m2": int((overlay[INTERSECTION_AREA_COLUMN] < SLIVER_AREA_M2).sum()),
        "min_barangay_percentage": float(overlay[PERCENTAGE_COLUMN].min()),
        "max_barangay_percentage": float(overlay[PERCENTAGE_COLUMN].max()),
        "grid_area_m2": float(totals["grid_area_m2"].sum()),
        "grid_area_uncovered_m2": float(
            (totals["grid_area_m2"] - totals["covered_area_m2"]).sum()
        ),
        "geometry_repairs": [grid_repair, barangay_repair],
    }
    return gpd.GeoDataFrame(overlay, geometry="geometry", crs=PROJECTED_CRS), report


def cell_coverage(
    overlay: gpd.GeoDataFrame, grid: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    assert_one_row_per_cell(grid)
    assert GRID_AREA_SOURCE_COLUMN in grid.columns, (
        f"the grid carries no {GRID_AREA_SOURCE_COLUMN} to divide by"
    )

    labels = overlay[URBAN_RURAL_COLUMN]
    assert labels.notna().all(), "an overlay row carries no urban_rural label"
    assert set(labels) <= {RURAL, URBAN, UNCLASSIFIED_LABEL}, (
        f"the overlay carries an unrecognised label, found {sorted(set(labels) - {RURAL, URBAN, UNCLASSIFIED_LABEL})}"
    )
    by_label = pd.DataFrame(
        {
            CELL_ID_COLUMN: overlay[CELL_ID_COLUMN],
            "label": labels,
            INTERSECTION_AREA_COLUMN: overlay[INTERSECTION_AREA_COLUMN],
        }
    ).pivot_table(
        index=CELL_ID_COLUMN, columns="label", values=INTERSECTION_AREA_COLUMN, aggfunc="sum"
    )

    coverage = pd.DataFrame({CELL_ID_COLUMN: grid[CELL_ID_COLUMN].to_numpy()})
    coverage[GRID_AREA_COLUMN] = grid[GRID_AREA_SOURCE_COLUMN].to_numpy()
    for label, column in (
        (RURAL, "rural_area_m2"),
        (URBAN, "urban_area_m2"),
        (UNCLASSIFIED_LABEL, "unclassified_area_m2"),
    ):
        areas = by_label[label] if label in by_label.columns else pd.Series(dtype=float)
        coverage[column] = coverage[CELL_ID_COLUMN].map(areas).fillna(0.0).to_numpy()

    coverage["barangay_covered_area_m2"] = (
        coverage["rural_area_m2"]
        + coverage["urban_area_m2"]
        + coverage["unclassified_area_m2"]
    )
    coverage["uncovered_area_m2"] = (
        coverage[GRID_AREA_COLUMN] - coverage["barangay_covered_area_m2"]
    )
    assert (coverage["uncovered_area_m2"] >= -AREA_TOLERANCE * coverage[GRID_AREA_COLUMN]).all(), (
        "barangay intersections cover more than the cell they sit in"
    )
    coverage["uncovered_area_m2"] = coverage["uncovered_area_m2"].clip(lower=0.0)

    for column, area in (
        ("rural_coverage", "rural_area_m2"),
        ("urban_coverage", "urban_area_m2"),
        ("unclassified_coverage", "unclassified_area_m2"),
        ("uncovered_coverage", "uncovered_area_m2"),
    ):
        coverage[column] = coverage[area] / coverage[GRID_AREA_COLUMN]

    assert np.allclose(coverage[list(SHARE_COLUMNS)].sum(axis=1).to_numpy(), 1.0), (
        "the coverage shares of a cell do not sum to one"
    )

    covered = coverage["barangay_covered_area_m2"]
    divided_by = covered.where(covered > 0)
    coverage["rural_share_of_covered"] = (coverage["rural_area_m2"] / divided_by).to_numpy()
    coverage["urban_share_of_covered"] = (coverage["urban_area_m2"] / divided_by).to_numpy()
    covered_shares = coverage["rural_share_of_covered"].fillna(0.0) + coverage[
        "urban_share_of_covered"
    ].fillna(0.0)
    assert (covered_shares <= 1.0 + AREA_TOLERANCE).all(), (
        "the rural and urban shares of the covered area sum above one"
    )

    counts = overlay.groupby(CELL_ID_COLUMN)[BARANGAY_PSGC_COLUMN].size()
    coverage["barangay_count"] = (
        coverage[CELL_ID_COLUMN].map(counts).fillna(0).astype(int).to_numpy()
    )
    without = coverage["barangay_count"] == 0
    assert (coverage.loc[without, "uncovered_coverage"] == 1.0).all(), (
        "a cell no barangay overlaps must be entirely uncovered"
    )
    assert coverage.loc[without, "rural_share_of_covered"].isna().all(), (
        "a cell no barangay overlaps must carry no covered-area share"
    )
    assert (coverage.loc[~without, "uncovered_coverage"] < 1.0).all(), (
        "a cell a barangay overlaps must be partly covered"
    )

    dominant = (
        overlay.sort_values(
            [CELL_ID_COLUMN, INTERSECTION_AREA_COLUMN, BARANGAY_PSGC_COLUMN],
            ascending=[True, False, True],
        )
        .drop_duplicates(CELL_ID_COLUMN)
        .set_index(CELL_ID_COLUMN)
    )
    coverage["dominant_barangay_psgc"] = (
        coverage[CELL_ID_COLUMN].map(dominant[BARANGAY_PSGC_COLUMN]).to_numpy()
    )
    coverage["dominant_barangay_share"] = (
        coverage[CELL_ID_COLUMN].map(dominant[PERCENTAGE_COLUMN] / 100.0).to_numpy()
    )
    assert coverage.loc[without, "dominant_barangay_psgc"].isna().all(), (
        "a cell no barangay overlaps must carry no dominant barangay"
    )
    assert coverage.loc[~without, "dominant_barangay_psgc"].notna().all(), (
        "a cell a barangay overlaps must carry a dominant barangay"
    )

    coverage = coverage[list(COVERAGE_COLUMNS)]
    report = {
        "cells": int(len(coverage)),
        "cells_with_barangay": int((~without).sum()),
        "cells_without_barangay": int(without.sum()),
        "cells_with_multiple_barangays": int((coverage["barangay_count"] > 1).sum()),
        "max_barangays_in_a_cell": int(coverage["barangay_count"].max()),
        "cells_wholly_covered": int((coverage["uncovered_coverage"] <= AREA_TOLERANCE).sum()),
        "cells_with_a_coverage_gap": int((coverage["uncovered_coverage"] > AREA_TOLERANCE).sum()),
        "cells_over_half_rural_by_grid_area": int((coverage["rural_coverage"] >= 0.5).sum()),
        "uncovered_area_m2": float(coverage["uncovered_area_m2"].sum()),
        "grid_area_m2": float(coverage[GRID_AREA_COLUMN].sum()),
    }
    return coverage, report
