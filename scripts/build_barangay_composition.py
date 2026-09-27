import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from str_suitability import config
from str_suitability.rural.psa import (
    MUNICIPALITY_NAME_COLUMN,
    MUNICIPALITY_PSGC_COLUMN,
    POLYGON_NAME_COLUMN,
    PSGC_COLUMN,
    URBAN_RURAL_COLUMN,
    classify_barangay_polygons,
    load_barangay_classification,
    load_barangay_polygons,
)
from str_suitability.spatial.barangay_composition import (
    BARANGAY_PSGC_COLUMN,
    CELL_ID_COLUMN,
    COVERED_SHARE_COLUMN,
    GRID_AREA_COLUMN,
    INTERSECTION_AREA_COLUMN,
    PERCENTAGE_COLUMN,
    UNCLASSIFIED_LABEL,
    barangay_cell_overlay,
    cell_coverage,
)
from str_suitability.spatial.join import normalize_psgc

GRID_PATH = config.PROCESSED_DIR / "grid_features.parquet"
RURAL_CLASSIFICATION_PATH = config.RURAL_PROCESSED_DIR / "grid_rural_classification.parquet"
COMPOSITION_PATH = config.PROCESSED_DIR / "barangay_grid_composition.parquet"
COVERAGE_PATH = config.PROCESSED_DIR / "grid_coverage.parquet"
SUMMARY_PATH = config.PROCESSED_DIR / "barangay_composition_summary.json"

BARANGAY_ID_COLUMN = "barangay_psgc"

COMPOSITION_COLUMNS = (
    CELL_ID_COLUMN,
    BARANGAY_ID_COLUMN,
    "barangay_name",
    "polygon_name",
    "municipality",
    "municipality_psgc",
    URBAN_RURAL_COLUMN,
    INTERSECTION_AREA_COLUMN,
    GRID_AREA_COLUMN,
    PERCENTAGE_COLUMN,
    COVERED_SHARE_COLUMN,
)

RECONCILED_COLUMNS = (
    ("rural_share_of_covered", "rural_area_share"),
    ("urban_share_of_covered", "urban_area_share"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_barangays() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    classification = load_barangay_classification(config.BARANGAY_CLASSIFICATION_PATH)
    polygons = load_barangay_polygons(config.BARANGAY_POLYGONS_PATH)
    for column in (MUNICIPALITY_NAME_COLUMN, MUNICIPALITY_PSGC_COLUMN):
        assert column in polygons.columns, f"the barangay polygons carry no {column}"
    return classify_barangay_polygons(polygons, classification), classification


def build_composition(
    grid: gpd.GeoDataFrame, barangays: gpd.GeoDataFrame, classification: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    overlay, overlay_report = barangay_cell_overlay(grid, barangays)
    coverage, coverage_report = cell_coverage(overlay, grid)

    names = classification.set_index(PSGC_COLUMN)["barangay_name"]
    composition = overlay.assign(
        barangay_name=overlay[BARANGAY_PSGC_COLUMN].map(names),
        polygon_name=overlay[POLYGON_NAME_COLUMN],
        municipality=overlay[MUNICIPALITY_NAME_COLUMN],
        municipality_psgc=overlay[MUNICIPALITY_PSGC_COLUMN].map(normalize_psgc),
    ).rename(columns={BARANGAY_PSGC_COLUMN: BARANGAY_ID_COLUMN})
    unclassified = composition[URBAN_RURAL_COLUMN] == UNCLASSIFIED_LABEL
    assert composition.loc[unclassified, "barangay_name"].isna().all(), (
        "a barangay polygon with no PSA record carries a PSA name"
    )
    assert composition.loc[~unclassified, "barangay_name"].notna().all(), (
        "a PSA-classified barangay carries no PSA name"
    )
    assert composition["municipality_psgc"].notna().all(), (
        "a barangay polygon resolves to no municipality PSGC code"
    )
    return composition[list(COMPOSITION_COLUMNS)], coverage, overlay_report, coverage_report


def reconcile(coverage: pd.DataFrame, path: Path) -> dict[str, object]:
    existing = pd.read_parquet(path)[[CELL_ID_COLUMN, *[theirs for _, theirs in RECONCILED_COLUMNS]]]
    merged = coverage[
        [CELL_ID_COLUMN, *[mine for mine, _ in RECONCILED_COLUMNS]]
    ].merge(existing, on=CELL_ID_COLUMN, how="left", validate="one_to_one")
    assert len(merged) == len(coverage), "the reconciliation join changed the cell count"

    report = {}
    for mine, theirs in RECONCILED_COLUMNS:
        both = merged[mine].notna() & merged[theirs].notna()
        difference = (merged.loc[both, mine] - merged.loc[both, theirs]).abs()
        report[theirs] = {
            "cells_compared": int(both.sum()),
            "cells_where_one_side_is_missing": int((merged[mine].isna() != merged[theirs].isna()).sum()),
            "max_abs_difference": float(difference.max()) if len(difference) else None,
            "cells_diverging_beyond_1e-9": int((difference > 1e-9).sum()),
        }
    return report


def main() -> None:
    inputs = (
        GRID_PATH,
        config.BARANGAY_POLYGONS_PATH,
        config.BARANGAY_CLASSIFICATION_PATH,
        RURAL_CLASSIFICATION_PATH,
    )
    for path in inputs:
        assert path.exists(), f"{path} is missing"
    before = {str(path): sha256(path) for path in inputs}

    grid = gpd.read_parquet(GRID_PATH)
    barangays, classification = load_barangays()
    composition, coverage, overlay_report, coverage_report = build_composition(
        grid, barangays, classification
    )

    composition.to_parquet(COMPOSITION_PATH, index=False)
    coverage.to_parquet(COVERAGE_PATH, index=False)

    summary = {
        "study_area": config.PROVINCE_NAME,
        "composition_rows": int(len(composition)),
        "overlay": overlay_report,
        "coverage": coverage_report,
        "reconciliation_against_rural_classification": reconcile(coverage, RURAL_CLASSIFICATION_PATH),
        "inputs_unchanged": {str(path): before[str(path)] == sha256(path) for path in inputs},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps(summary, indent=2, default=str))
    print(f"\nartifacts: {COMPOSITION_PATH}\n           {COVERAGE_PATH}\n           {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
