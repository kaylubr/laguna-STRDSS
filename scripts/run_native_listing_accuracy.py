"""Accuracy on each grid's own rural listing cells.

500 m is scored on its 122 listing cells, 1 km on 110, 1.5 km on 90, and
2 km on 71. The class rule is the same at every size: a forest trained on
the other municipalities, with that fold's Youden cutoff.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_grid_listing_accuracy import _featured_grid  # noqa: E402
from run_grid_size_validation import (  # noqa: E402
    SIZES_M,
    _with_barangay_centers,
)
from str_suitability import config
from str_suitability.ingest.load_osm import load_boundaries
from str_suitability.modeling.site_candidate import (
    CELL_ID_COLUMN,
    LAGUNA_ROADS_PATH,
    PRESENCE_LABEL,
    load_places,
)
from str_suitability.pipeline import load_water
from str_suitability.rural.classify import RURAL_CELL
from str_suitability.rural.psa import (
    classify_barangay_polygons,
    load_barangay_classification,
    load_barangay_polygons,
)
from str_suitability.spatial.grid import derive_land_boundary

OUT = config.PROJECT_ROOT / "results" / "grid_size_validation"
EXPECTED = {
    500: {"listed": 122, "empty": 4000},
    1000: {"listed": 110, "empty": 931},
    1500: {"listed": 90, "empty": 380},
    2000: {"listed": 71, "empty": 195},
}


def _counts(featured: pd.DataFrame) -> tuple[int, int]:
    rural = featured["cell_class"].eq(RURAL_CELL) & featured["municipality"].notna()
    listed = int((rural & featured["listings_in_cell"].ge(1)).sum())
    empty = int((rural & featured["listings_in_cell"].eq(0)).sum())
    return listed, empty


def _score_listed(oof: pd.DataFrame, report: dict[str, object], cell_size_m: int) -> pd.DataFrame:
    thresholds = {int(row["fold"]): float(row["youden_threshold"]) for row in report["folds"]}
    listed = oof.loc[oof[PRESENCE_LABEL].eq(1)].copy()
    listed[CELL_ID_COLUMN] = listed[CELL_ID_COLUMN].astype(str)
    listed["fold"] = listed["fold"].astype(int)
    listed["cell_size_m"] = cell_size_m
    listed["actual_label"] = "listed"
    listed["youden_threshold"] = listed["fold"].map(thresholds)
    if listed["youden_threshold"].isna().any() or listed["random_forest_raw"].isna().any():
        raise RuntimeError(f"{cell_size_m} m is missing a held-out score for a listing cell")
    listed["predicted_label"] = np.where(
        listed["random_forest_raw"].ge(listed["youden_threshold"]),
        "listed",
        "empty",
    )
    listed["correct"] = listed["predicted_label"].eq(listed["actual_label"])
    return listed


def _write_report(summary: pd.DataFrame) -> None:
    lines = [
        "# Accuracy on each grid's own listing cells",
        "",
        "Each grid is tested on the rural cells that contain a listing at that size. The actual class is listed. A forest trained on the other municipalities scores the cell. The predicted class is listed when the score is at or above that fold's Youden cutoff. A cell is correct only when the model also says listed. The rule is the same at every size. The denominator is that grid's own listing-cell count.",
        "",
        "| Grid size | Rural cells with a listing | Empty rural cells | Predicted correctly | Predicted incorrectly | Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {int(row.cell_size_m)} m | {int(row.listed_cells)} | {int(row.empty_cells)} | {int(row.correct)} | {int(row.incorrect)} | {row.accuracy_percent:.1f}% |"
        )
    best = summary.sort_values(["accuracy_percent", "cell_size_m"], ascending=[False, True]).iloc[0]
    lines.extend(
        [
            "",
            f"**{int(best.cell_size_m)} m is the most accurate on its own listing cells, at {best.accuracy_percent:.1f}% ({int(best.correct)} of {int(best.listed_cells)}).**",
            "",
        ]
    )
    (OUT / "native_listing_accuracy.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    province, municipalities = load_boundaries(config.BOUNDARY_DIR)
    land = derive_land_boundary(province, load_water(config.PROJECT_ROOT / "assets" / "osm"))
    barangays = _with_barangay_centers(
        classify_barangay_polygons(
            load_barangay_polygons(config.BARANGAY_POLYGONS_PATH),
            load_barangay_classification(config.BARANGAY_CLASSIFICATION_PATH),
        )
    )
    listings = pd.read_parquet(config.INTERIM_DIR / "airroi_targets.parquet")
    places, _place_report = load_places(config.LISTED_TOURIST_PLACES_PATH)
    roads = gpd.read_file(LAGUNA_ROADS_PATH)
    shared = (land, barangays, municipalities, listings, places, roads)
    summaries = []
    details = []
    for cell_size_m in SIZES_M:
        featured, oof, report = _featured_grid(cell_size_m, *shared)
        listed_count, empty_count = _counts(featured)
        expected = EXPECTED[cell_size_m]
        if listed_count != expected["listed"] or empty_count != expected["empty"]:
            raise RuntimeError(
                f"{cell_size_m} m has {listed_count} listing cells and {empty_count} empty cells, "
                f"expected {expected['listed']} and {expected['empty']}"
            )
        scored = _score_listed(oof, report, cell_size_m)
        if len(scored) != listed_count:
            raise RuntimeError(
                f"{cell_size_m} m scored {len(scored)} listing cells, expected {listed_count}"
            )
        correct = int(scored["correct"].sum())
        print(
            f"  {cell_size_m} m {correct} of {listed_count} listing cells, {empty_count} empty",
            flush=True,
        )
        summaries.append(
            {
                "cell_size_m": cell_size_m,
                "listed_cells": listed_count,
                "empty_cells": empty_count,
                "correct": correct,
                "incorrect": listed_count - correct,
                "accuracy_percent": 100.0 * correct / listed_count,
            }
        )
        details.append(scored)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "native_listing_accuracy.csv", index=False)
    pd.concat(details, ignore_index=True)[
        [
            "cell_size_m",
            CELL_ID_COLUMN,
            "municipality",
            "fold",
            "actual_label",
            "random_forest_raw",
            "youden_threshold",
            "predicted_label",
            "correct",
        ]
    ].rename(columns={"random_forest_raw": "score"}).sort_values(
        ["cell_size_m", CELL_ID_COLUMN]
    ).to_csv(OUT / "native_listing_predictions.csv", index=False)
    _write_report(summary)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
