"""How often each grid size calls the 110 known listing cells "listed".

The 110 locations are the rural 1 km cells that already contain an active
listing. One listing inside each cell is the test point. At every grid size
the same point is scored by the cell that contains it. The forest is trained
on the other municipalities, and the class uses that fold's Youden cutoff.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_grid_size_validation import (  # noqa: E402
    SIZES_M,
    _listing_counts,
    _municipality,
    _with_barangay_centers,
)
from str_suitability import config
from str_suitability.config import GEOGRAPHIC_CRS
from str_suitability.features.context import add_poblacion_distance
from str_suitability.features.road_distance import nearest_mapped_road_distance
from str_suitability.ingest.load_osm import load_boundaries
from str_suitability.modeling.site_candidate import (
    CELL_ID_COLUMN,
    LAGUNA_ROADS_PATH,
    PRESENCE_LABEL,
    build_feature_frame,
    build_training_sample,
    evaluate_forest,
    load_places,
)
from str_suitability.pipeline import load_water
from str_suitability.rural.classify import CELL_CLASS_COLUMN, RURAL_CELL, classify_grid_cells
from str_suitability.rural.psa import (
    classify_barangay_polygons,
    load_barangay_classification,
    load_barangay_polygons,
)
from str_suitability.spatial.grid import build_grid, derive_land_boundary
from str_suitability.spatial.join import assign_listings_to_cells

OUT = config.PROJECT_ROOT / "results" / "grid_size_validation"
EXPECTED_LISTED = 110


def _featured_grid(
    cell_size_m: int,
    land,
    barangays: gpd.GeoDataFrame,
    municipalities: gpd.GeoDataFrame,
    listings: pd.DataFrame,
    places: pd.DataFrame,
    roads: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, object]]:
    print(f"building {cell_size_m} m grid", flush=True)
    grid = build_grid(land, cell_size_m)
    grid[CELL_ID_COLUMN] = grid[CELL_ID_COLUMN].astype(str)
    classification, _report = classify_grid_cells(grid, barangays)
    grid = grid.merge(
        classification[[CELL_ID_COLUMN, CELL_CLASS_COLUMN]],
        on=CELL_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    grid["municipality"] = _municipality(grid, municipalities).to_numpy()
    grid["listings_in_cell"] = _listing_counts(grid, listings).to_numpy()
    grid, _poblacion = add_poblacion_distance(grid, barangays)
    distances = nearest_mapped_road_distance(grid, roads)
    distances[CELL_ID_COLUMN] = distances[CELL_ID_COLUMN].astype(str)
    grid = grid.merge(
        distances[[CELL_ID_COLUMN, "road_distance_km"]],
        on=CELL_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    featured = build_feature_frame(grid, places, roads)
    sample = build_training_sample(featured, negative_seed=42)
    oof, report = evaluate_forest(sample)
    return featured, oof, report


def _reference_points(grid_1km: gpd.GeoDataFrame, listings: pd.DataFrame) -> gpd.GeoDataFrame:
    rural = grid_1km["cell_class"].eq(RURAL_CELL)
    known = grid_1km.loc[
        rural & grid_1km["listings_in_cell"].ge(1) & grid_1km["municipality"].notna()
    ]
    if len(known) != EXPECTED_LISTED:
        raise RuntimeError(f"the 1 km grid has {len(known)} listing cells, not {EXPECTED_LISTED}")
    saved = gpd.read_parquet(config.RURAL_PROCESSED_DIR / "grid_suitability.parquet")
    saved[CELL_ID_COLUMN] = saved[CELL_ID_COLUMN].astype(str)
    saved_listed = set(
        saved.loc[
            saved["cell_class"].eq(RURAL_CELL)
            & saved["listings_in_cell"].ge(1)
            & saved["municipality"].notna(),
            CELL_ID_COLUMN,
        ]
    )
    if set(known[CELL_ID_COLUMN]) != saved_listed:
        raise RuntimeError("the rebuilt 1 km listing cells are not the saved 110 cells")

    joined, _report = assign_listings_to_cells(listings, grid_1km)
    active = joined.loc[joined["in_training_population"].astype(bool)].dropna(subset=[CELL_ID_COLUMN])
    active[CELL_ID_COLUMN] = active[CELL_ID_COLUMN].astype(str)
    active = active.loc[active[CELL_ID_COLUMN].isin(set(known[CELL_ID_COLUMN]))]
    active = active.sort_values([CELL_ID_COLUMN, "longitude", "latitude"])
    one = active.drop_duplicates(CELL_ID_COLUMN, keep="first")
    if len(one) != EXPECTED_LISTED:
        raise RuntimeError(f"only {len(one)} of the 110 cells contained an active listing point")
    points = gpd.GeoDataFrame(
        {"reference_cell_id": one[CELL_ID_COLUMN].to_numpy()},
        geometry=gpd.points_from_xy(one["longitude"], one["latitude"]),
        crs=GEOGRAPHIC_CRS,
    )
    return points


def _cell_for_each_point(points: gpd.GeoDataFrame, grid: gpd.GeoDataFrame) -> pd.DataFrame:
    cells = grid[[CELL_ID_COLUMN, "geometry"]].copy()
    located = gpd.sjoin(
        points.to_crs(grid.crs),
        cells,
        how="left",
        predicate="within",
    )
    located = located.sort_values(["reference_cell_id", CELL_ID_COLUMN]).drop_duplicates(
        "reference_cell_id", keep="first"
    )
    located["contained"] = located[CELL_ID_COLUMN].notna()
    missing = ~located["contained"]
    if missing.any():
        nearest = gpd.sjoin_nearest(
            points.loc[points["reference_cell_id"].isin(located.loc[missing, "reference_cell_id"])].to_crs(grid.crs),
            cells,
            how="left",
            distance_col="metres_to_cell",
        )
        nearest = nearest.sort_values(["reference_cell_id", "metres_to_cell"]).drop_duplicates(
            "reference_cell_id", keep="first"
        )
        on_boundary = nearest["metres_to_cell"].le(1)
        boundary_ids = nearest.loc[on_boundary].set_index("reference_cell_id")[CELL_ID_COLUMN]
        mapped = located.loc[missing, "reference_cell_id"].map(boundary_ids)
        located[CELL_ID_COLUMN] = located[CELL_ID_COLUMN].astype(object)
        located.loc[missing, CELL_ID_COLUMN] = [
            None if pd.isna(value) else str(value) for value in mapped
        ]
        located.loc[missing, "contained"] = located.loc[missing, CELL_ID_COLUMN].notna()
    if len(located) != EXPECTED_LISTED:
        raise RuntimeError("every one of the 110 listing points must be tested once")
    return located[["reference_cell_id", CELL_ID_COLUMN, "contained"]].reset_index(drop=True)


def _score_known_cells(
    cell_size_m: int,
    points: gpd.GeoDataFrame,
    featured: gpd.GeoDataFrame,
    oof: pd.DataFrame,
    report: dict[str, object],
) -> pd.DataFrame:
    located = _cell_for_each_point(points, featured)
    thresholds = {int(row["fold"]): float(row["youden_threshold"]) for row in report["folds"]}
    scores = oof.copy()
    scores[CELL_ID_COLUMN] = scores[CELL_ID_COLUMN].astype(str)
    scores["fold"] = scores["fold"].astype(int)
    lookup = featured[[CELL_ID_COLUMN, "municipality", "cell_class", "listings_in_cell"]].copy()
    lookup[CELL_ID_COLUMN] = lookup[CELL_ID_COLUMN].astype(str)
    inside = located.loc[located["contained"]].merge(lookup, on=CELL_ID_COLUMN, how="left", validate="many_to_one")
    inside = inside.merge(
        scores[[CELL_ID_COLUMN, "fold", "random_forest_raw", PRESENCE_LABEL]],
        on=CELL_ID_COLUMN,
        how="left",
        validate="many_to_one",
    )
    outside_rows = located.loc[~located["contained"]].copy()
    checked = pd.concat([inside, outside_rows], ignore_index=True)
    checked["cell_size_m"] = cell_size_m
    checked["actual_label"] = "listed"
    checked["youden_threshold"] = checked["fold"].map(thresholds)
    contained = checked["contained"].astype(bool)
    scored = contained & checked["random_forest_raw"].notna() & checked["listings_in_cell"].ge(1)
    checked["predicted_label"] = "empty"
    checked.loc[scored, "predicted_label"] = np.where(
        checked.loc[scored, "random_forest_raw"].ge(checked.loc[scored, "youden_threshold"]),
        "listed",
        "empty",
    )
    checked["correct"] = checked["predicted_label"].eq(checked["actual_label"])
    checked["note"] = [
        _unscored_note(row) for row in checked.itertuples(index=False)
    ]
    outside = int((~contained).sum())
    if outside:
        print(f"  {outside} listing points sit in a cell this grid does not keep", flush=True)
    notes = checked.loc[checked["note"].ne(""), "note"].value_counts()
    if len(notes):
        print(f"  not called listed by the rural model: {notes.to_dict()}", flush=True)
    return checked


def _unscored_note(row) -> str:
    if row.predicted_label == "listed" or row.correct:
        return ""
    if not bool(row.contained):
        return "the grid drops this cell"
    cell_class = getattr(row, "cell_class", None)
    if cell_class != RURAL_CELL:
        return f"the cell is {cell_class}"
    listings = getattr(row, "listings_in_cell", None)
    if pd.isna(listings) or listings < 1:
        return "the joined cell has no listing"
    if pd.isna(getattr(row, "random_forest_raw", None)):
        return "the cell is outside the held-out sample"
    return ""


def _write_report(summary: pd.DataFrame, details: pd.DataFrame) -> None:
    lines = [
        "# Accuracy on the 110 existing listing cells",
        "",
        "The same 110 rural 1 km cells are tested at every grid size. Each cell already contains an active listing. One listing inside that cell is the test point.",
        "",
        "At every size the point is given to the cell that contains it. A forest trained on the other municipalities scores that cell. The predicted class is listed when the score is at or above that fold's Youden cutoff, and empty otherwise. The actual class is listed, because the cell contains the listing. A prediction is correct only when the model also says listed.",
        "",
        "The rule does not change between grid sizes. A coarser grid can put more than one of the 110 points in the same cell. Each of the 110 points is still counted once. A point that falls in a cell the grid drops, because less than half of that cell is land, is counted incorrect: there is no kept cell there for the model to call listed.",
        "",
        "| Grid size | Listing cells tested | Predicted correctly | Predicted incorrectly | Accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {int(row.cell_size_m)} m | {int(row.tested)} | {int(row.correct)} | {int(row.incorrect)} | {row.accuracy_percent:.1f}% |"
        )
    best = summary.sort_values(["accuracy_percent", "cell_size_m"], ascending=[False, True]).iloc[0]
    lines.extend(
        [
            "",
            f"**{int(best.cell_size_m)} m is the most accurate on these 110 listing cells, at {best.accuracy_percent:.1f}%.**",
            "",
        ]
    )
    for row in summary.itertuples(index=False):
        if int(row.unique_cells_scored) != int(row.tested):
            lines.append(
                f"At {int(row.cell_size_m)} m the 110 points fall in {int(row.unique_cells_scored)} different cells. Points that share a cell share that cell's prediction, and each point is still counted."
            )
    for row in summary.itertuples(index=False):
        if int(row.outside_grid) == 0:
            continue
        lines.append(
            f"{int(row.cell_size_m)} m drops {int(row.outside_grid)} of the 110 points because the cell around the listing is less than half land. Those points are in the incorrect count."
        )
    if any(int(row.outside_grid) > 0 for row in summary.itertuples(index=False)):
        lines.append("")
    for cell_size_m, group in details.groupby("cell_size_m"):
        notes = group.loc[group["note"].ne(""), "note"].value_counts()
        for note, count in notes.items():
            lines.append(f"At {int(cell_size_m)} m, {int(count)} incorrect points are incorrect because {note}.")
    if details["note"].ne("").any():
        lines.append("")
    (OUT / "listing_cell_accuracy.md").write_text("\n".join(lines), encoding="utf-8")


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
    featured_1km, oof_1km, report_1km = _featured_grid(1000, *shared)
    points = _reference_points(featured_1km, listings)
    checked = [_score_known_cells(1000, points, featured_1km, oof_1km, report_1km)]
    print(f"  1000 m correct {int(checked[0]['correct'].sum())} of {len(checked[0])}", flush=True)
    for cell_size_m in SIZES_M:
        if cell_size_m == 1000:
            continue
        featured, oof, report = _featured_grid(cell_size_m, *shared)
        scored = _score_known_cells(cell_size_m, points, featured, oof, report)
        checked.append(scored)
        print(f"  {cell_size_m} m correct {int(scored['correct'].sum())} of {len(scored)}", flush=True)
    details = pd.concat(checked, ignore_index=True)
    details = details.sort_values(["cell_size_m", "reference_cell_id"])
    details[
        [
            "cell_size_m",
            "reference_cell_id",
            CELL_ID_COLUMN,
            "municipality",
            "actual_label",
            "fold",
            "random_forest_raw",
            "youden_threshold",
            "predicted_label",
            "correct",
            "contained",
            "note",
        ]
    ].rename(columns={CELL_ID_COLUMN: "scored_cell_id", "random_forest_raw": "score"}).to_csv(
        OUT / "listing_cell_predictions.csv", index=False
    )
    rows = []
    for cell_size_m, group in details.groupby("cell_size_m"):
        correct = int(group["correct"].sum())
        tested = int(len(group))
        rows.append(
            {
                "cell_size_m": int(cell_size_m),
                "tested": tested,
                "correct": correct,
                "incorrect": tested - correct,
                "accuracy_percent": 100.0 * correct / tested,
                "unique_cells_scored": int(group.loc[group["contained"], CELL_ID_COLUMN].nunique()),
                "outside_grid": int((~group["contained"]).sum()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("cell_size_m")
    summary.to_csv(OUT / "listing_cell_accuracy.csv", index=False)
    _write_report(summary, details)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
