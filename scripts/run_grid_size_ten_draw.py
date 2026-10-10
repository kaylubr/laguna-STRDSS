"""Ten empty-cell draws at 500 m, 1 km, 1.5 km, and 2 km.

The 1 km row uses the saved thesis grid. Other sizes are built with the
same rural, listing, and land rules. Writes only under
results/grid_size_ten_draw/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_grid_size_validation import (
    _listing_counts,
    _municipality,
    _with_barangay_centers,
)
from run_site_candidate_robustness import SEEDS, full_population
from str_suitability import config
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
    load_cells,
    load_places,
)
from str_suitability.pipeline import load_water
from str_suitability.rural.classify import (
    CELL_CLASS_COLUMN,
    RURAL_CELL,
    classify_grid_cells,
)
from str_suitability.rural.psa import (
    classify_barangay_polygons,
    load_barangay_classification,
    load_barangay_polygons,
)
from str_suitability.spatial.grid import build_grid, derive_land_boundary

OUT = config.PROJECT_ROOT / "results" / "grid_size_ten_draw"
SIZES_M = (500, 1500, 2000)
PUBLISHED = {
    "roc_mean": 0.6946731118128164,
    "roc_min": 0.6205890744389195,
    "roc_max": 0.7632940501436742,
    "seed_42": 0.7274258287458627,
}
TOLERANCE = 1e-9


def _population_counts(cells: pd.DataFrame) -> tuple[int, int]:
    population = full_population(cells)
    listed = int((population[PRESENCE_LABEL] == 1).sum())
    empty = int((population[PRESENCE_LABEL] == 0).sum())
    return listed, empty


def _draw_rows(cells: pd.DataFrame, cell_size_m: int) -> list[dict[str, object]]:
    rows = []
    for seed in SEEDS:
        sample = build_training_sample(cells, negative_seed=seed)
        _oof, report = evaluate_forest(sample)
        roc = float(report["metrics"]["roc_auc"]["mean"])
        pr = float(report["metrics"]["pr_auc"]["mean"])
        print(f"  {cell_size_m} m seed {seed}: ROC-AUC {roc:.6f} PR-AUC {pr:.6f}", flush=True)
        rows.append(
            {
                "cell_size_m": cell_size_m,
                "seed": seed,
                "training_rows": int(report["training_rows"]),
                "training_listed": int(report["training_positives"]),
                "training_empty": int(report["training_negatives"]),
                "roc_auc": roc,
                "pr_auc": pr,
            }
        )
    return rows


def _spread(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "lowest": float(array.min()),
        "highest": float(array.max()),
        "std": float(array.std(ddof=1)),
    }


def _gate(rows: list[dict[str, object]]) -> list[str]:
    roc = [float(row["roc_auc"]) for row in rows]
    seed_42 = float(next(row["roc_auc"] for row in rows if row["seed"] == 42))
    spread = _spread(roc)
    checks = {
        "mean": (spread["mean"], PUBLISHED["roc_mean"]),
        "lowest": (spread["lowest"], PUBLISHED["roc_min"]),
        "highest": (spread["highest"], PUBLISHED["roc_max"]),
        "seed 42": (seed_42, PUBLISHED["seed_42"]),
    }
    problems = []
    for name, (actual, expected) in checks.items():
        if abs(actual - expected) > TOLERANCE:
            problems.append(f"{name} ROC-AUC is {actual:.12f}, published {expected:.12f}")
    return problems


def _build_grid(cell_size_m: int, shared) -> pd.DataFrame:
    land, barangays, municipalities, listings, places, roads = shared
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
    return build_feature_frame(grid, places, roads)


def _all_empty(cells: pd.DataFrame, cell_size_m: int, listed: int, empty: int) -> dict[str, object]:
    population = full_population(cells)
    print(f"all-empty {cell_size_m} m: {listed} listed and {empty} empty", flush=True)
    _oof, report = evaluate_forest(population)
    chance = listed / (listed + empty)
    return {
        "cell_size_m": cell_size_m,
        "listed_cells": listed,
        "empty_cells": empty,
        "rows": int(report["training_rows"]),
        "roc_auc": float(report["metrics"]["roc_auc"]["mean"]),
        "pr_auc": float(report["metrics"]["pr_auc"]["mean"]),
        "pr_auc_chance": chance,
    }


def _summary_row(cell_size_m: int, listed: int, empty: int, rows: list[dict[str, object]]) -> dict[str, object]:
    roc = _spread([float(row["roc_auc"]) for row in rows])
    pr = _spread([float(row["pr_auc"]) for row in rows])
    return {
        "cell_size_m": cell_size_m,
        "listed_cells": listed,
        "empty_cells": empty,
        "training_rows": int(rows[0]["training_rows"]),
        "roc_auc_mean": roc["mean"],
        "roc_auc_lowest": roc["lowest"],
        "roc_auc_highest": roc["highest"],
        "roc_auc_std": roc["std"],
        "pr_auc_mean": pr["mean"],
        "pr_auc_lowest": pr["lowest"],
        "pr_auc_highest": pr["highest"],
        "pr_auc_std": pr["std"],
    }


def _against_1km(draws: pd.DataFrame) -> pd.DataFrame:
    base = draws.loc[draws["cell_size_m"].eq(1000), ["seed", "roc_auc"]].rename(columns={"roc_auc": "roc_auc_1km"})
    rows = []
    for cell_size_m in SIZES_M:
        paired = draws.loc[draws["cell_size_m"].eq(cell_size_m), ["seed", "roc_auc"]].merge(base, on="seed")
        difference = paired["roc_auc"] - paired["roc_auc_1km"]
        rows.append(
            {
                "cell_size_m": cell_size_m,
                "seeds": int(len(paired)),
                "seeds_beating_1km": int((difference > 0).sum()),
                "seeds_below_1km": int((difference < 0).sum()),
                "seeds_tied_with_1km": int(np.isclose(difference, 0).sum()),
                "mean_roc_auc_difference": float(difference.mean()),
                "std_roc_auc_difference": float(difference.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def _write_summary(summary: pd.DataFrame, versus: pd.DataFrame, all_empty: pd.DataFrame, flags: list[str]) -> None:
    one_km = summary.loc[summary["cell_size_m"].eq(1000)].iloc[0]
    lines = [
        "# Grid size across ten empty-cell draws",
        "",
        "Each draw keeps every listed rural cell and an equal number of empty rural cells. The empty cells change with the seed. The forest, the six features, and the five municipality folds stay the same. The 1 km row is the saved thesis grid. It reproduced the published ten-draw ROC-AUC: mean 0.695, range 0.621 to 0.763, and 0.727 at seed 42.",
        "",
        "Standard deviation is the sample standard deviation across the ten draws.",
        "",
        "| Grid | Listed | Empty | Training rows | ROC-AUC mean | ROC-AUC lowest | ROC-AUC highest | ROC-AUC std | PR-AUC mean | PR-AUC lowest | PR-AUC highest | PR-AUC std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {int(row.cell_size_m)} m | {int(row.listed_cells)} | {int(row.empty_cells)} | {int(row.training_rows)} | "
            f"{row.roc_auc_mean:.3f} | {row.roc_auc_lowest:.3f} | {row.roc_auc_highest:.3f} | {row.roc_auc_std:.3f} | "
            f"{row.pr_auc_mean:.3f} | {row.pr_auc_lowest:.3f} | {row.pr_auc_highest:.3f} | {row.pr_auc_std:.3f} |"
        )
    lines.extend(
        [
            "",
            "Paired with the 1 km draw that used the same empty-cell seed. A seed beats 1 km when its ROC-AUC is higher.",
            "",
            "| Grid | Seeds above 1 km | Seeds below 1 km | Mean ROC-AUC difference | Std of the difference |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in versus.itertuples(index=False):
        lines.append(
            f"| {int(row.cell_size_m)} m | {int(row.seeds_beating_1km)} of {int(row.seeds)} | {int(row.seeds_below_1km)} of {int(row.seeds)} | "
            f"{row.mean_roc_auc_difference:+.3f} | {row.std_roc_auc_difference:.3f} |"
        )
    lines.extend(
        [
            "",
            "All empty rural cells, one run, no draw. Chance PR-AUC is the share of listed cells.",
            "",
            "| Grid | ROC-AUC | PR-AUC | Chance PR-AUC |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in all_empty.itertuples(index=False):
        lines.append(
            f"| {int(row.cell_size_m)} m | {row.roc_auc:.3f} | {row.pr_auc:.3f} | {row.pr_auc_chance:.3f} |"
        )
    lines.extend(["", "## What is larger than the draw-to-draw spread", ""])
    for row in versus.itertuples(index=False):
        gap = abs(float(row.mean_roc_auc_difference))
        noise = float(row.std_roc_auc_difference)
        direction = "higher" if row.mean_roc_auc_difference > 0 else "lower"
        if gap > noise and int(row.seeds_beating_1km) in {0, int(row.seeds)}:
            lines.append(
                f"{int(row.cell_size_m)} m is clearly {direction} than 1 km. The mean ROC-AUC difference is {row.mean_roc_auc_difference:+.3f}, "
                f"and every seed is on that side. The draw-to-draw spread of the difference is {noise:.3f}."
            )
        elif gap <= noise:
            lines.append(
                f"{int(row.cell_size_m)} m is {direction} than 1 km by {row.mean_roc_auc_difference:+.3f} ROC-AUC. "
                f"That gap is smaller than the draw-to-draw spread of the difference, {noise:.3f}, so it is inside the noise of which empty cells were drawn."
            )
        else:
            lines.append(
                f"{int(row.cell_size_m)} m is {direction} than 1 km by {row.mean_roc_auc_difference:+.3f} ROC-AUC, "
                f"with {int(row.seeds_beating_1km)} of {int(row.seeds)} seeds above 1 km. "
                f"The spread of those differences is {noise:.3f}. The direction is uneven across seeds, so it is not a clean separation."
            )
    one_km_spread = float(one_km.roc_auc_highest - one_km.roc_auc_lowest)
    lines.append(
        f"The 1 km draws themselves run from {one_km.roc_auc_lowest:.3f} to {one_km.roc_auc_highest:.3f}, a spread of {one_km_spread:.3f}."
    )
    if flags:
        lines.extend(["", "## Checks that need attention", ""])
        lines.extend(f"- {flag}" for flag in flags)
    else:
        lines.extend(["", "No count or reproduction check failed.", ""])
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("1 km original grid", flush=True)
    cells_1km, _places = load_cells()
    listed, empty = _population_counts(cells_1km)
    if listed != 110 or empty != 931:
        raise SystemExit(f"original 1 km grid has {listed} listed and {empty} empty cells, not 110 and 931")
    rows = _draw_rows(cells_1km, 1000)
    problems = _gate(rows)
    if problems:
        print("1 km ten-draw result did not reproduce the published numbers:", flush=True)
        for problem in problems:
            print(f"  {problem}", flush=True)
        raise SystemExit(1)
    print("1 km reproduction gate passed", flush=True)

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

    built = {1000: cells_1km}
    flags = []
    expected = {500: (122, 4000), 1500: (90, 380), 2000: (71, 195)}
    for cell_size_m in SIZES_M:
        built[cell_size_m] = _build_grid(cell_size_m, shared)
        size_listed, size_empty = _population_counts(built[cell_size_m])
        if (size_listed, size_empty) != expected[cell_size_m]:
            flags.append(
                f"{cell_size_m} m has {size_listed} listed and {size_empty} empty cells; "
                f"an earlier build of the same rules had {expected[cell_size_m][0]} and {expected[cell_size_m][1]}"
            )
        rows.extend(_draw_rows(built[cell_size_m], cell_size_m))

    draws = pd.DataFrame(rows)
    all_empty_rows = []
    summary_rows = []
    for cell_size_m, cells in built.items():
        size_rows = [row for row in rows if row["cell_size_m"] == cell_size_m]
        size_listed, size_empty = _population_counts(cells)
        summary_rows.append(_summary_row(cell_size_m, size_listed, size_empty, size_rows))
        all_empty_rows.append(_all_empty(cells, cell_size_m, size_listed, size_empty))
    summary = pd.DataFrame(summary_rows).sort_values("cell_size_m")
    versus = _against_1km(draws)
    all_empty = pd.DataFrame(all_empty_rows).sort_values("cell_size_m")
    draws.sort_values(["cell_size_m", "seed"]).to_csv(OUT / "draws.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    versus.to_csv(OUT / "roc_vs_1km.csv", index=False)
    all_empty.to_csv(OUT / "all_empty.csv", index=False)
    _write_summary(summary, versus, all_empty, flags)
    print(summary.to_string(index=False), flush=True)
    print(versus.to_string(index=False), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
