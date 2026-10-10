"""Permutation importance at 500 m, 1 km, and 1.5 km.

The 1 km row uses the saved thesis grid. Writes only under
results/grid_importance/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_grid_size_ten_draw import _build_grid
from run_grid_size_validation import _with_barangay_centers
from run_site_candidate_robustness import SEEDS, full_population
from str_suitability import config
from str_suitability.ingest.load_osm import load_boundaries
from str_suitability.modeling.site_candidate import (
    LAGUNA_ROADS_PATH,
    build_training_sample,
    evaluate_forest,
    load_cells,
    load_places,
)
from str_suitability.pipeline import load_water
from str_suitability.rural.psa import (
    classify_barangay_polygons,
    load_barangay_classification,
    load_barangay_polygons,
)
from str_suitability.spatial.grid import derive_land_boundary

OUT = config.PROJECT_ROOT / "results" / "grid_importance"
ROAD_FEATURES = ("distance_to_major_road_km", "road_distance_km")
FEATURE_LABELS = {
    "distance_to_major_road_km": "Nearest major road",
    "road_distance_km": "Nearest mapped road",
    "listed_places_within_radius": "Landscape places within 5 km",
    "distance_to_nearest_town_center_km": "Nearest town centre",
    "distance_to_listed_tourist_place": "Nearest landscape place",
    "attraction_count_within_5km": "Falls and mountains within 5 km",
}
PUBLISHED_SEED_42 = {
    "distance_to_major_road_km": 0.11433676464703628,
    "road_distance_km": 0.07984960813568623,
    "listed_places_within_radius": 0.018226433120108847,
    "distance_to_nearest_town_center_km": 0.018110563401416556,
    "distance_to_listed_tourist_place": 0.013523254260202473,
    "attraction_count_within_5km": 0.009723363730685464,
}
PUBLISHED_ALL_EMPTY = {
    "distance_to_major_road_km": 0.07671572121998968,
    "road_distance_km": 0.06161688675519712,
}
TOLERANCE = 1e-9


def _records(report: dict, cell_size_m: int, population: str, seed: int | None) -> list[dict[str, object]]:
    ranked = sorted(
        report["permutation_importance"],
        key=lambda row: float(row["mean_auc_drop"]),
        reverse=True,
    )
    records = []
    for rank, row in enumerate(ranked, start=1):
        folds = [float(value) for value in row["fold_values"]]
        record = {
            "cell_size_m": cell_size_m,
            "population": population,
            "seed": seed if seed is not None else "",
            "feature": row["feature"],
            "feature_label": FEATURE_LABELS[row["feature"]],
            "rank": rank,
            "mean_auc_drop": float(row["mean_auc_drop"]),
            "std_across_folds": float(row["std_auc_drop"]),
            "folds_positive": int(sum(value > 0 for value in folds)),
            "folds": 5,
        }
        for index, value in enumerate(folds, start=1):
            record[f"fold_{index}"] = value
        records.append(record)
    return records


def _evaluate(cells: pd.DataFrame, cell_size_m: int, population: str, seed: int | None) -> list[dict[str, object]]:
    if population == "balanced":
        sample = build_training_sample(cells, negative_seed=int(seed))
        label = f"{cell_size_m} m seed {seed}"
    else:
        sample = full_population(cells)
        label = f"{cell_size_m} m all empty"
    _oof, report = evaluate_forest(sample)
    print(f"  {label}", flush=True)
    for row in sorted(report["permutation_importance"], key=lambda item: float(item["mean_auc_drop"]), reverse=True):
        print(f"    {row['feature']} {float(row['mean_auc_drop']):.6f}", flush=True)
    return _records(report, cell_size_m, population, seed)


def _mismatch(actual: dict[str, float], expected: dict[str, float], title: str) -> list[str]:
    problems = []
    for feature, published in expected.items():
        value = actual[feature]
        if abs(value - published) > TOLERANCE:
            problems.append(
                f"{title}: {feature} drop is {value:.12f}, published {published:.12f} "
                f"(rounds to {value:.3f} vs {published:.3f})"
            )
    return problems


def _roads_are_top_two(frame: pd.DataFrame) -> bool:
    top = set(frame.nsmallest(2, "rank")["feature"])
    return top == set(ROAD_FEATURES)


def _balanced_summary(draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cell_size_m, feature), group in draws.groupby(["cell_size_m", "feature"], sort=False):
        drops = group["mean_auc_drop"]
        rows.append(
            {
                "cell_size_m": int(cell_size_m),
                "feature": feature,
                "feature_label": FEATURE_LABELS[feature],
                "mean_drop_across_draws": float(drops.mean()),
                "lowest_draw": float(drops.min()),
                "highest_draw": float(drops.max()),
                "std_across_draws": float(drops.std(ddof=1)),
                "seed_42_mean_drop": float(group.loc[group["seed"].eq(42), "mean_auc_drop"].iloc[0]),
                "seed_42_std_across_folds": float(group.loc[group["seed"].eq(42), "std_across_folds"].iloc[0]),
                "seed_42_folds_positive": int(group.loc[group["seed"].eq(42), "folds_positive"].iloc[0]),
                "positive_fold_measurements": int(group["folds_positive"].sum()),
                "fold_measurements": int(group["folds"].sum()),
            }
        )
    frame = pd.DataFrame(rows)
    frame["rank"] = frame.groupby("cell_size_m")["mean_drop_across_draws"].rank(ascending=False, method="min").astype(int)
    return frame.sort_values(["cell_size_m", "rank", "feature"])


def _dominance(draws: pd.DataFrame, all_empty: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    intruders = []
    for cell_size_m, group in draws.groupby("cell_size_m"):
        wins = 0
        for _seed, draw in group.groupby("seed"):
            if _roads_are_top_two(draw):
                wins += 1
            else:
                others = sorted(set(draw.nsmallest(2, "rank")["feature"]) - set(ROAD_FEATURES))
                intruders.append(
                    f"{int(cell_size_m)} m balanced seed {int(_seed)} puts "
                    + ", ".join(FEATURE_LABELS[feature] for feature in others)
                    + " in the top two"
                )
        rows.append(
            {
                "cell_size_m": int(cell_size_m),
                "population": "balanced",
                "runs": int(group["seed"].nunique()),
                "runs_where_roads_are_top_two": wins,
            }
        )
    for cell_size_m, group in all_empty.groupby("cell_size_m"):
        roads_top = _roads_are_top_two(group)
        if not roads_top:
            others = sorted(set(group.nsmallest(2, "rank")["feature"]) - set(ROAD_FEATURES))
            intruders.append(
                f"{int(cell_size_m)} m all-empty puts "
                + ", ".join(FEATURE_LABELS[feature] for feature in others)
                + " in the top two"
            )
        rows.append(
            {
                "cell_size_m": int(cell_size_m),
                "population": "all_empty",
                "runs": 1,
                "runs_where_roads_are_top_two": int(roads_top),
            }
        )
    return pd.DataFrame(rows), intruders


def _write_summary(balanced: pd.DataFrame, all_empty: pd.DataFrame, dominance: pd.DataFrame, intruders: list[str]) -> None:
    lines = [
        "# Road-feature importance at three grid sizes",
        "",
        "Each number is the drop in ROC-AUC after shuffling one feature in the held-out cells, averaged over 20 shuffles and then over the five municipality folds. The balanced rows draw an equal number of empty cells. The 1 km seed-42 drops and the 1 km all-empty road drops matched the published values.",
        "",
    ]
    for cell_size_m in (500, 1000, 1500):
        lines.append(f"## {cell_size_m} m, ten balanced draws")
        lines.append("")
        lines.append("| Rank | Feature | Mean drop | Lowest draw | Highest draw | Seed 42 drop | Seed 42 folds that fell |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
        subset = balanced.loc[balanced["cell_size_m"].eq(cell_size_m)].sort_values("rank")
        for row in subset.itertuples(index=False):
            lines.append(
                f"| {int(row.rank)} | {row.feature_label} | {row.mean_drop_across_draws:.3f} | {row.lowest_draw:.3f} | "
                f"{row.highest_draw:.3f} | {row.seed_42_mean_drop:.3f} | {int(row.seed_42_folds_positive)} of 5 |"
            )
        lines.append("")
        lines.append(f"## {cell_size_m} m, all empty cells")
        lines.append("")
        lines.append("| Rank | Feature | Mean drop | Std across folds | Folds that fell |")
        lines.append("| ---: | --- | ---: | ---: | ---: |")
        empty = all_empty.loc[all_empty["cell_size_m"].eq(cell_size_m)].sort_values("rank")
        for row in empty.itertuples(index=False):
            lines.append(
                f"| {int(row.rank)} | {row.feature_label} | {row.mean_auc_drop:.3f} | {row.std_across_folds:.3f} | "
                f"{int(row.folds_positive)} of 5 |"
            )
        lines.append("")
    lines.append("## Do the two road features stay on top?")
    lines.append("")
    for row in dominance.itertuples(index=False):
        lines.append(
            f"{int(row.cell_size_m)} m, {row.population}: the two road features are the top two in "
            f"{int(row.runs_where_roads_are_top_two)} of {int(row.runs)} runs."
        )
    lines.append("")
    if intruders:
        lines.append("A feature other than the two road distances entered the top two in these runs:")
        lines.append("")
        lines.extend(f"- {item}" for item in intruders)
    else:
        lines.append("No landscape or town-centre feature entered the top two in any balanced draw or all-empty run.")
    lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_shared():
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
    return (land, barangays, municipalities, listings, places, roads)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("1 km original grid, seed 42", flush=True)
    cells_1km, _places = load_cells()
    seed_42 = _evaluate(cells_1km, 1000, "balanced", 42)
    actual = {row["feature"]: float(row["mean_auc_drop"]) for row in seed_42}
    problems = _mismatch(actual, PUBLISHED_SEED_42, "1 km seed 42")
    if problems:
        print("1 km seed 42 importance did not reproduce the published drops:", flush=True)
        for problem in problems:
            print(f"  {problem}", flush=True)
        raise SystemExit(1)
    print("1 km seed 42 importance gate passed", flush=True)

    all_empty_1km = _evaluate(cells_1km, 1000, "all_empty", None)
    actual_empty = {row["feature"]: float(row["mean_auc_drop"]) for row in all_empty_1km}
    problems = _mismatch(actual_empty, PUBLISHED_ALL_EMPTY, "1 km all-empty")
    if problems:
        print("1 km all-empty road importance did not reproduce the published drops:", flush=True)
        for problem in problems:
            print(f"  {problem}", flush=True)
        raise SystemExit(1)
    print("1 km all-empty importance gate passed", flush=True)

    records = seed_42 + all_empty_1km
    for seed in SEEDS:
        if seed == 42:
            continue
        records.extend(_evaluate(cells_1km, 1000, "balanced", seed))

    shared = _load_shared()
    for cell_size_m in (500, 1500):
        cells = _build_grid(cell_size_m, shared)
        for seed in SEEDS:
            records.extend(_evaluate(cells, cell_size_m, "balanced", seed))
        records.extend(_evaluate(cells, cell_size_m, "all_empty", None))

    detail = pd.DataFrame(records)
    balanced = detail.loc[detail["population"].eq("balanced")].copy()
    all_empty = detail.loc[detail["population"].eq("all_empty")].copy()
    summary = _balanced_summary(balanced)
    dominance, intruders = _dominance(balanced, all_empty)
    detail.to_csv(OUT / "importance_detail.csv", index=False)
    summary.to_csv(OUT / "importance_balanced.csv", index=False)
    all_empty.sort_values(["cell_size_m", "rank"]).to_csv(OUT / "importance_all_empty.csv", index=False)
    dominance.to_csv(OUT / "road_top_two.csv", index=False)
    _write_summary(summary, all_empty, dominance, intruders)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
