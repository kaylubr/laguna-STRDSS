"""Feature-group experiments (spec groups A, B, C, E, F, G, I, J, K).

Each experiment answers one specific question: does adding these columns to the
baseline five features change municipality-grouped validation performance, and
is any change consistent across held-out municipalities (Leave-One-Municipality-
Out), not just on one split? Every experiment reuses the SAME grouped-CV folds
as the baseline (target and municipality groups are unchanged for these
experiments), so comparisons are apples-to-apples.

Run with: uv run python -m experiments.run_feature_experiments
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from str_suitability.modeling.location_classifier import (
    CLASSIFIER_FEATURES,
    MUNICIPALITY_COLUMN,
    PERFORMANCE_CLASS,
)

from experiments.lab.common import (
    RESULTS_COLUMNS,
    evaluate_grouped,
    evaluate_lomo,
    grouped_kfold_splits,
    results_row,
    save_experiment,
)

EXPERIMENTS_DIR = Path(__file__).parent
FEATURE_TABLE_PATH = EXPERIMENTS_DIR / "feature_engineering" / "engineered_features.parquet"
RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"

BASELINE_FEATURES = list(CLASSIFIER_FEATURES)


def make_registry() -> list[dict]:
    return [
        # --- Group A: Airbnb market multi-radius (count, density, revenue, occupancy)
        dict(id="EXP-A01", group="feature_market", model="random_forest",
             add=["airbnb_count_1km", "avg_revenue_1km", "avg_occupancy_1km"],
             q="Does swapping the default 5 km market neighborhood for a 1 km one, added alongside "
               "the baseline 5 km features, change validation performance?"),
        dict(id="EXP-A02", group="feature_market", model="random_forest",
             add=["airbnb_count_3km", "avg_revenue_3km", "avg_occupancy_3km"],
             q="Same question at 3 km."),
        dict(id="EXP-A03", group="feature_market", model="random_forest",
             add=["airbnb_count_10km", "avg_revenue_10km", "avg_occupancy_10km"],
             q="Same question at 10 km."),
        dict(id="EXP-A04", group="feature_market", model="random_forest",
             add=["airbnb_density_5km"],
             q="Does adding listing density (count / circle area) at the production radius help "
               "beyond the raw surrounding listing count already in the baseline?"),
        dict(id="EXP-A05", group="feature_market", model="random_forest",
             add=["high_performer_pct_5km", "moderate_performer_pct_5km", "low_performer_pct_5km"],
             q="Does the market-distribution mix (share of nearby listings that are themselves "
               "historically Low/Moderate/High performers) add information beyond the mean?"),
        dict(id="EXP-A06", group="feature_market", model="random_forest",
             add=[f"airbnb_count_{s}" for s in ("500m", "1km", "3km", "5km", "10km")],
             q="Does having every radius's raw count available at once (letting the forest choose "
               "scale) beat any single fixed radius?"),
        # --- Group B: market stability
        dict(id="EXP-B01", group="feature_market", model="random_forest",
             add=["revenue_std_5km", "occupancy_std_5km"],
             q="Does the SPREAD of nearby revenue/occupancy (not just the mean) carry information?"),
        dict(id="EXP-B02", group="feature_market", model="random_forest",
             add=["revenue_iqr_5km", "occupancy_iqr_5km"],
             q="Same question using IQR instead of standard deviation (robust to outlier listings)."),
        # --- Group C: POI multi-radius + real categories
        dict(id="EXP-C01", group="feature_poi", model="random_forest",
             add=["poi_count_1km"],
             q="Does a landscape-place count at 1 km add information beyond the baseline's 5 km "
               "listed_places_within_radius?"),
        dict(id="EXP-C02", group="feature_poi", model="random_forest",
             add=["poi_count_10km"],
             q="Same question at 10 km."),
        dict(id="EXP-C03", group="feature_poi", model="random_forest",
             add=["nearest_poi_distance_km", "avg_distance_3_nearest_poi_km"],
             q="Does the nearest-place distance and average distance to the 3 nearest places add "
               "information beyond the baseline's single nearest-place distance and a 5 km count?"),
        dict(id="EXP-C04", group="feature_poi", model="random_forest",
             add=["falls_count_5km", "mountains_count_5km", "lakes_ponds_count_5km", "rivers_landscape_count_5km"],
             q="Does knowing WHICH kind of landscape place is nearby (not just how many) help, using "
               "the real categories in poi_laguna.json?"),
        # --- Group E: road accessibility (kept separate from the core model per the spec)
        dict(id="EXP-E01", group="feature_road", model="random_forest",
             add=["nearest_road_distance_km"],
             q="Independent replication: does straight-line distance to the nearest mapped road add "
               "information? (The production model already tested this in "
               "compare_road_accessibility and did not adopt it; this is a separate re-test.)"),
        dict(id="EXP-E02", group="feature_road", model="random_forest",
             add=["nearest_major_road_distance_km", "nearest_local_road_distance_km"],
             q="Does distinguishing major vs. local road distance (using the real road_class field) "
               "help more than one undifferentiated nearest-road distance?"),
        dict(id="EXP-E03", group="feature_road", model="random_forest",
             add=["road_count_3km", "road_density_3km"],
             q="Does road density within 3 km (a network-coverage signal, not just nearest distance) "
               "help?"),
        # --- Group F: access to services (proxy: OSM POI density + poblacion distance; the dataset
        # has no separate hospital/market/supermarket tags, so this is the closest real substitute)
        dict(id="EXP-F01", group="feature_accessibility", model="random_forest",
             add=["poi_density_commercial", "poi_density_transportation", "distance_to_poblacion"],
             q="Does commercial/transport OSM POI density plus distance to the town center add "
               "information? (No hospital/market/supermarket tags exist in this dataset; this is a "
               "documented proxy, not the literal Group F feature list.)"),
        # --- Group I: water / nature
        dict(id="EXP-I01", group="feature_accessibility", model="random_forest",
             add=["distance_to_laguna_de_bay", "distance_to_other_water"],
             q="Does distance to Laguna de Bay or to other mapped water add information?"),
        # --- Group J: urban proximity
        dict(id="EXP-J01", group="feature_accessibility", model="random_forest",
             add=["distance_to_nearest_urban_cell_km", "urban_area_share_5km"],
             q="Does distance to the nearest urban grid cell, or the urban share of cells within 5 "
               "km, add information beyond the market/tourist features already in the baseline?"),
        # --- Group G: population
        dict(id="EXP-G01", group="feature_population", model="random_forest",
             add=["population_density_per_km2"],
             q="Does municipality population density add information? (ADR-0013 caution: this value "
               "is constant within a municipality and correlates with municipality identity, which "
               "the production model deliberately excludes as a direct predictor.)"),
        # --- Group K: spatial context (nearby labeled squares' own class, self excluded)
        dict(id="EXP-K01", group="spatial_radius", model="random_forest",
             add=["nearby_high_square_count_5km", "nearby_moderate_square_count_5km", "nearby_low_square_count_5km"],
             q="Does knowing how many nearby OTHER labeled squares were historically Low/Moderate/"
               "High (self excluded) add information, the same way the production surrounding-market "
               "features use other listings?"),
        dict(id="EXP-K02", group="spatial_radius", model="random_forest",
             add=["nearby_high_square_count_3km", "nearby_moderate_square_count_3km", "nearby_low_square_count_3km"],
             q="Same question at 3 km instead of 5 km."),
    ]


def run_experiment(entry: dict, engineered: pd.DataFrame, baseline_folds, existing_rows: list) -> dict:
    feature_cols = BASELINE_FEATURES + entry["add"]
    grouped = evaluate_grouped(
        engineered, feature_cols, PERFORMANCE_CLASS, MUNICIPALITY_COLUMN,
        model_name=entry["model"], folds=baseline_folds,
    )
    lomo = evaluate_lomo(engineered, feature_cols, PERFORMANCE_CLASS, MUNICIPALITY_COLUMN, model_name=entry["model"])
    config = {
        "experiment_id": entry["id"],
        "question": entry["q"],
        "baseline_features": BASELINE_FEATURES,
        "added_features": entry["add"],
        "all_features": feature_cols,
        "model": entry["model"],
        "n_labeled_cells": int(len(engineered)),
        "n_municipalities": int(engineered[MUNICIPALITY_COLUMN].nunique()),
        "validation_scheme_grouped_cv": grouped["scheme"],
        "validation_scheme_lomo": lomo["scheme"],
    }
    notes = (
        f"# {entry['id']}: {entry['group']}\n\n"
        f"**Question.** {entry['q']}\n\n"
        f"**Added features.** {', '.join(entry['add'])}\n\n"
        f"**Grouped-CV macro F1.** {grouped['macro_f1']:.4f} "
        f"(baseline reproduced macro F1 is in experiments/baseline/metrics.json)\n\n"
        f"**Leave-one-municipality-out macro F1.** mean {lomo['macro_f1']:.4f}, "
        f"std across {lomo['n_folds']} municipality folds {lomo['cv']['macro_f1']['std']:.4f}.\n\n"
        "A feature addition is only worth adopting if it helps on the grouped-CV score AND does not "
        "make the leave-one-municipality-out result meaningfully less consistent (lower mean, "
        "similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the "
        "cross-experiment comparison.\n"
    )
    save_experiment(
        EXPERIMENTS_DIR / entry["group"] / entry["id"], config, grouped, feature_cols, notes, lomo_metrics=lomo,
    )
    row = results_row(
        entry["id"], entry["group"], entry["model"], grouped, len(feature_cols),
        lomo_metrics=lomo, notes=entry["q"][:180],
    )
    print(
        f"{entry['id']:10s} {entry['group']:20s} macro_f1={grouped['macro_f1']:.4f} "
        f"lomo_mean={lomo['macro_f1']:.4f} lomo_std={lomo['cv']['macro_f1']['std']:.4f} "
        f"n_features={len(feature_cols)}"
    )
    return row


def main() -> None:
    engineered = pd.read_parquet(FEATURE_TABLE_PATH)
    baseline_folds, scheme = grouped_kfold_splits(
        engineered[BASELINE_FEATURES], engineered[PERFORMANCE_CLASS].astype(int), engineered[MUNICIPALITY_COLUMN].astype(str)
    )
    print("baseline folds scheme:", scheme, "n_folds:", len(baseline_folds))

    registry = make_registry()
    rows = []
    for entry in registry:
        rows.append(run_experiment(entry, engineered, baseline_folds, rows))

    # --- Combined candidate: take every individual addition whose grouped-CV macro F1 beat the
    # baseline AND whose LOMO mean also beat the baseline LOMO mean (computed just below), then
    # combine them into one feature set. This is reported honestly as "highest observed on this
    # data", not assumed to generalize, per the spec's "avoid winner by accident" instruction.
    baseline_lomo = evaluate_lomo(engineered, BASELINE_FEATURES, PERFORMANCE_CLASS, MUNICIPALITY_COLUMN)
    baseline_grouped = evaluate_grouped(
        engineered, BASELINE_FEATURES, PERFORMANCE_CLASS, MUNICIPALITY_COLUMN, folds=baseline_folds
    )
    print(
        f"{'BASELINE':10s} {'baseline':20s} macro_f1={baseline_grouped['macro_f1']:.4f} "
        f"lomo_mean={baseline_lomo['macro_f1']:.4f} lomo_std={baseline_lomo['cv']['macro_f1']['std']:.4f} "
        f"n_features={len(BASELINE_FEATURES)}"
    )

    winners = [
        entry for entry, row in zip(registry, rows, strict=True)
        if row["macro_f1"] is not None and row["macro_f1"] > baseline_grouped["macro_f1"]
        and row["mean_municipality_score"] != "" and row["mean_municipality_score"] > baseline_lomo["macro_f1"]
    ]
    combined_add = sorted({feature for entry in winners for feature in entry["add"]})
    if combined_add:
        combined_entry = dict(
            id="EXP-COMBINED-01", group="feature_market", model="random_forest", add=combined_add,
            q=(
                "Combines every individual feature addition above that beat the baseline on BOTH "
                "grouped-CV macro F1 and mean leave-one-municipality-out macro F1: "
                + ", ".join(entry["id"] for entry in winners) + ". This tests whether the individual "
                "gains stack, or cancel out / overfit when combined."
            ),
        )
        rows.append(run_experiment(combined_entry, engineered, baseline_folds, rows))
    else:
        print("No individual feature addition beat the baseline on both metrics; no combined candidate built.")

    if RESULTS_CSV.exists():
        RESULTS_CSV.unlink()  # this script is the first writer in the results.csv pipeline
    from experiments.lab.common import append_result_rows
    baseline_row = results_row(
        "EXP-000-baseline", "baseline", "random_forest", baseline_grouped, len(BASELINE_FEATURES),
        lomo_metrics=baseline_lomo, notes="The production five-feature model, reproduced independently.",
    )
    append_result_rows(RESULTS_CSV, [baseline_row, *rows])
    print(f"\nWrote {1 + len(rows)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
