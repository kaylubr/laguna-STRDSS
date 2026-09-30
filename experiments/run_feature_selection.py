"""Drop redundant columns from EXP-COMBINED-01 and re-score the same folds.

A lean subset keeps only the additions that beat the baseline on both grouped
macro F1 and leave-one-municipality-out macro F1. A correlation subset drops
one of each pair with |r| > 0.9, always keeping the five production features.
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
    evaluate_grouped,
    evaluate_lomo,
    results_row,
    append_result_rows,
    save_experiment,
    grouped_kfold_splits,
)

EXPERIMENTS_DIR = Path(__file__).parent
RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"
FEATURE_TABLE_PATH = EXPERIMENTS_DIR / "feature_engineering" / "engineered_features.parquet"
COMBINED_PATH = EXPERIMENTS_DIR / "feature_market" / "EXP-COMBINED-01" / "feature_list.json"

LEAN_ADDED = (
    "airbnb_count_1km",
    "avg_revenue_1km",
    "avg_occupancy_1km",
    "high_performer_pct_5km",
    "moderate_performer_pct_5km",
    "low_performer_pct_5km",
    "revenue_iqr_5km",
    "occupancy_iqr_5km",
    "nearest_road_distance_km",
    "road_density_3km",
    "poi_density_commercial",
    "poi_density_transportation",
    "distance_to_poblacion",
    "distance_to_laguna_de_bay",
    "distance_to_nearest_urban_cell_km",
    "urban_area_share_5km",
)


def drop_correlated(frame: pd.DataFrame, columns: list[str], keep: list[str], threshold: float = 0.9) -> list[str]:
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr().abs()
    selected = list(keep)
    for column in columns:
        if column in selected:
            continue
        if any(corr.loc[column, kept] > threshold for kept in selected if kept in corr.index and column in corr.index):
            continue
        selected.append(column)
    return selected


def run_one(exp_id: str, features: list[str], frame: pd.DataFrame, folds, notes: str) -> dict:
    grouped = evaluate_grouped(frame, features, PERFORMANCE_CLASS, MUNICIPALITY_COLUMN, folds=folds)
    lomo = evaluate_lomo(frame, features, PERFORMANCE_CLASS, MUNICIPALITY_COLUMN)
    out = EXPERIMENTS_DIR / "final_candidates" / exp_id
    save_experiment(
        out,
        {
            "experiment_id": exp_id,
            "features": features,
            "n_features": len(features),
            "notes": notes,
        },
        grouped,
        features,
        notes,
        lomo_metrics=lomo,
    )
    return results_row(
        exp_id,
        "feature_selection",
        "random_forest",
        grouped,
        len(features),
        lomo_metrics=lomo,
        notes=notes,
    )


def main() -> None:
    frame = pd.read_parquet(FEATURE_TABLE_PATH)
    combined = json.loads(COMBINED_PATH.read_text(encoding="utf-8"))
    folds, _ = grouped_kfold_splits(
        frame[list(CLASSIFIER_FEATURES)],
        frame[PERFORMANCE_CLASS].astype(int),
        frame[MUNICIPALITY_COLUMN].astype(str),
    )
    lean = list(CLASSIFIER_FEATURES) + [column for column in LEAN_ADDED if column in frame.columns]
    reduced = drop_correlated(frame, combined, list(CLASSIFIER_FEATURES))
    corr_path = EXPERIMENTS_DIR / "final_candidates" / "correlation_matrix.csv"
    corr_path.parent.mkdir(parents=True, exist_ok=True)
    frame[combined].apply(pd.to_numeric, errors="coerce").corr().to_csv(corr_path)

    rows = [
        run_one(
            "EXP-SEL-lean",
            lean,
            frame,
            folds,
            "Lean set: production five plus the additions that beat the baseline on both grouped and leave-one-municipality-out macro F1.",
        ),
        run_one(
            "EXP-SEL-decorrelated",
            reduced,
            frame,
            folds,
            f"EXP-COMBINED-01 after dropping pairs with |r| > 0.9. Kept {len(reduced)} of {len(combined)} columns.",
        ),
    ]
    append_result_rows(RESULTS_CSV, rows)
    print(json.dumps([{k: row[k] for k in ("experiment_id", "n_features", "macro_f1", "mean_municipality_score")} for row in rows], indent=2))


if __name__ == "__main__":
    main()
