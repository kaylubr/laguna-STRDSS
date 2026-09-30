"""Regression alternative (spec group O): predict cell_mean_revenue and
cell_mean_occupancy directly instead of a Low/Moderate/High class, using the
same five surrounding features and the same municipality-grouped validation.

MAE/RMSE/R2 here are NOT compared to the classifier's accuracy/F1 numbers
anywhere in this repo; classification and regression scores are not on the
same scale, and mixing them would be a methodology error. The only question
this answers is whether continuous prediction looks more appropriate than the
three-class split, on its own terms.

Run with: uv run python -m experiments.run_regression
"""

from __future__ import annotations

import json
from pathlib import Path

from str_suitability.modeling.location_classifier import (
    CELL_OCCUPANCY,
    CELL_REVENUE,
    CLASSIFIER_FEATURES,
    MUNICIPALITY_COLUMN,
    label_performance,
    market_features,
)

from experiments.lab import data as lab_data
from experiments.lab.common import append_result_rows
from experiments.lab.regression import evaluate_regression

EXPERIMENTS_DIR = Path(__file__).parent
RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"
BASELINE_FEATURES = list(CLASSIFIER_FEATURES)


def main() -> None:
    grid = lab_data.load_grid()
    active_listings = lab_data.load_active_listings()
    places = lab_data.load_places_with_category()
    features = market_features(grid, active_listings, places)
    labeled, _ = label_performance(features)

    rows = []
    for exp_id, target_col, label in (
        ("EXP-O-revenue", CELL_REVENUE, "cell_mean_revenue"),
        ("EXP-O-occupancy", CELL_OCCUPANCY, "cell_mean_occupancy"),
    ):
        result = evaluate_regression(labeled, BASELINE_FEATURES, target_col, MUNICIPALITY_COLUMN)
        cfg = {
            "experiment_id": exp_id, "target": label, "model": "RandomForestRegressor",
            "features": BASELINE_FEATURES, "n": result["n"], "scheme": result["scheme"],
        }
        notes = (
            f"# {exp_id}\n\nPredicts {label} directly (continuous), same five surrounding "
            f"features, same municipality-grouped folds as the classifier.\n\n"
            f"MAE: {result['mae']:.4f} (+/-{result['mae_std']:.4f}), "
            f"RMSE: {result['rmse']:.4f} (+/-{result['rmse_std']:.4f}), "
            f"R2: {result['r2']:.4f} (+/-{result['r2_std']:.4f}).\n\n"
            "These numbers are not on the same scale as the classifier's accuracy/F1 and are not "
            "compared to them anywhere in this report. A negative or near-zero R2 means the model "
            "explains little variance in the continuous target on held-out municipalities; that is "
            "itself a useful, honestly-reported result about whether continuous prediction is "
            "viable with this feature set and sample size.\n"
        )
        out_dir = EXPERIMENTS_DIR / "final_candidates" / "regression" / exp_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        (out_dir / "metrics.json").write_text(
            json.dumps({k: v for k, v in result.items() if k != "folds"}, indent=2), encoding="utf-8"
        )
        (out_dir / "notes.md").write_text(notes, encoding="utf-8")
        rows.append({
            "experiment_id": exp_id, "feature_group": "regression", "model": "RandomForestRegressor",
            "grid_size": "1000m", "poi_radius": "", "airbnb_radius": "", "target_definition": label,
            "n_features": len(BASELINE_FEATURES), "n_train": "", "n_test": result["n"],
            "accuracy": "", "macro_precision": "", "macro_recall": "", "macro_f1": "", "weighted_f1": "",
            "high_precision": "", "high_recall": "", "validation_score": result["r2"],
            "mean_municipality_score": "", "std_municipality_score": "",
            "notes": f"MAE={result['mae']:.4f} RMSE={result['rmse']:.4f} R2={result['r2']:.4f}; not comparable to classification scores",
        })
        print(f"{exp_id:18s} MAE={result['mae']:.4f} RMSE={result['rmse']:.4f} R2={result['r2']:.4f}")

    append_result_rows(RESULTS_CSV, rows)
    print(f"Wrote {len(rows)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
