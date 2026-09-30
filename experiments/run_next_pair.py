"""Score the two strongest extras alone, then only those two together.

Same 1 km grid, 25/50/25 labels, production Random Forest settings, and the
same municipality folds as the baseline. Does not touch the thesis map.
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
    append_result_rows,
    evaluate_grouped,
    evaluate_lomo,
    grouped_kfold_splits,
    paired_change,
    results_row,
    save_experiment,
)

EXPERIMENTS_DIR = Path(__file__).parent
FEATURE_TABLE_PATH = EXPERIMENTS_DIR / "feature_engineering" / "engineered_features.parquet"
RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"
BASELINE = list(CLASSIFIER_FEATURES)
MIX = ["high_performer_pct_5km", "moderate_performer_pct_5km", "low_performer_pct_5km"]
ROAD = ["road_density_3km"]


def high_calls(metrics: dict) -> tuple[int, int]:
    matrix = metrics["confusion_matrix"]
    predicted = sum(row[2] for row in matrix)
    matched = matrix[2][2]
    return matched, predicted


def run(exp_id: str, added: list[str], frame: pd.DataFrame, folds, baseline_grouped, baseline_lomo) -> dict:
    features = BASELINE + added
    grouped = evaluate_grouped(frame, features, PERFORMANCE_CLASS, MUNICIPALITY_COLUMN, folds=folds)
    lomo = evaluate_lomo(frame, features, PERFORMANCE_CLASS, MUNICIPALITY_COLUMN)
    paired = paired_change(grouped["folds"], baseline_grouped["folds"])
    lomo_paired = paired_change(lomo["folds"], baseline_lomo["folds"], keys=("macro_f1",))
    matched, predicted = high_calls(grouped)
    both_up = grouped["macro_f1"] > baseline_grouped["macro_f1"] and lomo["macro_f1"] > baseline_lomo["macro_f1"]
    notes = (
        f"Adds {', '.join(added)} to the five production features. "
        f"Grouped macro F1 {grouped['macro_f1']:.4f} vs baseline {baseline_grouped['macro_f1']:.4f}. "
        f"LOMO {lomo['macro_f1']:.4f} vs baseline {baseline_lomo['macro_f1']:.4f}. "
        f"High calls {matched}/{predicted}. "
        f"Paired grouped macro F1 change {paired['macro_f1']['mean']:.4f} "
        f"(SD {paired['macro_f1']['std']:.4f}). "
        f"Paired LOMO macro F1 change {lomo_paired['macro_f1']['mean']:.4f} "
        f"(SD {lomo_paired['macro_f1']['std']:.4f}). "
        + (
            "Both grouped CV and leave-one-municipality-out rose."
            if both_up
            else "The gain was not present on both validation views."
        )
    )
    out = EXPERIMENTS_DIR / "final_candidates" / exp_id
    save_experiment(
        out,
        {
            "experiment_id": exp_id,
            "added_features": added,
            "features": features,
            "paired_grouped": paired,
            "paired_lomo_macro_f1": lomo_paired["macro_f1"],
            "high_matched": matched,
            "high_predicted": predicted,
            "both_views_improved": both_up,
        },
        grouped,
        features,
        f"# {exp_id}\n\n{notes}\n",
        lomo_metrics=lomo,
    )
    print(notes)
    return results_row(
        exp_id,
        "next_pair",
        "random_forest",
        grouped,
        len(features),
        lomo_metrics=lomo,
        notes=notes,
    )


def main() -> None:
    frame = pd.read_parquet(FEATURE_TABLE_PATH)
    folds, scheme = grouped_kfold_splits(
        frame[BASELINE],
        frame[PERFORMANCE_CLASS].astype(int),
        frame[MUNICIPALITY_COLUMN].astype(str),
    )
    baseline_grouped = evaluate_grouped(frame, BASELINE, PERFORMANCE_CLASS, MUNICIPALITY_COLUMN, folds=folds)
    baseline_lomo = evaluate_lomo(frame, BASELINE, PERFORMANCE_CLASS, MUNICIPALITY_COLUMN)
    print("scheme", scheme, "baseline grouped", round(baseline_grouped["macro_f1"], 4), "lomo", round(baseline_lomo["macro_f1"], 4))
    rows = [
        run("EXP-NEXT-MIX", MIX, frame, folds, baseline_grouped, baseline_lomo),
        run("EXP-NEXT-ROAD", ROAD, frame, folds, baseline_grouped, baseline_lomo),
        run("EXP-NEXT-BOTH", MIX + ROAD, frame, folds, baseline_grouped, baseline_lomo),
    ]
    append_result_rows(RESULTS_CSV, rows)
    summary = [
        {
            "experiment_id": row["experiment_id"],
            "macro_f1": row["macro_f1"],
            "lomo": row["mean_municipality_score"],
        }
        for row in rows
    ]
    (EXPERIMENTS_DIR / "final_candidates" / "next_pair_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
