"""The 1 km evaluation suite on the 500 m grid.

Seed-42 municipality holdout, Youden classification, ten negative-sample
seeds, and the all-empty run. Writes only under results/grid_size_validation/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_500m_map import _featured
from run_grid_size_validation import _with_barangay_centers
from run_hidden_label_validation import _classification
from run_site_candidate_robustness import SEEDS, _across_draws, full_population
from str_suitability import config
from str_suitability.ingest.load_osm import load_boundaries
from str_suitability.modeling.site_candidate import (
    CELL_ID_COLUMN,
    LAGUNA_ROADS_PATH,
    PRESENCE_LABEL,
    build_training_sample,
    evaluate_forest,
    load_places,
)
from str_suitability.pipeline import load_water
from str_suitability.rural.psa import (
    classify_barangay_polygons,
    load_barangay_classification,
    load_barangay_polygons,
)
from str_suitability.spatial.grid import derive_land_boundary

OUT = config.PROJECT_ROOT / "results" / "grid_size_validation"
ONE_KM = {
    "rows": 220,
    "listed": 110,
    "empty_sampled": 110,
    "empty_all": 931,
    "roc_auc": 0.7274258287458627,
    "pr_auc": 0.7128459949005714,
    "macro_f1": 0.6561302663653507,
    "accuracy": 0.6727272727272727,
    "precision_listed": 0.6338028169014085,
    "recall_listed": 0.8181818181818182,
    "f1_listed": 0.7142857142857143,
    "tn": 58,
    "fp": 52,
    "fn": 20,
    "tp": 90,
    "ten_roc_mean": 0.6946731118128164,
    "ten_roc_min": 0.6205890744389195,
    "ten_roc_max": 0.7632940501436742,
    "ten_pr_mean": 0.6667287733817282,
    "all_empty_roc": 0.6945554339703915,
    "all_empty_pr": 0.2193783062293458,
    "all_empty_f1": 0.4786327985165924,
}


def _mean(report: dict[str, object], metric: str) -> float:
    return float(report["metrics"][metric]["mean"])


def _classify_oof(oof: pd.DataFrame, report: dict[str, object]) -> pd.DataFrame:
    thresholds = {int(row["fold"]): float(row["youden_threshold"]) for row in report["folds"]}
    checked = oof.copy()
    checked["fold"] = checked["fold"].astype(int)
    checked["youden_threshold"] = checked["fold"].map(thresholds)
    checked["predicted_listed"] = checked["random_forest_raw"].ge(checked["youden_threshold"])
    checked["actual_label"] = np.where(checked[PRESENCE_LABEL].eq(1), "listed", "empty")
    checked["predicted_label"] = np.where(checked["predicted_listed"], "listed", "empty")
    checked["correct"] = checked["predicted_label"].eq(checked["actual_label"])
    return checked


def _top_features(report: dict[str, object]) -> list[dict[str, object]]:
    rows = sorted(
        report["permutation_importance"],
        key=lambda row: float(row["mean_auc_drop"]),
        reverse=True,
    )
    return [
        {"feature": row["feature"], "mean_auc_drop": float(row["mean_auc_drop"])}
        for row in rows
    ]


def _write_report(summary: dict[str, object]) -> None:
    seed = summary["seed_42"]
    classification = seed["classification"]
    ten = summary["ten_seeds"]
    full = summary["all_empty"]
    lines = [
        "# 500 m evaluation, same checks as the 1 km model",
        "",
        "The forest, the six features, the municipality holdout, the Youden cutoff, and the ten negative-sample seeds are the ones used for the 1 km model. The 500 m balanced sample is 122 listed cells plus 122 empty cells. The 1 km sample is 110 plus 110. The all-empty run uses every empty rural cell.",
        "",
        "| Check | 1 km | 500 m |",
        "| --- | ---: | ---: |",
        f"| Seed 42 ROC-AUC | {ONE_KM['roc_auc']:.3f} | {seed['roc_auc']:.3f} |",
        f"| Seed 42 PR-AUC | {ONE_KM['pr_auc']:.3f} | {seed['pr_auc']:.3f} |",
        f"| Seed 42 macro F1 | {ONE_KM['macro_f1']:.3f} | {seed['macro_f1']:.3f} |",
        f"| Accuracy | {ONE_KM['accuracy']:.3f} | {classification['accuracy']:.3f} |",
        f"| Precision, listed | {ONE_KM['precision_listed']:.3f} | {classification['precision_listed']:.3f} |",
        f"| Recall, listed | {ONE_KM['recall_listed']:.3f} | {classification['recall_listed']:.3f} |",
        f"| F1, listed | {ONE_KM['f1_listed']:.3f} | {classification['f1_listed']:.3f} |",
        f"| Ten-seed ROC-AUC mean | {ONE_KM['ten_roc_mean']:.3f} | {ten['roc_auc']['mean']:.3f} |",
        f"| Ten-seed ROC-AUC range | {ONE_KM['ten_roc_min']:.3f}–{ONE_KM['ten_roc_max']:.3f} | {ten['roc_auc']['min']:.3f}–{ten['roc_auc']['max']:.3f} |",
        f"| Ten-seed PR-AUC mean | {ONE_KM['ten_pr_mean']:.3f} | {ten['pr_auc']['mean']:.3f} |",
        f"| All-empty ROC-AUC | {ONE_KM['all_empty_roc']:.3f} | {full['roc_auc']:.3f} |",
        f"| All-empty PR-AUC | {ONE_KM['all_empty_pr']:.3f} | {full['pr_auc']:.3f} |",
        f"| All-empty macro F1 | {ONE_KM['all_empty_f1']:.3f} | {full['macro_f1']:.3f} |",
        "",
        "Seed 42 confusion matrix. Rows are the actual class.",
        "",
        "|  | Predicted empty | Predicted listed |",
        "| --- | ---: | ---: |",
        f"| Actually empty | {classification['true_negative']} | {classification['false_positive']} |",
        f"| Actually listed | {classification['false_negative']} | {classification['true_positive']} |",
        "",
        "The 1 km matrix was 58 true empty, 52 false listed, 20 missed listings, and 90 recovered listings.",
        "",
        "Seed 42 permutation drop in ROC-AUC, largest first:",
        "",
    ]
    for row in seed["permutation"]:
        lines.append(f"- {row['feature']}: {row['mean_auc_drop']:.3f}")
    lines.append("")
    (OUT / "evaluation_500m.md").write_text("\n".join(lines), encoding="utf-8")


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
    featured = _featured(land, barangays, municipalities, listings, places, roads)

    draws = []
    seed_42 = None
    for seed in SEEDS:
        sample = build_training_sample(featured, negative_seed=seed)
        oof, report = evaluate_forest(sample)
        record = {
            "seed": seed,
            "training_rows": int(report["training_rows"]),
            "roc_auc": _mean(report, "roc_auc"),
            "pr_auc": _mean(report, "pr_auc"),
            "macro_f1": _mean(report, "macro_f1"),
        }
        draws.append(record)
        print(
            f"seed {seed}: ROC-AUC {record['roc_auc']:.3f} PR-AUC {record['pr_auc']:.3f}",
            flush=True,
        )
        if seed == 42:
            checked = _classify_oof(oof, report)
            labels = checked[PRESENCE_LABEL].astype(int).to_numpy()
            predictions = checked["predicted_listed"].astype(int).to_numpy()
            seed_42 = {
                "roc_auc": record["roc_auc"],
                "pr_auc": record["pr_auc"],
                "macro_f1": record["macro_f1"],
                "classification": _classification(labels, predictions),
                "permutation": _top_features(report),
                "predictions": checked,
            }
    if seed_42 is None:
        raise RuntimeError("seed 42 was not in the evaluation seeds")

    population = full_population(featured)
    listed = int((population[PRESENCE_LABEL] == 1).sum())
    empty = int((population[PRESENCE_LABEL] == 0).sum())
    if listed != 122 or empty != 4000:
        raise RuntimeError(f"500 m all-empty population is {listed} listed and {empty} empty")
    print(f"all-empty run: {listed} listed and {empty} empty", flush=True)
    _oof, full_report = evaluate_forest(population)
    full = {
        "rows": int(full_report["training_rows"]),
        "roc_auc": _mean(full_report, "roc_auc"),
        "pr_auc": _mean(full_report, "pr_auc"),
        "macro_f1": _mean(full_report, "macro_f1"),
        "permutation": _top_features(full_report),
    }
    print(
        f"all-empty ROC-AUC {full['roc_auc']:.3f} PR-AUC {full['pr_auc']:.3f}",
        flush=True,
    )

    roc = _across_draws(
        [{"metrics": {"roc_auc": {"mean": row["roc_auc"]}, "pr_auc": {"mean": row["pr_auc"]}}} for row in draws],
        "roc_auc",
    )
    pr = _across_draws(
        [{"metrics": {"roc_auc": {"mean": row["roc_auc"]}, "pr_auc": {"mean": row["pr_auc"]}}} for row in draws],
        "pr_auc",
    )
    summary = {
        "grid_m": 500,
        "seed_42": {key: value for key, value in seed_42.items() if key != "predictions"},
        "ten_seeds": {"roc_auc": roc, "pr_auc": pr, "draws": draws},
        "all_empty": full,
        "one_km_reference": ONE_KM,
    }
    (OUT / "evaluation_500m.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    seed_42["predictions"][
        [
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
        ["fold", CELL_ID_COLUMN]
    ).to_csv(OUT / "evaluation_500m_predictions.csv", index=False)
    _write_report(summary)
    print(f"wrote {OUT / 'evaluation_500m.md'}", flush=True)


if __name__ == "__main__":
    main()
