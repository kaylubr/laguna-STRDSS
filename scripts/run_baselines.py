"""Baselines on the same folds and the same empty-cell draws as the main forest.

Writes only under results/baselines/. The main six-feature forest is scored
first. The run stops if seed 42 is not 0.727 or the ten-draw mean is not 0.695.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from str_suitability import config
from str_suitability.modeling.site_candidate import (
    FEATURES,
    FOLDS,
    PRESENCE_LABEL,
    build_training_sample,
    load_cells,
    random_forest,
)
from str_suitability.rural.classify import RURAL_CELL

OUT = config.PROJECT_ROOT / "results" / "baselines"
SEEDS = (7, 13, 21, 42, 99, 101, 202, 303, 404, 505)
MAJOR = "distance_to_major_road_km"
ROAD = "road_distance_km"
ROADS = (ROAD, MAJOR)
PUBLISHED_SEED_42 = 0.7274258287458627
PUBLISHED_TEN = 0.6946731118128164
TOLERANCE = 1e-9


def _population(cells: pd.DataFrame) -> pd.DataFrame:
    rural = cells.loc[cells["cell_class"].eq(RURAL_CELL)].copy()
    eligible = rural.loc[rural["municipality"].notna() & rural["municipality"].astype(str).ne("")]
    sample = eligible.reset_index(drop=True)
    sample[PRESENCE_LABEL] = (sample["listings_in_cell"] >= 1).astype(int)
    return sample


def _splits(sample: pd.DataFrame):
    labels = sample[PRESENCE_LABEL].astype(int)
    groups = sample["municipality"].astype(str)
    features = sample[list(FEATURES)]
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    return list(splitter.split(features, labels, groups))


def _positive_scores(model, frame: pd.DataFrame) -> np.ndarray:
    class_index = list(model.classes_).index(1)
    return model.predict_proba(frame)[:, class_index]


def _logreg():
    return make_pipeline(
        FunctionTransformer(np.log1p),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )


def _fold_metrics(sample: pd.DataFrame, train_index, test_index, score_test) -> dict[str, float]:
    labels = sample.iloc[test_index][PRESENCE_LABEL].astype(int).to_numpy()
    if len(np.unique(labels)) < 2:
        raise RuntimeError("a held-out fold has only one class")
    scores = score_test(sample.iloc[train_index], sample.iloc[test_index])
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def _mean_over_folds(sample: pd.DataFrame, score_test) -> dict[str, float]:
    folds = [_fold_metrics(sample, train_index, test_index, score_test) for train_index, test_index in _splits(sample)]
    return {
        "roc_auc": float(np.mean([row["roc_auc"] for row in folds])),
        "pr_auc": float(np.mean([row["pr_auc"] for row in folds])),
    }


def _ranker(column: str):
    def score_test(_train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        return -test[column].to_numpy(dtype=float)

    return score_test


def _fitted(make_model, columns: tuple[str, ...]):
    def score_test(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        model = make_model()
        model.fit(train[list(columns)], train[PRESENCE_LABEL].astype(int))
        return _positive_scores(model, test[list(columns)])

    return score_test


MODELS = (
    ("rank_nearest_major_road", _ranker(MAJOR)),
    ("rank_nearest_road", _ranker(ROAD)),
    ("logreg_six_features", _fitted(_logreg, FEATURES)),
    ("rf_roads_only", _fitted(random_forest, ROADS)),
    ("rf_six_features", _fitted(random_forest, FEATURES)),
)


def _evaluate_reference(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    score_test = _fitted(random_forest, FEATURES)
    for seed, sample in samples.items():
        metrics = _mean_over_folds(sample, score_test)
        print(f"  rf_six_features {seed}: ROC-AUC {metrics['roc_auc']:.6f}", flush=True)
        rows.append({"model": "rf_six_features", "run": str(seed), **metrics})
    return pd.DataFrame(rows)


def _gate(reference: pd.DataFrame) -> list[str]:
    ten = reference.loc[reference["run"] != "all", "roc_auc"]
    seed_42 = float(reference.loc[reference["run"].eq("42"), "roc_auc"].iloc[0])
    mean = float(ten.mean())
    problems = []
    if abs(seed_42 - PUBLISHED_SEED_42) > TOLERANCE:
        problems.append(f"seed 42 ROC-AUC is {seed_42:.12f}, published {PUBLISHED_SEED_42:.12f}")
    if abs(mean - PUBLISHED_TEN) > TOLERANCE:
        problems.append(f"ten-draw mean ROC-AUC is {mean:.12f}, published {PUBLISHED_TEN:.12f}")
    return problems


def _summary(runs: pd.DataFrame) -> pd.DataFrame:
    ten = runs.loc[runs["run"] != "all"]
    roc = ten.pivot(index="run", columns="model", values="roc_auc")
    reference = roc["rf_six_features"]
    rows = []
    for model in [name for name, _score in MODELS]:
        values = ten.loc[ten["model"].eq(model), "roc_auc"]
        difference = roc[model] - reference
        all_empty = runs.loc[runs["run"].eq("all") & runs["model"].eq(model), "roc_auc"]
        pr = ten.loc[ten["model"].eq(model), "pr_auc"]
        rows.append(
            {
                "model": model,
                "roc_auc_mean": float(values.mean()),
                "roc_auc_lowest": float(values.min()),
                "roc_auc_highest": float(values.max()),
                "all_empty_roc_auc": float(all_empty.iloc[0]),
                "pr_auc_mean": float(pr.mean()),
                "diff_vs_rf_mean": float(difference.mean()),
                "diff_vs_rf_sd": float(difference.std(ddof=1)),
                "seeds_above_rf": int((difference > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells, _places = load_cells()
    if (cells[list(FEATURES)] < 0).any().any():
        raise RuntimeError("log1p requires every feature to be non-negative")
    samples = {str(seed): build_training_sample(cells, negative_seed=seed) for seed in SEEDS}
    samples["all"] = _population(cells)
    listed = int(samples["all"][PRESENCE_LABEL].sum())
    empty = int((samples["all"][PRESENCE_LABEL] == 0).sum())
    if listed != 110 or empty != 931:
        raise SystemExit(f"original grid has {listed} listed and {empty} empty cells, not 110 and 931")

    print("main forest", flush=True)
    reference = _evaluate_reference(samples)
    problems = _gate(reference)
    if problems:
        print("the main forest did not reproduce the published ROC-AUC:", flush=True)
        for problem in problems:
            print(f"  {problem}", flush=True)
        raise SystemExit(1)
    print("reproduction gate passed", flush=True)

    rows = reference.to_dict("records")
    for name, score_test in MODELS:
        if name == "rf_six_features":
            continue
        for seed, sample in samples.items():
            metrics = _mean_over_folds(sample, score_test)
            print(f"  {name} {seed}: ROC-AUC {metrics['roc_auc']:.6f}", flush=True)
            rows.append({"model": name, "run": str(seed), **metrics})

    runs = pd.DataFrame(rows)
    summary = _summary(runs)
    runs.to_csv(OUT / "baseline_runs.csv", index=False)
    summary.to_csv(OUT / "baseline_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"wrote {OUT / 'baseline_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
