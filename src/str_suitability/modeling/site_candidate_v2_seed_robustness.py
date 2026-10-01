"""Seed robustness for the site_candidate_v2 presence comparison.

Each seed redraws the empty rural squares, then compares the six-feature v2
random forest with the three-feature baseline forest under the same
municipality-grouped 5-fold scheme. Results go to a separate file. Stored v2
results are not changed, and no seed is selected as the winner.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from str_suitability import config
from str_suitability.features.road_distance import LAGUNA_ROADS_PATH
from str_suitability.modeling.location_classifier import _group_splits
from str_suitability.modeling.presence import PRESENCE_FEATURES, PRESENCE_LABEL, presence_forest
from str_suitability.modeling.site_candidate_v2 import (
    V2_FEATURES,
    _attach_v2_features,
    _load_location_cells,
    _positive_probability,
    load_places_with_category,
)
from str_suitability.rural.classify import RURAL_CELL

SEEDS = (42, 7, 13, 21, 99, 101, 202, 303, 404, 505)
CSV_FILENAME = "site_candidate_v2_seed_robustness.csv"
REPORT_FILENAME = "site_candidate_v2_seed_robustness.md"


def draw_training_sample(cells: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Balanced 110 positive / 110 negative rural sample redrawn at one seed."""
    rural = cells.loc[cells["cell_class"] == RURAL_CELL].copy()
    eligible = rural.loc[rural["municipality"].notna() & (rural["municipality"].astype(str) != "")]
    listed = eligible.loc[eligible["listings_in_cell"] >= 1]
    pool = eligible.loc[eligible["listings_in_cell"] == 0]
    unlisted = pool.sample(n=len(listed), random_state=seed)
    sample = pd.concat([listed, unlisted], ignore_index=True)
    sample[PRESENCE_LABEL] = (sample["listings_in_cell"] >= 1).astype(int)
    return sample


def _fold_aucs(features: pd.DataFrame, target: pd.Series, folds) -> list[float]:
    aucs = []
    for train_index, test_index in folds:
        model = presence_forest()
        model.fit(features.iloc[train_index], target.iloc[train_index])
        probability = _positive_probability(model, features.iloc[test_index])
        aucs.append(float(roc_auc_score(target.iloc[test_index], probability)))
    return aucs


def evaluate_seed(sample: pd.DataFrame, seed: int) -> dict[str, object]:
    target = sample[PRESENCE_LABEL].astype(int)
    groups = sample["municipality"].astype(str)
    v2_x = sample[list(V2_FEATURES)]
    baseline_x = sample[list(PRESENCE_FEATURES)]
    folds, scheme = _group_splits(baseline_x, target, groups)
    v2 = _fold_aucs(v2_x, target, folds)
    baseline = _fold_aucs(baseline_x, target, folds)
    gains = [left - right for left, right in zip(v2, baseline, strict=True)]
    return {
        "seed": int(seed),
        "scheme": scheme,
        "v2_roc_auc_mean": float(np.mean(v2)),
        "baseline_roc_auc_mean": float(np.mean(baseline)),
        "gain_mean": float(np.mean(gains)),
        "folds_positive_gain": int(sum(value > 0 for value in gains)),
        "n_folds": int(len(gains)),
        "v2_fold_roc_auc": v2,
        "baseline_fold_roc_auc": baseline,
        "gain_fold": gains,
    }


def run_seed_robustness(output_dir: Path | None = None, seeds: tuple[int, ...] = SEEDS) -> pd.DataFrame:
    output_dir = config.RURAL_PROCESSED_DIR if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cells, _place_report = _load_location_cells()
    places, _attraction_report = load_places_with_category(config.LISTED_TOURIST_PLACES_PATH)
    roads = gpd.read_file(LAGUNA_ROADS_PATH)
    cells = _attach_v2_features(cells, places, roads)
    rows = [evaluate_seed(draw_training_sample(cells, seed), seed) for seed in seeds]
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / CSV_FILENAME, index=False)
    (output_dir / REPORT_FILENAME).write_text(summarize_seed_robustness(frame), encoding="utf-8")
    return frame


def summarize_seed_robustness(frame: pd.DataFrame) -> str:
    gains = frame["gain_mean"].to_numpy(dtype=float)
    v2 = frame["v2_roc_auc_mean"].to_numpy(dtype=float)
    lines = [
        "# site_candidate_v2 seed robustness",
        "",
        "Empty rural squares are redrawn at ten seeds. For each seed the six-feature v2 random",
        "forest and the three-feature baseline forest are scored with the same municipality-grouped",
        "5-fold scheme. All seeds are reported, including unfavorable ones. No seed is selected.",
        "",
        f"Seeds: {', '.join(str(int(value)) for value in frame['seed'])}.",
        "",
        f"v2 ROC-AUC mean across seeds: {v2.mean():.3f} (range {v2.min():.3f} to {v2.max():.3f}).",
        f"Mean gain over baseline across seeds: {gains.mean():.3f} "
        f"(range {gains.min():.3f} to {gains.max():.3f}).",
        "",
        "| seed | v2 ROC-AUC | baseline ROC-AUC | gain | folds with positive gain |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.seed} | {row.v2_roc_auc_mean:.3f} | {row.baseline_roc_auc_mean:.3f} | "
            f"{row.gain_mean:.3f} | {row.folds_positive_gain} of {row.n_folds} |"
        )
    lines.append("")
    return "\n".join(lines)
