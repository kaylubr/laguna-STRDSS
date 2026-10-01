"""Random-forest full-grid scores and the thesis-model annotation for site_candidate_v2.

The six-feature random forest is treated as the thesis model because the study's
algorithm is a random forest. The stored v2 comparison keeps its own selection
(the logistic regression, which has the higher mean ROC-AUC) visible. This module
writes a separate score file and adds the ``thesis_model`` field to the report. It
does not overwrite any logistic output and it does not retune anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from str_suitability import config
from str_suitability.features.road_distance import LAGUNA_ROADS_PATH
from str_suitability.modeling.presence import PRESENCE_LABEL
from str_suitability.modeling.site_candidate_v2 import (
    FEATURE_VERSION,
    V2_FEATURES,
    _attach_v2_features,
    _load_location_cells,
    _locked_training_rows,
    _positive_probability,
    _score_bands,
    load_places_with_category,
    presence_forest,
    require_observed_values,
)
from str_suitability.rural.classify import CELL_ID_COLUMN, RURAL_CELL

SCORES_FILENAME = "site_candidate_v2_rf_scores.parquet"
REPORT_FILENAME = "site_candidate_v2_report.json"
MODEL_NAME = "v2_random_forest"
PRECONDITION = "random forest is unscaled; StandardScaler is not applied"
THESIS_MODEL_REASON = (
    "The study's algorithm is a random forest, so the random forest is the thesis model. "
    "The stored selected_v2_model, the logistic regression with the higher mean ROC-AUC, is kept visible."
)


def rural_scoring_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """Cells the v2 model scores: rural, with a municipality."""
    return cells.loc[
        (cells["cell_class"] == RURAL_CELL)
        & cells["municipality"].notna()
        & (cells["municipality"].astype(str) != "")
    ].copy()


def rf_score_frame(sample: pd.DataFrame, rural: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """The score grid written for the random forest, shaped like the logistic grid."""
    training_ids = set(sample[CELL_ID_COLUMN].astype(str))
    scored = pd.DataFrame(
        {
            CELL_ID_COLUMN: rural[CELL_ID_COLUMN].astype(str).to_numpy(),
            "municipality": rural["municipality"].astype(str).to_numpy(),
            "listings_in_cell_descriptive": rural["listings_in_cell"].to_numpy(),
            "zero_current_listings": rural["listings_in_cell"].to_numpy() == 0,
            "in_locked_training_sample": rural[CELL_ID_COLUMN].astype(str).isin(training_ids).to_numpy(),
            "candidate_pattern_score": scores,
            "display_band": _score_bands(pd.Series(scores)).to_numpy(),
            "model_name": MODEL_NAME,
            "feature_version": FEATURE_VERSION,
            "training_rows": int(len(sample)),
            "training_positives": int((sample[PRESENCE_LABEL] == 1).sum()),
            "training_negatives": int((sample[PRESENCE_LABEL] == 0).sum()),
            "random_state": int(config.RANDOM_STATE),
            "preprocessing": PRECONDITION,
        }
    )
    for column in V2_FEATURES:
        scored[column] = rural[column].to_numpy()
    return scored


def write_random_forest_scores(output_dir: Path | None = None) -> pd.DataFrame:
    """Fit the v2 random forest on all 220 locked rows and score the full rural grid."""
    output_dir = config.RURAL_PROCESSED_DIR if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cells, _place_report = _load_location_cells()
    places, _attraction_report = load_places_with_category(config.LISTED_TOURIST_PLACES_PATH)
    roads = gpd.read_file(LAGUNA_ROADS_PATH)
    cells = _attach_v2_features(cells, places, roads)
    sample = _locked_training_rows(cells)
    require_observed_values(sample, V2_FEATURES)
    model = presence_forest()
    model.fit(sample[list(V2_FEATURES)], sample[PRESENCE_LABEL].astype(int))
    rural = rural_scoring_cells(cells)
    require_observed_values(rural, V2_FEATURES)
    scores = _positive_probability(model, rural[list(V2_FEATURES)])
    scored = rf_score_frame(sample, rural, scores)
    scored.to_parquet(output_dir / SCORES_FILENAME, index=False)
    return scored


def annotate_report_with_thesis_model(output_dir: Path | None = None) -> dict:
    """Add the thesis_model field. The stored selection and every metric are left as is."""
    output_dir = config.RURAL_PROCESSED_DIR if output_dir is None else output_dir
    path = output_dir / REPORT_FILENAME
    report = json.loads(path.read_text(encoding="utf-8"))
    report["thesis_model"] = {
        "name": MODEL_NAME,
        "reason": THESIS_MODEL_REASON,
        "scores_file": SCORES_FILENAME,
        "stored_selection_kept": report.get("selected_v2_model"),
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
