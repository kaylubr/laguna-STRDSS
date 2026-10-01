import json

import numpy as np
import pandas as pd

from str_suitability.modeling.site_candidate_v2 import V2_FEATURES
from str_suitability.modeling.site_candidate_v2_rf import (
    annotate_report_with_thesis_model,
    rf_score_frame,
    rural_scoring_cells,
)


def _square(cell_id: str, cell_class: str, municipality: str, listings: int) -> dict:
    return {
        "cell_id": cell_id,
        "cell_class": cell_class,
        "municipality": municipality,
        "listings_in_cell": listings,
    }


def test_thesis_model_field_is_added_and_the_stored_selection_is_kept(tmp_path):
    original = {
        "selected_v2_model": "v2_logistic_regression",
        "adopted_for_map": True,
        "metrics": {"v2_random_forest": {"roc_auc": {"mean": 0.7274258287458627}}},
    }
    path = tmp_path / "site_candidate_v2_report.json"
    path.write_text(json.dumps(original), encoding="utf-8")

    updated = annotate_report_with_thesis_model(tmp_path)

    assert updated["thesis_model"]["name"] == "v2_random_forest"
    assert updated["thesis_model"]["stored_selection_kept"] == "v2_logistic_regression"
    assert updated["selected_v2_model"] == "v2_logistic_regression"
    assert updated["metrics"] == original["metrics"]
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["thesis_model"]["scores_file"] == "site_candidate_v2_rf_scores.parquet"


def test_rural_scoring_cells_keep_only_rural_with_a_municipality():
    frame = pd.DataFrame(
        [
            _square("keep", "rural", "A", 0),
            _square("urban", "urban", "A", 0),
            _square("unclassified", "unclassified", "A", 0),
            _square("no-name", "rural", "", 0),
        ]
    )
    frame.loc[3, "municipality"] = np.nan

    kept = rural_scoring_cells(frame)

    assert kept["cell_id"].tolist() == ["keep"]


def test_rf_score_frame_labels_the_random_forest_and_shares_the_v2_features():
    sample = pd.DataFrame({"cell_id": ["a", "b"], "presence": [1, 0]})
    rural = pd.DataFrame(
        {
            "cell_id": ["a", "c"],
            "municipality": ["A", "B"],
            "listings_in_cell": [1, 0],
            **{name: [1.0, 0.0] for name in V2_FEATURES},
        }
    )

    scored = rf_score_frame(sample, rural, np.array([0.9, 0.1]))

    assert set(scored["model_name"]) == {"v2_random_forest"}
    assert set(V2_FEATURES).issubset(scored.columns)
    assert scored["in_locked_training_sample"].tolist() == [True, False]
