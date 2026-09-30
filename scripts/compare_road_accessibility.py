"""Compare the current five-feature forest with the same forest plus road distance.

This does not refit or replace the map model.
"""

import json

import pandas as pd

from str_suitability.config import PROCESSED_DIR, RURAL_PROCESSED_DIR
from str_suitability.features.road_distance import GRID_ROAD_DISTANCE_PATH, ROAD_DISTANCE_KM
from str_suitability.modeling.experiments import compare_road_accessibility
from str_suitability.modeling.location_classifier import CLASSIFIER_FEATURES
from str_suitability.rural.classify import CELL_ID_COLUMN

LABELED_PATH = RURAL_PROCESSED_DIR / "location_training_cells.parquet"
REPORT_PATH = RURAL_PROCESSED_DIR / "road_accessibility_comparison.json"


def main() -> None:
    labeled = pd.read_parquet(LABELED_PATH)
    distances = pd.read_parquet(GRID_ROAD_DISTANCE_PATH)
    compared = labeled.merge(distances, on=CELL_ID_COLUMN, how="left", validate="one_to_one")
    assert compared[ROAD_DISTANCE_KM].notna().all(), "a labeled cell has no road distance"
    assert list(CLASSIFIER_FEATURES) == [
        "surrounding_mean_revenue",
        "surrounding_mean_occupancy",
        "surrounding_listing_count",
        "listed_places_within_radius",
        "distance_to_listed_tourist_place",
    ]
    report = compare_road_accessibility(compared)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "final_model_changed": report["final_model_changed"],
        "validation": report["validation"],
        "model_a": report["model_a_five_features"]["metrics"],
        "model_b": report["model_b_with_road_distance"]["metrics"],
        "change": report["model_b_minus_model_a"],
        "reading": report["reading"],
    }, indent=2))


if __name__ == "__main__":
    main()
