import pandas as pd

from str_suitability.modeling.site_candidate_v2 import V2_FEATURES
from str_suitability.modeling.site_candidate_v2_seed_robustness import (
    draw_training_sample,
    evaluate_seed,
)


def _square(cell_id: str, municipality: str, listings: int) -> dict:
    return {
        "cell_id": cell_id,
        "cell_class": "rural",
        "municipality": municipality,
        "listings_in_cell": listings,
        "listed_places_within_radius": 1,
        "distance_to_listed_tourist_place": 2.0,
        "road_distance_km": 0.2,
        "distance_to_nearest_town_center_km": 3.0,
        "distance_to_major_road_km": 1.0,
        "attraction_count_within_5km": 1,
    }


def test_seed_robustness_redraws_negatives_per_seed():
    rows = [_square(f"listed-{name}", name, 1) for name in "ABCD"]
    rows += [_square(f"empty-{name}-{index}", name, 0) for name in "ABCD" for index in range(4)]
    frame = pd.DataFrame(rows)

    first = draw_training_sample(frame, 42)
    again = draw_training_sample(frame, 42)
    other = draw_training_sample(frame, 7)

    assert int(first["presence"].sum()) == 4
    assert int((first["presence"] == 0).sum()) == 4
    assert first["cell_id"].tolist() == again["cell_id"].tolist()
    negative_sets = {
        tuple(sorted(frame_set.loc[frame_set["presence"] == 0, "cell_id"]))
        for frame_set in (first, other, draw_training_sample(frame, 13))
    }
    assert len(negative_sets) > 1


def test_evaluate_seed_reports_gain_and_fold_count():
    rows = []
    for name in ("A", "B", "C", "D", "E", "F"):
        for listings, suffix in ((1, "yes"), (0, "no")):
            rows.append(_square(f"{name}-{suffix}", name, listings))
    sample = pd.DataFrame(rows)
    sample["presence"] = (sample["listings_in_cell"] >= 1).astype(int)

    result = evaluate_seed(sample, 42)

    assert result["seed"] == 42
    assert result["scheme"].startswith("stratified_group_kfold")
    assert result["n_folds"] >= 2
    assert 0.0 <= result["v2_roc_auc_mean"] <= 1.0
    assert -1.0 <= result["gain_mean"] <= 1.0
    assert 0 <= result["folds_positive_gain"] <= result["n_folds"]
    assert set(V2_FEATURES).issubset(sample.columns)
