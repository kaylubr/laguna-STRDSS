import numpy as np
import pandas as pd

from str_suitability.modeling.site_candidate import (
    FEATURES,
    PRESENCE_LABEL,
    build_training_sample,
    evaluate_forest,
    score_rural_cells,
)


def _cell(cell_id: str, cell_class: str, municipality: str, listings: int, value: float) -> dict:
    return {
        "cell_id": cell_id,
        "cell_class": cell_class,
        "municipality": municipality,
        "listings_in_cell": listings,
        **{feature: value for feature in FEATURES},
    }


def test_training_sample_is_balanced_and_rural_with_municipality_only():
    cells = pd.DataFrame(
        [
            _cell("listed-a", "rural", "A", 1, 1.0),
            _cell("listed-b", "rural", "B", 2, 2.0),
            _cell("empty-a", "rural", "A", 0, 3.0),
            _cell("empty-b", "rural", "B", 0, 4.0),
            _cell("empty-c", "rural", "C", 0, 5.0),
            _cell("urban", "urban", "A", 1, 6.0),
            _cell("no-town", "rural", "", 1, 7.0),
        ]
    )

    sample = build_training_sample(cells)
    repeated = build_training_sample(cells)

    assert len(sample) == 4
    assert int(sample[PRESENCE_LABEL].sum()) == 2
    assert int((sample[PRESENCE_LABEL] == 0).sum()) == 2
    assert sample["cell_id"].tolist() == repeated["cell_id"].tolist()
    assert not {"urban", "no-town"}.intersection(sample["cell_id"])


def test_score_frame_scores_named_rural_cells_only():
    sample = pd.DataFrame(
        [
            _cell("listed-a", "rural", "A", 1, 1.0),
            _cell("listed-b", "rural", "B", 1, 2.0),
            _cell("empty-a", "rural", "A", 0, 3.0),
            _cell("empty-b", "rural", "B", 0, 4.0),
        ]
    )
    sample[PRESENCE_LABEL] = [1, 1, 0, 0]
    cells = pd.concat(
        [
            sample,
            pd.DataFrame([_cell("urban", "urban", "A", 0, 5.0)]),
            pd.DataFrame([_cell("unnamed", "rural", "", 0, 6.0)]),
        ],
        ignore_index=True,
    )

    scored = score_rural_cells(sample, cells)

    assert scored["cell_id"].tolist() == sample["cell_id"].tolist()
    assert scored["candidate_pattern_score"].between(0.0, 1.0).all()
    assert scored["in_training_sample"].all()
    assert scored["is_empty"].tolist() == [False, False, True, True]


def test_evaluation_reports_only_the_random_forest(monkeypatch):
    import str_suitability.modeling.site_candidate as site_candidate

    monkeypatch.setattr(site_candidate, "PERMUTATION_REPEATS", 1)
    rows = []
    for index in range(10):
        municipality = f"M{index}"
        rows.extend(
            [
                _cell(f"{municipality}-listed", "rural", municipality, 1, float(index)),
                _cell(f"{municipality}-empty", "rural", municipality, 0, float(index + 1)),
            ]
        )
    sample = pd.DataFrame(rows)
    sample[PRESENCE_LABEL] = [1, 0] * 10

    oof, report = evaluate_forest(sample)

    assert report["model"] == "random_forest"
    assert report["validation_scheme"] == "stratified_group_kfold_5"
    assert set(report["metrics"]) == {"roc_auc", "pr_auc", "macro_f1"}
    assert len(report["folds"]) == 5
    assert oof["random_forest_raw"].notna().all()
    assert len(report["permutation_importance"]) == len(FEATURES)