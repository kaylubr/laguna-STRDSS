import numpy as np
import pandas as pd
import pytest

from str_suitability.pipeline import FEATURE_COLUMNS
from str_suitability.rural.classify import (
    CELL_ID_COLUMN,
    RURAL_CELL,
    URBAN_CELL,
    classify_grid_cells,
    select_cells_with_class,
)
from str_suitability.rural.prediction import predict_rural_cells, rural_scoring_input


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, features):
        return np.full(len(features), float(self.value))


class BrokenModel:
    def predict(self, features):
        values = np.full(len(features), 1.0)
        values[0] = np.nan
        return values


MODELS = {"revenue": ConstantModel(1000.0), "occupancy": ConstantModel(0.25)}


def rural_domain(make_grid, make_barangays):
    grid = make_grid(12)
    classification, _ = classify_grid_cells(grid, make_barangays(8, 4))
    return grid, classification, select_cells_with_class(grid, classification, RURAL_CELL)


def test_predictions_are_keyed_by_rural_cell(make_grid, make_barangays):
    _, _, rural = rural_domain(make_grid, make_barangays)
    predictions = predict_rural_cells(rural, MODELS, FEATURE_COLUMNS)
    assert set(predictions) == {"predicted_revenue", "predicted_occupancy"}
    for column, frame in predictions.items():
        assert column in frame.columns
        assert frame[CELL_ID_COLUMN].tolist() == rural[CELL_ID_COLUMN].tolist()
        assert len(frame) == 8


def test_every_rural_cell_receives_a_prediction(make_grid, make_barangays):
    _, _, rural = rural_domain(make_grid, make_barangays)
    predictions = predict_rural_cells(rural, MODELS, FEATURE_COLUMNS)
    assert predictions["predicted_revenue"]["predicted_revenue"].notna().all()
    assert predictions["predicted_occupancy"]["predicted_occupancy"].notna().all()
    assert set(predictions["predicted_revenue"]["predicted_revenue"]) == {1000.0}


def test_urban_cells_are_not_in_the_prediction_domain(make_grid, make_barangays):
    grid, classification, rural = rural_domain(make_grid, make_barangays)
    urban = select_cells_with_class(grid, classification, URBAN_CELL)
    assert len(urban) == 4
    assert not set(urban[CELL_ID_COLUMN]) & set(rural[CELL_ID_COLUMN])
    predictions = predict_rural_cells(rural, MODELS, FEATURE_COLUMNS)
    assert len(predictions["predicted_revenue"]) == len(grid) - len(urban)


def test_prediction_needs_a_non_empty_domain(make_grid, make_barangays):
    grid, _, _ = rural_domain(make_grid, make_barangays)
    with pytest.raises(AssertionError, match="no rural cells to predict"):
        predict_rural_cells(grid.iloc[0:0], MODELS, FEATURE_COLUMNS)


def test_a_missing_prediction_is_rejected(make_grid, make_barangays):
    _, _, rural = rural_domain(make_grid, make_barangays)
    with pytest.raises(AssertionError, match="missing prediction"):
        predict_rural_cells(rural, {"revenue": BrokenModel()}, FEATURE_COLUMNS)


def test_scoring_input_joins_predictions_and_validates_indicators(make_grid, make_barangays):
    _, _, rural = rural_domain(make_grid, make_barangays)
    predictions = predict_rural_cells(rural, MODELS, FEATURE_COLUMNS)
    joined = rural_scoring_input(rural, predictions)
    assert len(joined) == len(rural)
    assert joined["predicted_revenue"].notna().all()
    assert joined["predicted_occupancy"].notna().all()
    assert set(joined[CELL_ID_COLUMN]) == set(rural[CELL_ID_COLUMN])


def test_scoring_input_rejects_predictions_for_other_cells(make_grid, make_barangays):
    _, _, rural = rural_domain(make_grid, make_barangays)
    predictions = predict_rural_cells(rural, MODELS, FEATURE_COLUMNS)
    tampered = {name: frame.copy() for name, frame in predictions.items()}
    tampered["predicted_revenue"].loc[0, CELL_ID_COLUMN] = "r9999c9999"
    with pytest.raises(AssertionError, match="different set of cells"):
        rural_scoring_input(rural, tampered)
