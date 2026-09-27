import geopandas as gpd
import pandas as pd

from str_suitability.rural.classify import CELL_ID_COLUMN
from str_suitability.suitability.inputs import join_predictions
from str_suitability.suitability.validate import validate_suitability_inputs

PREDICTION_PREFIX = "predicted_"


def predict_rural_cells(
    rural_cells: gpd.GeoDataFrame,
    models: dict[str, object],
    feature_columns: list[str],
) -> dict[str, pd.DataFrame]:
    assert len(rural_cells) > 0, "there are no rural cells to predict"
    assert rural_cells[CELL_ID_COLUMN].is_unique, "the rural prediction domain repeats a cell"

    predictions = {}
    for name, model in models.items():
        column = f"{PREDICTION_PREFIX}{name}"
        values = model.predict(rural_cells[feature_columns])
        assert pd.notna(values).all(), f"the rural {name} model produced a missing prediction"
        predictions[column] = pd.DataFrame({CELL_ID_COLUMN: rural_cells[CELL_ID_COLUMN], column: values})
    return predictions


def rural_scoring_input(
    rural_cells: gpd.GeoDataFrame, predictions: dict[str, pd.DataFrame]
) -> gpd.GeoDataFrame:
    joined = join_predictions(rural_cells, predictions)
    validate_suitability_inputs(joined)
    return joined
