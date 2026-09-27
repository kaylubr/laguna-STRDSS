import geopandas as gpd
import pandas as pd

from str_suitability.audit.density import (
    DENSITY_PREFIX,
    read_grid,
    rural_cell_ids,
    rural_rows,
    tag_group_density,
)
from str_suitability.audit.poi_tags import (
    ACCOMMODATION_VALUES,
    VISITOR_AMENITY_VALUES,
    tag_value_mask,
)
from str_suitability.config import SUITABILITY_INDICATOR_DIRECTIONS
from str_suitability.rural.classify import CELL_ID_COLUMN
from str_suitability.suitability.entropy_weights import WEIGHT_SUM_TOLERANCE, weights_from_normalized
from str_suitability.suitability.inputs import join_predictions
from str_suitability.suitability.normalize import (
    POSITIVE_DIRECTION,
    indicator_from_normalized,
    minimum_maximum_normalize,
)
from str_suitability.taxonomy import PoiCategory

POI_FILENAME = "laguna_pois.parquet"
POI_INDICATOR = "poi_density_total"
PREDICTED_COLUMNS = ("predicted_revenue", "predicted_occupancy")
ACCOMMODATION_COLUMN = f"{DENSITY_PREFIX}accommodation"
VISITOR_AMENITY_COLUMN = f"{DENSITY_PREFIX}visitor_amenity"

POI_CATEGORY_NAMES = tuple(str(category) for category in PoiCategory)
NON_POI_INDICATORS = tuple(
    indicator for indicator in SUITABILITY_INDICATOR_DIRECTIONS if indicator != POI_INDICATOR
)


def build_rural_scoring_frame(processed_dir, rural_dir, feature_dir) -> gpd.GeoDataFrame:
    grid = read_grid(processed_dir)
    pois = gpd.read_parquet(feature_dir / POI_FILENAME)
    groups = tag_group_density(
        grid,
        pois,
        {
            "accommodation": tag_value_mask(pois, "tourism", ACCOMMODATION_VALUES),
            "visitor_amenity": tag_value_mask(pois, "tourism", VISITOR_AMENITY_VALUES),
        },
    )
    grid = grid.merge(groups, on=CELL_ID_COLUMN, how="left", validate="one_to_one")
    rural = rural_rows(grid, rural_cell_ids(rural_dir))
    predictions = {
        column: pd.read_parquet(rural_dir / f"{column}.parquet") for column in PREDICTED_COLUMNS
    }
    return join_predictions(rural, predictions)


def weights_for_directions(frame: pd.DataFrame, directions: dict[str, str]) -> pd.Series:
    normalized = minimum_maximum_normalize(frame, directions)
    weights = weights_from_normalized(normalized)
    assert abs(float(weights.sum()) - 1.0) <= WEIGHT_SUM_TOLERANCE, (
        f"weights sum to {weights.sum()} instead of 1"
    )
    return pd.Series(
        {indicator_from_normalized(column): float(weight) for column, weight in weights.items()}
    )


def weights_with_poi_indicator(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    modified = frame.copy()
    modified[POI_INDICATOR] = values.astype(float).to_numpy()
    return weights_for_directions(modified, SUITABILITY_INDICATOR_DIRECTIONS)


def visitor_density(frame: pd.DataFrame) -> pd.Series:
    return (
        frame[f"{DENSITY_PREFIX}tourist_attraction"]
        + frame[f"{DENSITY_PREFIX}restaurants"]
        + frame[f"{DENSITY_PREFIX}recreation"]
        + frame[ACCOMMODATION_COLUMN]
    )


def poi_scenarios(frame: pd.DataFrame) -> dict[str, pd.Series]:
    values = {
        "A all POIs": frame[POI_INDICATOR],
        "B visitor-relevant only": visitor_density(frame),
        "C other_facilities removed": frame[POI_INDICATOR] - frame[f"{DENSITY_PREFIX}other_facilities"],
    }
    for category in POI_CATEGORY_NAMES:
        values[f"D {category} alone"] = frame[f"{DENSITY_PREFIX}{category}"]
    return values


def split_indicator_directions() -> dict[str, str]:
    directions = {
        indicator: SUITABILITY_INDICATOR_DIRECTIONS[indicator] for indicator in NON_POI_INDICATORS
    }
    directions.update({f"{DENSITY_PREFIX}{category}": POSITIVE_DIRECTION for category in POI_CATEGORY_NAMES})
    return directions


def ewm_scenarios(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    records = []
    degenerate = []
    for label, values in poi_scenarios(frame).items():
        if float(values.max()) == float(values.min()):
            degenerate.append(label)
            records.append({"scenario": label, "poi_weight": None, "poi_weight_change": None})
            continue
        weights = weights_with_poi_indicator(frame, values)
        records.append({"scenario": label, "poi_weight": float(weights[POI_INDICATOR])})

    table = pd.DataFrame(records)
    baseline = table.loc[table["scenario"] == "A all POIs", "poi_weight"].iloc[0]
    table["poi_weight_change"] = table["poi_weight"] - baseline

    split_weights = weights_for_directions(frame, split_indicator_directions())
    split_bloc = float(
        split_weights[[f"{DENSITY_PREFIX}{category}" for category in POI_CATEGORY_NAMES]].sum()
    )
    report = {
        "baseline_poi_weight": float(baseline),
        "degenerate_scenarios": degenerate,
        "split_indicator_count": len(split_indicator_directions()),
        "split_poi_bloc_weight": split_bloc,
        "split_weights": {name: float(weight) for name, weight in split_weights.items()},
    }
    return table, report
