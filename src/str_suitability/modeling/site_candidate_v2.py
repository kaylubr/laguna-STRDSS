"""Candidate-site pattern model, version 2.

The score ranks rural 1 km cells by similarity to the observable location
conditions of rural cells where an active short-term rental is already
observed. It is a screening measure. It is not a forecast of revenue,
occupancy, profitability, investment return, regulatory suitability, legality,
utilities, flood safety, or building feasibility.

The feature list below was fixed before any outer-fold result was inspected.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from str_suitability import config
from str_suitability.features.context import POBLACION_DISTANCE_COLUMN
from str_suitability.features.road_distance import (
    DISTANCE_CRS,
    GRID_ROAD_DISTANCE_PATH,
    LAGUNA_ROADS_PATH,
    ROAD_DISTANCE_KM,
    _all_mapped_roads,
    _assert_metre_crs,
)
from str_suitability.modeling.location_classifier import _group_splits
from str_suitability.modeling.presence import (
    PRESENCE_FEATURES,
    PRESENCE_LABEL,
    evaluate_presence,
    listed_places_for_training,
    presence_forest,
    refresh_listed_places,
)
from str_suitability.rural.classify import CELL_ID_COLUMN, RURAL_CELL
from str_suitability.rural.site_score import (
    DISTANCE_COLUMN,
    LATITUDE_COLUMN,
    LISTED_PLACES_WITHIN_RADIUS,
    LONGITUDE_COLUMN,
)
from str_suitability.spatial.haversine import haversine_km

FEATURE_VERSION = "site_candidate_v2"
ATTRACTION_COUNT = "attraction_count_within_5km"
TOWN_CENTER_KM = "distance_to_nearest_town_center_km"
MAJOR_ROAD_KM = "distance_to_major_road_km"
MAJOR_ROAD_CLASS = "major"
ATTRACTION_CATEGORIES = ("Falls", "Mountains")
SPEARMAN_FLAG = 0.85
BASELINE_METRIC_TOLERANCE = 0.0015
DOCUMENTED_BASELINE = {
    "roc_auc": 0.634,
    "accuracy": 0.575,
    "macro_f1": 0.570,
}
BOOTSTRAP_DRAWS = 1000
INNER_FOLDS = 3
DISPLAY_LOW = 0.40
DISPLAY_HIGH = 0.60
YOUDEN_NOTE = (
    "Binary operating metrics use Youden's J chosen on inner grouped scores "
    "from the outer training municipalities only. The cut is not a business "
    "suitability threshold. The published baseline accuracy and macro F1 "
    "remain the existing screen's 0.5 class cut, replicated separately."
)

V2_FEATURES = (
    LISTED_PLACES_WITHIN_RADIUS,
    DISTANCE_COLUMN,
    ROAD_DISTANCE_KM,
    TOWN_CENTER_KM,
    MAJOR_ROAD_KM,
    ATTRACTION_COUNT,
)

FORBIDDEN_FEATURE_COLUMNS = frozenset(
    {
        "listings_in_cell",
        "presence",
        "ttm_revenue",
        "ttm_occupancy",
        "ttm_average_daily_rate",
        "price",
        "bedrooms",
        "bathrooms",
        "rating",
        "reviews",
        "review_count",
        "host_id",
        "superhost",
        "surrounding_mean_revenue",
        "surrounding_mean_occupancy",
        "surrounding_median_revenue",
        "surrounding_median_occupancy",
        "surrounding_listing_count",
        "active_listings_within_radius",
        "nearby_mean_revenue",
        "nearby_mean_occupancy",
        "listings_nearby",
        "competition_listing_count",
    }
)

OMITTED_FEATURES = (
    {
        "feature": "travel_time_to_nearest_town_center_min",
        "reason": "No validated routable road graph or speed table is in the repository. Straight-line major-road distance is used instead.",
    },
    {
        "feature": "distance_to_transport_access_km",
        "reason": "No reliable transport-terminal or port point layer is in the repository. In-cell bus-stop counts are not a substitute.",
    },
    {
        "feature": "tourism_poi_count_within_5km",
        "reason": "The same count is already the baseline feature listed_places_within_radius. It is not duplicated under a second name.",
    },
    {
        "feature": "food_service_poi_count_within_5km",
        "reason": "No validated food-service point layer is in the repository.",
    },
    {
        "feature": "distance_to_nearest_tourism_poi_km",
        "reason": "The same distance is already the baseline feature distance_to_listed_tourist_place.",
    },
    {
        "feature": "mean_slope_pct",
        "reason": "No elevation or slope raster is in the repository.",
    },
    {
        "feature": "mean_slope_degrees",
        "reason": "No elevation or slope raster is in the repository.",
    },
    {
        "feature": "elevation_m",
        "reason": "No elevation raster is in the repository.",
    },
    {
        "feature": "terrain_ruggedness",
        "reason": "No elevation raster is in the repository.",
    },
    {
        "feature": "built_up_share",
        "reason": "No validated non-Airbnb built-up or land-use layer is in the repository.",
    },
)


def assert_feature_matrix(columns: list[str] | tuple[str, ...]) -> None:
    """Fail when a forbidden Airbnb-derived column is offered as a feature."""
    leaked = sorted(set(columns) & FORBIDDEN_FEATURE_COLUMNS)
    assert not leaked, f"forbidden leakage columns entered the feature matrix: {leaked}"
    assert list(columns) == list(V2_FEATURES) or list(columns) == list(PRESENCE_FEATURES), (
        f"the feature matrix is not the fixed baseline or the fixed v2 list: {list(columns)}"
    )


def dedupe_places(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep the first row of each non-empty place_id. Blank place ids stay."""
    working = frame.copy()
    working[LONGITUDE_COLUMN] = pd.to_numeric(working.get(LONGITUDE_COLUMN), errors="coerce")
    working[LATITUDE_COLUMN] = pd.to_numeric(working.get(LATITUDE_COLUMN), errors="coerce")
    located = working.dropna(subset=[LONGITUDE_COLUMN, LATITUDE_COLUMN]).copy()
    place_id = located.get("place_id", pd.Series("", index=located.index)).fillna("").astype(str).str.strip()
    identified = located.loc[place_id.ne("")].copy()
    identified["_place_id"] = place_id.loc[place_id.ne("")].to_numpy()
    identified = identified.drop_duplicates("_place_id", keep="first")
    unnamed = located.loc[place_id.eq("")]
    places = pd.concat([identified, unnamed], ignore_index=True)
    if "category" not in places.columns:
        places["category"] = ""
    places["category"] = places["category"].fillna("").astype(str)
    report = {
        "records": int(len(frame)),
        "without_coordinates": int(len(frame) - len(located)),
        "duplicate_place_ids_removed": int(place_id.ne("").sum() - identified["_place_id"].nunique())
        if len(identified)
        else int(place_id.ne("").sum()),
        "places_used": int(len(places)),
    }
    return places, report


def load_places_with_category(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}. Expected a JSON list with latitude, longitude, place_id, name, and category."
        )
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{path} is not a JSON list of places")
    places, report = dedupe_places(pd.DataFrame(records))
    required = {"Falls", "Mountains"}
    present = set(places["category"])
    missing = sorted(required - present)
    if missing:
        raise ValueError(
            f"{path} has no places in categories {missing}. "
            "attraction_count_within_5km is not replaced with another source."
        )
    return places, report


def attraction_count_within_radius(cells: pd.DataFrame, places: pd.DataFrame) -> pd.Series:
    """Count Falls and Mountains within 5 km. Zero means none in range, not a missing file."""
    attractions = places.loc[places["category"].isin(ATTRACTION_CATEGORIES)]
    if len(attractions) == 0:
        return pd.Series(np.nan, index=cells.index, name=ATTRACTION_COUNT)
    distances = haversine_km(
        cells[LONGITUDE_COLUMN].to_numpy()[:, None],
        cells[LATITUDE_COLUMN].to_numpy()[:, None],
        attractions[LONGITUDE_COLUMN].to_numpy()[None, :],
        attractions[LATITUDE_COLUMN].to_numpy()[None, :],
    )
    counts = (distances <= config.NEIGHBORHOOD_RADIUS_KM).sum(axis=1).astype(int)
    return pd.Series(counts, index=cells.index, name=ATTRACTION_COUNT)


def nearest_major_road_km(grid: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> pd.Series:
    """Straight-line kilometres from the cell centroid to the nearest road_class major segment.

    Measured in EPSG:32651 metres and divided by 1000. This is not travel time.
    An empty major-road set returns missing values rather than zero.
    """
    if "road_class" not in roads.columns:
        raise ValueError(
            f"{LAGUNA_ROADS_PATH} has no road_class field. "
            "Expected properties.road_class with value 'major'. No substitute road source is used."
        )
    assert grid.crs is not None, "the grid has no CRS"
    assert roads.crs is not None, "the road network has no CRS"
    major = roads.loc[roads["road_class"].astype(str).eq(MAJOR_ROAD_CLASS)].copy()
    major = major.loc[~major.geometry.is_empty & major.geometry.notna()]
    if len(major) == 0:
        return pd.Series(np.nan, index=grid.index, name=MAJOR_ROAD_KM)
    projected_grid = grid.to_crs(DISTANCE_CRS)
    projected_roads = _all_mapped_roads(major).to_crs(DISTANCE_CRS)
    _assert_metre_crs(projected_grid)
    _assert_metre_crs(projected_roads)
    centroids = gpd.GeoDataFrame(
        {CELL_ID_COLUMN: projected_grid[CELL_ID_COLUMN].astype(str).to_numpy()},
        geometry=projected_grid.geometry.centroid,
        crs=DISTANCE_CRS,
    )
    joined = centroids.sjoin_nearest(projected_roads[["geometry"]], how="left", distance_col="major_road_distance_m")
    joined = joined.sort_values("major_road_distance_m").drop_duplicates(CELL_ID_COLUMN, keep="first")
    distances = centroids[[CELL_ID_COLUMN]].merge(
        joined[[CELL_ID_COLUMN, "major_road_distance_m"]],
        on=CELL_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    kilometres = distances["major_road_distance_m"] / 1000.0
    by_id = pd.Series(kilometres.to_numpy(), index=distances[CELL_ID_COLUMN].astype(str), name=MAJOR_ROAD_KM)
    return pd.Series(grid[CELL_ID_COLUMN].astype(str).map(by_id).to_numpy(), index=grid.index, name=MAJOR_ROAD_KM)


def require_observed_values(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> None:
    """Missing external values stay missing. They are not filled with a constant."""
    for column in columns:
        missing = int(frame[column].isna().sum())
        assert missing == 0, (
            f"{column} is missing on {missing} rows. "
            "The pipeline does not impute that gap with a constant."
        )


def research_use_decision(deltas: list[float], v2_grid_complete: bool, baseline_grid_complete: bool) -> str:
    """Apply the pre-specified comparison rule after every outer fold has been scored."""
    mean_delta = float(np.mean(deltas))
    positive_folds = int(sum(float(delta) > 0 for delta in deltas))
    consistent = len(deltas) == 5 and mean_delta > 0 and positive_folds >= 4
    if consistent and v2_grid_complete:
        return "deploy v2 as a separate layer"
    if baseline_grid_complete:
        return "retain baseline"
    return "do not deploy either beyond research use"


def _positive_probability(model, features: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(features)
    column = list(model.classes_).index(1)
    return raw[:, column]


class ScaledLogisticRegression:
    """Scaler and logistic regression, both fitted on the rows passed to fit."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=config.RANDOM_STATE,
        )

    def fit(self, features: pd.DataFrame, target: pd.Series):
        self.model.fit(self.scaler.fit_transform(features), target)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(features))


def _youden_threshold(observed: np.ndarray, scores: np.ndarray) -> float:
    _fpr, _tpr, thresholds = roc_curve(observed, scores)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    j = _tpr[finite] - _fpr[finite]
    return float(thresholds[finite][int(np.argmax(j))])


def _inner_oof(model_factory, features: pd.DataFrame, target: pd.Series, groups: pd.Series) -> np.ndarray | None:
    n_groups = int(groups.nunique())
    splits = min(INNER_FOLDS, n_groups)
    if splits < 2:
        return None
    folder = GroupKFold(n_splits=splits)
    oof = np.full(len(features), np.nan)
    for train_index, test_index in folder.split(features, target, groups):
        model = model_factory()
        model.fit(features.iloc[train_index], target.iloc[train_index])
        oof[test_index] = _positive_probability(model, features.iloc[test_index])
    if not np.isfinite(oof).all():
        return None
    return oof


def _operating_metrics(observed: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    predicted = (scores >= threshold).astype(int)
    labels = [0, 1]
    matrix = confusion_matrix(observed, predicted, labels=labels)
    return {
        "threshold": float(threshold),
        "macro_f1": float(f1_score(observed, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(observed, predicted)),
        "precision": float(precision_score(observed, predicted, zero_division=0)),
        "recall": float(recall_score(observed, predicted, zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _threshold_free(observed: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    if len(set(observed.tolist())) < 2 or not np.isfinite(scores).all():
        return {"roc_auc": None, "pr_auc": None, "brier": None}
    return {
        "roc_auc": float(roc_auc_score(observed, scores)),
        "pr_auc": float(average_precision_score(observed, scores)),
        "brier": float(brier_score_loss(observed, scores)),
    }


def _mean_std(values: list[float | None]) -> dict[str, float | int | None]:
    usable = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not usable:
        return {"mean": None, "std": None, "n_folds": 0, "values": []}
    std = float(np.std(usable, ddof=1)) if len(usable) > 1 else 0.0
    return {"mean": float(np.mean(usable)), "std": std, "n_folds": len(usable), "values": usable}


def _permutation_auc_drop(model, features: pd.DataFrame, observed: np.ndarray, seed: int) -> list[float]:
    """Held-out drop in ROC-AUC after shuffling one column. Not a causal effect."""
    baseline = float(roc_auc_score(observed, _positive_probability(model, features)))
    rng = np.random.default_rng(seed)
    drops = []
    for column in features.columns:
        repeated = []
        values = features[column].to_numpy().copy()
        for _repeat in range(20):
            shuffled = features.copy()
            shuffled[column] = rng.permutation(values)
            repeated.append(baseline - float(roc_auc_score(observed, _positive_probability(model, shuffled))))
        drops.append(float(np.mean(repeated)))
    return drops


def _spearman_flags(features: pd.DataFrame) -> list[dict[str, object]]:
    matrix = features.corr(method="spearman")
    flags = []
    names = list(features.columns)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            value = matrix.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= SPEARMAN_FLAG:
                flags.append({"left": left, "right": right, "spearman": float(value)})
    return flags


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _coverage(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[dict[str, object]]:
    rows = []
    total = int(len(frame))
    for column in columns:
        present = int(frame[column].notna().sum())
        rows.append(
            {
                "feature": column,
                "rows": total,
                "computed": present,
                "missing": total - present,
                "percent_computed": float(present / total) if total else None,
            }
        )
    return rows


def _load_location_cells() -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    cells = gpd.read_parquet(config.RURAL_PROCESSED_DIR / "grid_suitability.parquet")
    places, place_report = listed_places_for_training(config.LISTED_TOURIST_PLACES_PATH)
    cells = refresh_listed_places(cells, places)
    roads = pd.read_parquet(GRID_ROAD_DISTANCE_PATH)
    roads[CELL_ID_COLUMN] = roads[CELL_ID_COLUMN].astype(str)
    cells[CELL_ID_COLUMN] = cells[CELL_ID_COLUMN].astype(str)
    merged = cells.merge(
        roads[[CELL_ID_COLUMN, ROAD_DISTANCE_KM]],
        on=CELL_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    merged.attrs["place_report"] = place_report
    return merged, place_report


def _attach_v2_features(cells: gpd.GeoDataFrame, places: pd.DataFrame, roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if POBLACION_DISTANCE_COLUMN not in cells.columns:
        raise KeyError(
            "grid_suitability.parquet has no distance_to_poblacion column. "
            "distance_to_nearest_town_center_km is not replaced with another source."
        )
    enriched = cells.copy()
    enriched[TOWN_CENTER_KM] = enriched[POBLACION_DISTANCE_COLUMN]
    enriched[ATTRACTION_COUNT] = attraction_count_within_radius(enriched, places)
    enriched[MAJOR_ROAD_KM] = nearest_major_road_km(enriched, roads)
    return enriched


def _locked_training_rows(cells: pd.DataFrame) -> pd.DataFrame:
    path = config.RURAL_PROCESSED_DIR / "presence_training.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing locked baseline sample: {path}")
    locked = pd.read_parquet(path)
    locked[CELL_ID_COLUMN] = locked[CELL_ID_COLUMN].astype(str)
    feature_frame = cells[[CELL_ID_COLUMN, *V2_FEATURES, "cell_class", "listings_in_cell"]].copy()
    feature_frame[CELL_ID_COLUMN] = feature_frame[CELL_ID_COLUMN].astype(str)
    merged = locked.merge(feature_frame, on=CELL_ID_COLUMN, how="left", validate="one_to_one", suffixes=("", "_grid"))
    for column in PRESENCE_FEATURES:
        grid_column = f"{column}_grid"
        assert merged[column].notna().all(), f"the locked baseline sample is missing {column}"
        assert merged[grid_column].notna().all(), f"the current grid is missing {column} for a locked training cell"
        assert np.allclose(
            merged[column].to_numpy(dtype=float),
            merged[grid_column].to_numpy(dtype=float),
        ), f"the current grid changed the locked baseline feature {column}"
    assert int((merged[PRESENCE_LABEL] == 1).sum()) == 110
    assert int((merged[PRESENCE_LABEL] == 0).sum()) == 110
    positive = merged[PRESENCE_LABEL].to_numpy() == 1
    assert np.array_equal(positive, merged["listings_in_cell"].to_numpy() >= 1)
    return merged


def _replicate_baseline(sample: pd.DataFrame) -> dict[str, object]:
    evaluation = evaluate_presence(sample)
    stored_path = config.RURAL_PROCESSED_DIR / "presence_screen.json"
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    checks = []
    for metric, documented in DOCUMENTED_BASELINE.items():
        replicated = float(evaluation["forest"][metric]["mean"])
        stored_value = float(stored["forest"][metric]["mean"])
        checks.append(
            {
                "metric": metric,
                "documented": documented,
                "replicated": replicated,
                "stored_presence_screen": stored_value,
                "within_documented_tolerance": abs(replicated - documented) <= BASELINE_METRIC_TOLERANCE,
                "matches_stored_screen": abs(replicated - stored_value) <= 1e-9,
            }
        )
    assert all(item["within_documented_tolerance"] and item["matches_stored_screen"] for item in checks), (
        f"baseline replication failed: {checks}"
    )
    return {"tolerance": BASELINE_METRIC_TOLERANCE, "checks": checks, "evaluation": evaluation}


def _fold_rows(sample: pd.DataFrame, folds) -> list[dict[str, object]]:
    rows = []
    groups = sample["municipality"].astype(str)
    for fold_id, (_train_index, test_index) in enumerate(folds, start=1):
        held = groups.iloc[test_index]
        counts = held.value_counts().sort_index()
        rows.append(
            {
                "fold": fold_id,
                "n": int(len(test_index)),
                "municipalities": [{"name": name, "rows": int(count)} for name, count in counts.items()],
            }
        )
    return rows


def _fit_fold_models(sample: pd.DataFrame) -> dict[str, object]:
    assert_feature_matrix(V2_FEATURES)
    assert_feature_matrix(PRESENCE_FEATURES)
    baseline_x = sample[list(PRESENCE_FEATURES)]
    v2_x = sample[list(V2_FEATURES)]
    target = sample[PRESENCE_LABEL].astype(int)
    groups = sample["municipality"].astype(str)
    folds, scheme = _group_splits(baseline_x, target, groups)
    for train_index, test_index in folds:
        overlap = set(groups.iloc[train_index]) & set(groups.iloc[test_index])
        assert not overlap, f"a municipality appears in both the outer train and test fold: {sorted(overlap)}"

    oof = pd.DataFrame(
        {
            CELL_ID_COLUMN: sample[CELL_ID_COLUMN].astype(str),
            "municipality": groups.to_numpy(),
            "y_presence": target.to_numpy(),
            "fold": np.nan,
            "baseline_rf_raw": np.nan,
            "v2_random_forest_raw": np.nan,
            "v2_random_forest_calibrated": np.nan,
            "v2_logistic_raw": np.nan,
            "v2_logistic_calibrated": np.nan,
        }
    )
    fold_metrics: dict[str, list[dict[str, object]]] = {
        "baseline_random_forest": [],
        "v2_logistic_regression": [],
        "v2_random_forest": [],
    }
    correlations = []
    importances = []
    for fold_id, (train_index, test_index) in enumerate(folds, start=1):
        y_train = target.iloc[train_index]
        y_test = target.iloc[test_index].to_numpy()
        oof.loc[test_index, "fold"] = fold_id
        factories = {
            "baseline_random_forest": (baseline_x, presence_forest),
            "v2_random_forest": (v2_x, presence_forest),
            "v2_logistic_regression": (v2_x, ScaledLogisticRegression),
        }
        raw_test = {}
        for name, (features, factory) in factories.items():
            inner = _inner_oof(factory, features.iloc[train_index], y_train, groups.iloc[train_index])
            assert inner is not None, f"{name} fold {fold_id} has no inner grouped scores for calibration"
            threshold = _youden_threshold(y_train.to_numpy(), inner)
            platt = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=config.RANDOM_STATE)
            platt.fit(inner.reshape(-1, 1), y_train.to_numpy())
            model = factory()
            model.fit(features.iloc[train_index], y_train)
            scores = _positive_probability(model, features.iloc[test_index])
            calibrated = platt.predict_proba(scores.reshape(-1, 1))[:, 1]
            raw_test[name] = scores
            free = _threshold_free(y_test, scores)
            operating = _operating_metrics(y_test, scores, threshold)
            calibrated_free = _threshold_free(y_test, calibrated)
            fold_metrics[name].append(
                {
                    "fold": fold_id,
                    **free,
                    "brier_calibrated": calibrated_free["brier"],
                    **{f"youden_{key}" if key != "confusion_matrix" else "confusion_matrix": value for key, value in operating.items()},
                }
            )
            if name == "baseline_random_forest":
                oof.loc[test_index, "baseline_rf_raw"] = scores
            elif name == "v2_random_forest":
                oof.loc[test_index, "v2_random_forest_raw"] = scores
                oof.loc[test_index, "v2_random_forest_calibrated"] = calibrated
            else:
                oof.loc[test_index, "v2_logistic_raw"] = scores
                oof.loc[test_index, "v2_logistic_calibrated"] = calibrated
            if name != "baseline_random_forest" and len(set(y_test.tolist())) > 1:
                importances.append(
                    {
                        "model": name,
                        "fold": fold_id,
                        "features": list(V2_FEATURES),
                        "mean_auc_drop": _permutation_auc_drop(
                            model,
                            features.iloc[test_index],
                            y_test,
                            config.RANDOM_STATE + fold_id,
                        ),
                    }
                )
        correlations.append(
            {
                "fold": fold_id,
                "strong_pairs": _spearman_flags(v2_x.iloc[train_index]),
                "matrix": v2_x.iloc[train_index].corr(method="spearman").round(6).to_dict(),
            }
        )
    return {
        "scheme": scheme,
        "folds": folds,
        "fold_membership": _fold_rows(sample, folds),
        "metrics": fold_metrics,
        "oof": oof,
        "correlations": correlations,
        "importances": importances,
    }


def _summarize_metrics(fold_metrics: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    metric_names = (
        "roc_auc",
        "pr_auc",
        "brier",
        "brier_calibrated",
        "youden_macro_f1",
        "youden_balanced_accuracy",
        "youden_precision",
        "youden_recall",
    )
    summary = {}
    for model, folds in fold_metrics.items():
        summary[model] = {name: _mean_std([fold.get(name) for fold in folds]) for name in metric_names}
        summary[model]["folds"] = folds
    return summary


def _matched_deltas(summary: dict[str, object], candidate: str) -> dict[str, object]:
    baseline = summary["baseline_random_forest"]["roc_auc"]["values"]
    challenger = summary[candidate]["roc_auc"]["values"]
    deltas = [float(left) - float(right) for left, right in zip(challenger, baseline, strict=True)]
    return {
        "candidate": candidate,
        "baseline_roc_auc": baseline,
        "candidate_roc_auc": challenger,
        "delta_auc_fold": deltas,
        **_mean_std(deltas),
    }


def _cluster_bootstrap(oof: pd.DataFrame, candidate_column: str) -> dict[str, object]:
    rng = np.random.default_rng(config.RANDOM_STATE)
    municipalities = oof["municipality"].astype(str).unique()
    deltas = []
    for _draw in range(BOOTSTRAP_DRAWS):
        drawn = rng.choice(municipalities, size=len(municipalities), replace=True)
        parts = [oof.loc[oof["municipality"].astype(str).eq(name)] for name in drawn]
        sample = pd.concat(parts, ignore_index=True)
        if sample["y_presence"].nunique() < 2:
            continue
        baseline = roc_auc_score(sample["y_presence"], sample["baseline_rf_raw"])
        candidate = roc_auc_score(sample["y_presence"], sample[candidate_column])
        deltas.append(float(candidate - baseline))
    array = np.asarray(deltas, dtype=float)
    return {
        "draws": int(len(array)),
        "mean": float(array.mean()) if len(array) else None,
        "interval_low": float(np.quantile(array, 0.025)) if len(array) else None,
        "interval_high": float(np.quantile(array, 0.975)) if len(array) else None,
    }


def _importance_summary(importances: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for model in ("v2_random_forest", "v2_logistic_regression"):
        model_rows = [item for item in importances if item["model"] == model]
        if not model_rows:
            continue
        matrix = np.vstack([item["mean_auc_drop"] for item in model_rows])
        for index, feature in enumerate(V2_FEATURES):
            values = matrix[:, index]
            rows.append(
                {
                    "model": model,
                    "feature": feature,
                    "mean_auc_drop": float(values.mean()),
                    "std_auc_drop": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "fold_values": [float(value) for value in values],
                }
            )
    return rows


def _score_bands(scores: pd.Series) -> pd.Series:
    bands = np.full(len(scores), "0.40 to <0.60", dtype=object)
    bands[scores.to_numpy() < DISPLAY_LOW] = "<0.40"
    bands[scores.to_numpy() >= DISPLAY_HIGH] = ">=0.60"
    return pd.Series(bands, index=scores.index)


def _feature_dictionary() -> pd.DataFrame:
    rows = [
        {
            "feature_name": LISTED_PLACES_WITHIN_RADIUS,
            "feature_group": "tourism",
            "plain_language_definition": "How many curated landscape places sit within 5 km of the cell.",
            "exact_computation": "Count of deduped poi_laguna.json places whose haversine distance from the cell is at most 5 km.",
            "units": "count",
            "source_dataset": "data/poi_laguna.json",
            "source_fields": "latitude, longitude, place_id",
            "spatial_operation": "haversine kilometres, radius 5",
            "missing_data_rule": "A place without coordinates is left out of the universe. A cell with no place inside 5 km is 0. The source file is required.",
            "valid_range_or_sanity_check": "Integer from 0 through the number of deduped places.",
            "computable_for_empty_airbnb_cell": "yes",
            "non_leaky_reason": "The place file is a landscape list. Airbnb listings are not read.",
        },
        {
            "feature_name": DISTANCE_COLUMN,
            "feature_group": "tourism",
            "plain_language_definition": "Kilometres to the nearest curated landscape place.",
            "exact_computation": "Minimum haversine kilometres from the cell to the same deduped poi_laguna.json places used for the count.",
            "units": "kilometres",
            "source_dataset": "data/poi_laguna.json",
            "source_fields": "latitude, longitude, place_id",
            "spatial_operation": "minimum haversine kilometres",
            "missing_data_rule": "If the place file has no usable coordinates, the pipeline fails. A finite distance is expected for every cell when places exist.",
            "valid_range_or_sanity_check": "Finite and greater than or equal to 0.",
            "computable_for_empty_airbnb_cell": "yes",
            "non_leaky_reason": "Same place universe as the count. Airbnb listings are not read.",
        },
        {
            "feature_name": ROAD_DISTANCE_KM,
            "feature_group": "access",
            "plain_language_definition": "Straight-line kilometres from the cell centroid to the nearest mapped road.",
            "exact_computation": "sjoin_nearest in EPSG:32651 from geometry.centroid to every non-empty road, then metres divided by 1000. road_class is not a filter.",
            "units": "kilometres",
            "source_dataset": "data/processed/grid_road_distance.parquet derived from data/laguna_roads.geojson",
            "source_fields": "road geometry",
            "spatial_operation": "nearest geometry in EPSG:32651",
            "missing_data_rule": "A cell with no joined road fails the pipeline. The value is not filled with zero.",
            "valid_range_or_sanity_check": "Finite and greater than or equal to 0. Kilometres equal metres / 1000.",
            "computable_for_empty_airbnb_cell": "yes",
            "non_leaky_reason": "The road layer is the OSM extract. Airbnb listings are not read.",
        },
        {
            "feature_name": TOWN_CENTER_KM,
            "feature_group": "access",
            "plain_language_definition": "Straight-line kilometres to the nearest town-center anchor.",
            "exact_computation": "Copy of the stored distance_to_poblacion value: haversine kilometres to a barangay whose name matches poblacion, or the municipality centroid when that municipality has no poblacion name.",
            "units": "kilometres",
            "source_dataset": "data/processed/rural/grid_suitability.parquet column distance_to_poblacion, derived from data/laguna.geojson",
            "source_fields": "adm4_name, adm3_name, center_lon, center_lat",
            "spatial_operation": "minimum haversine kilometres",
            "missing_data_rule": "A null stored distance stays null and the pipeline fails. It is not filled with a constant.",
            "valid_range_or_sanity_check": "Finite and greater than or equal to 0.",
            "computable_for_empty_airbnb_cell": "yes",
            "non_leaky_reason": "The anchors are barangay names and municipality centroids. Airbnb listings are not read.",
        },
        {
            "feature_name": MAJOR_ROAD_KM,
            "feature_group": "access",
            "plain_language_definition": "Straight-line kilometres from the cell centroid to the nearest road tagged major.",
            "exact_computation": "Filter laguna_roads.geojson to road_class equal to major, then sjoin_nearest in EPSG:32651 and divide metres by 1000.",
            "units": "kilometres",
            "source_dataset": "data/laguna_roads.geojson",
            "source_fields": "geometry, road_class",
            "spatial_operation": "nearest major geometry in EPSG:32651",
            "missing_data_rule": "If no major roads exist, every cell is missing and the pipeline fails. A missing join is not filled with zero.",
            "valid_range_or_sanity_check": "Finite and greater than or equal to 0. Kilometres equal metres / 1000.",
            "computable_for_empty_airbnb_cell": "yes",
            "non_leaky_reason": "The road layer is the OSM extract. Airbnb listings are not read.",
        },
        {
            "feature_name": ATTRACTION_COUNT,
            "feature_group": "tourism",
            "plain_language_definition": "How many waterfall or mountain places sit within 5 km.",
            "exact_computation": "Same deduped poi_laguna.json universe, restricted to category Falls or Mountains, counted by haversine distance of at most 5 km.",
            "units": "count",
            "source_dataset": "data/poi_laguna.json",
            "source_fields": "latitude, longitude, place_id, category",
            "spatial_operation": "haversine kilometres, radius 5",
            "missing_data_rule": "True absence inside 5 km is 0. If the file or the Falls and Mountains categories are missing, the pipeline fails instead of substituting another source.",
            "valid_range_or_sanity_check": "Integer from 0 through the number of deduped Falls and Mountains, and never above listed_places_within_radius.",
            "computable_for_empty_airbnb_cell": "yes",
            "non_leaky_reason": "The subset is the landscape list. Airbnb listings are not read.",
        },
    ]
    return pd.DataFrame(rows)


def _source_registry(grid_crs: str) -> pd.DataFrame:
    unknown = "unknown"
    rows = [
        {
            "source_name": "poi_laguna.json",
            "local_path": "data/poi_laguna.json",
            "url": unknown,
            "license": unknown,
            "retrieval_date": unknown,
            "layer_name": "curated landscape places",
            "geographic_coverage": "Laguna places in the local file",
            "crs": "WGS 84 longitude and latitude",
            "preprocessing": "Drop rows without numeric coordinates. Keep the first row of each non-empty place_id. Keep blank place_id rows. Attraction counts keep category Falls or Mountains only.",
            "field_definitions": "category, name, latitude, longitude, place_id",
            "origin": "locally generated",
        },
        {
            "source_name": "laguna_roads.geojson",
            "local_path": "data/laguna_roads.geojson",
            "url": unknown,
            "license": unknown,
            "retrieval_date": unknown,
            "layer_name": "mapped roads",
            "geographic_coverage": "Laguna road extract used by this project",
            "crs": "as stored on the file; distances are calculated after projection to EPSG:32651",
            "preprocessing": "Baseline road distance uses every non-empty geometry. v2 major-road distance keeps road_class equal to major only.",
            "field_definitions": "geometry, highway_type, name, osm_id, ref, road_class",
            "origin": "downloaded",
        },
        {
            "source_name": "grid_road_distance.parquet",
            "local_path": "data/processed/grid_road_distance.parquet",
            "url": unknown,
            "license": unknown,
            "retrieval_date": unknown,
            "layer_name": "cell road distance",
            "geographic_coverage": "study grid",
            "crs": "EPSG:32651 metres, stored as kilometres",
            "preprocessing": "Existing nearest-road table. Not rebuilt for v2.",
            "field_definitions": "cell_id, road_distance_m, road_distance_km",
            "origin": "derived",
        },
        {
            "source_name": "grid_suitability.parquet",
            "local_path": "data/processed/rural/grid_suitability.parquet",
            "url": unknown,
            "license": unknown,
            "retrieval_date": unknown,
            "layer_name": "study grid",
            "geographic_coverage": "Laguna 1 km grid",
            "crs": grid_crs,
            "preprocessing": "Existing grid. Geometry and cell ids are not rebuilt. distance_to_poblacion is copied to distance_to_nearest_town_center_km.",
            "field_definitions": "cell_id, geometry, longitude, latitude, municipality, cell_class, listings_in_cell, distance_to_poblacion",
            "origin": "derived",
        },
        {
            "source_name": "laguna.geojson",
            "local_path": "data/laguna.geojson",
            "url": unknown,
            "license": unknown,
            "retrieval_date": unknown,
            "layer_name": "barangay polygons",
            "geographic_coverage": "Laguna barangays",
            "crs": "as stored; town-center distance uses longitude and latitude anchors",
            "preprocessing": "Already applied in the stored distance_to_poblacion column. Poblacion names match poblaci[oó]n or (pob. Municipalities without a match use a municipality centroid fallback.",
            "field_definitions": "adm4_name, adm3_name, center_lon, center_lat",
            "origin": "locally generated",
        },
        {
            "source_name": "presence_training.parquet",
            "local_path": "data/processed/rural/presence_training.parquet",
            "url": unknown,
            "license": unknown,
            "retrieval_date": unknown,
            "layer_name": "locked 220-cell sample",
            "geographic_coverage": "rural Laguna cells with a municipality",
            "crs": "not a geometry file",
            "preprocessing": "Read only. The 110 positive and 110 negative rows and the three baseline feature values are kept. Negatives are not resampled.",
            "field_definitions": "cell_id, municipality, presence, listed_places_within_radius, distance_to_listed_tourist_place, road_distance_km",
            "origin": "derived",
        },
        {
            "source_name": "listings label",
            "local_path": "listings_in_cell on the study grid; active listings were defined when that column was built",
            "url": unknown,
            "license": unknown,
            "retrieval_date": unknown,
            "layer_name": "label only",
            "geographic_coverage": "rural cells",
            "crs": "not used as a distance",
            "preprocessing": "presence is 1 when listings_in_cell is at least 1 and 0 when it is 0. The column is not a model feature.",
            "field_definitions": "listings_in_cell",
            "origin": "derived",
        },
    ]
    return pd.DataFrame(rows)


def _comparison_table(summary: dict[str, object], deltas: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for model, payload in summary.items():
        fold_count = len(payload["roc_auc"]["values"])
        for fold_id in range(fold_count):
            row = {"model": model, "fold": fold_id + 1}
            for metric in (
                "roc_auc",
                "pr_auc",
                "brier",
                "brier_calibrated",
                "youden_macro_f1",
                "youden_balanced_accuracy",
                "youden_precision",
                "youden_recall",
            ):
                row[metric] = payload[metric]["values"][fold_id]
            if model == "baseline_random_forest":
                row["delta_auc_vs_baseline"] = 0.0
            else:
                row["delta_auc_vs_baseline"] = deltas[model]["delta_auc_fold"][fold_id]
            rows.append(row)
        summary_row = {"model": model, "fold": "mean"}
        std_row = {"model": model, "fold": "sample_sd"}
        for metric in (
            "roc_auc",
            "pr_auc",
            "brier",
            "brier_calibrated",
            "youden_macro_f1",
            "youden_balanced_accuracy",
            "youden_precision",
            "youden_recall",
        ):
            summary_row[metric] = payload[metric]["mean"]
            std_row[metric] = payload[metric]["std"]
        if model == "baseline_random_forest":
            summary_row["delta_auc_vs_baseline"] = 0.0
            std_row["delta_auc_vs_baseline"] = 0.0
        else:
            summary_row["delta_auc_vs_baseline"] = deltas[model]["mean"]
            std_row["delta_auc_vs_baseline"] = deltas[model]["std"]
        rows.extend([summary_row, std_row])
    return pd.DataFrame(rows)


def _validation_markdown(report: dict[str, object]) -> str:
    lines = [
        "# site_candidate_v2 validation",
        "",
        report["interpretation"],
        "",
        f"Research-use decision: `{report['research_use_decision']}`.",
        "",
        "## Sample",
        "",
        f"Rural cells: {report['population']['rural_cells']}.",
        f"Positive rural cells (`listings_in_cell >= 1`): {report['population']['positive_rural_cells']}.",
        f"Empty rural cells (`listings_in_cell == 0`): {report['population']['empty_rural_cells']}.",
        f"Training rows: {report['population']['training_rows']} "
        f"({report['population']['training_positives']} positive, {report['population']['training_negatives']} negative).",
        f"Negative-sampling seed recorded on the locked sample: {report['population']['negative_sampling_seed']}.",
        "The negative rows were read from `presence_training.parquet` and were not drawn again.",
        f"Label: {report['population']['label_definition']}.",
        f"Exclusions: {report['population']['exclusions']}.",
        "",
        "## Features",
        "",
        "Fixed v2 features: " + ", ".join(f"`{name}`" for name in report["v2_features"]) + ".",
        "",
        "Omitted requested features:",
        "",
    ]
    for item in report["omitted_features"]:
        lines.append(f"- `{item['feature']}`: {item['reason']}")
    lines.extend(["", "## Coverage on rural cells with a municipality", ""])
    for item in report["coverage_rural_with_municipality"]:
        lines.append(
            f"- `{item['feature']}`: {item['computed']} of {item['rows']} "
            f"({item['percent_computed']:.1%}), missing {item['missing']}."
        )
    lines.extend(["", "## Outer folds", ""])
    for fold in report["folds"]:
        names = ", ".join(f"{item['name']} ({item['rows']})" for item in fold["municipalities"])
        lines.append(f"- Fold {fold['fold']}: {fold['n']} rows. {names}.")
    lines.extend(["", "## Metrics", ""])
    lines.append(
        "Primary metric is ROC-AUC. Sample standard deviation uses ddof=1. "
        "Youden cuts are training-side only and are not a business threshold."
    )
    lines.append("")
    for model, payload in report["metrics"].items():
        auc = payload["roc_auc"]
        lines.append(
            f"- `{model}` ROC-AUC mean {auc['mean']:.3f} (sample SD {auc['std']:.3f}); "
            f"fold values {', '.join(f'{value:.3f}' for value in auc['values'])}."
        )
        lines.append(
            f"  Macro F1 {payload['youden_macro_f1']['mean']:.3f}, "
            f"balanced accuracy {payload['youden_balanced_accuracy']['mean']:.3f}, "
            f"precision {payload['youden_precision']['mean']:.3f}, "
            f"recall {payload['youden_recall']['mean']:.3f}, "
            f"PR-AUC {payload['pr_auc']['mean']:.3f}, "
            f"Brier {payload['brier']['mean']:.3f}."
        )
    lines.extend(["", "## Confusion matrices at the training-side Youden cut", ""])
    for model, payload in report["metrics"].items():
        for fold in payload["folds"]:
            lines.append(f"- `{model}` fold {fold['fold']}: {fold['confusion_matrix']}.")
    lines.extend(["", "## Matched AUC differences", ""])
    for name, payload in report["matched_deltas"].items():
        lines.append(
            f"- `{name}` deltas {', '.join(f'{value:.3f}' for value in payload['delta_auc_fold'])}; "
            f"mean {payload['mean']:.3f}; sample SD {payload['std']:.3f}."
        )
        bootstrap = report["municipality_bootstrap"][name]
        lines.append(
            f"  Municipality bootstrap mean {bootstrap['mean']:.3f}, "
            f"95% interval {bootstrap['interval_low']:.3f} to {bootstrap['interval_high']:.3f} "
            f"over {bootstrap['draws']} draws."
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "The model represents observational association, not causation.",
            "The negative examples are sampled rather than all empty rural cells.",
            "The positive sample is limited to 110 rural cells.",
            "Generalization across municipalities and geography remains uncertain.",
            "The score does not directly predict revenue, occupancy, costs, legality, utilities, flood risk, building feasibility, or investor return.",
            "A high score means stronger similarity to observed rural STR-location conditions. It does not prove that an STR will succeed.",
            "",
        ]
    )
    return "\n".join(lines)


def _selected_v2_model_name(summary: dict[str, object]) -> str:
    logistic = float(summary["v2_logistic_regression"]["roc_auc"]["mean"])
    forest = float(summary["v2_random_forest"]["roc_auc"]["mean"])
    if logistic > forest:
        return "v2_logistic_regression"
    return "v2_random_forest"


def _fit_and_score(sample: pd.DataFrame, cells: pd.DataFrame, model_name: str) -> tuple[pd.DataFrame, object]:
    if model_name == "v2_logistic_regression":
        model = ScaledLogisticRegression()
    else:
        model = presence_forest()
    model.fit(sample[list(V2_FEATURES)], sample[PRESENCE_LABEL].astype(int))
    rural = cells.loc[
        (cells["cell_class"] == RURAL_CELL)
        & cells["municipality"].notna()
        & (cells["municipality"].astype(str) != "")
    ].copy()
    require_observed_values(rural, V2_FEATURES)
    scores = _positive_probability(model, rural[list(V2_FEATURES)])
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
            "model_name": model_name,
            "feature_version": FEATURE_VERSION,
            "training_rows": int(len(sample)),
            "training_positives": int((sample[PRESENCE_LABEL] == 1).sum()),
            "training_negatives": int((sample[PRESENCE_LABEL] == 0).sum()),
            "random_state": int(config.RANDOM_STATE),
            "preprocessing": "logistic uses StandardScaler fit on the full 220-row sample; random forest is unscaled",
        }
    )
    for column in V2_FEATURES:
        scored[column] = rural[column].to_numpy()
    return scored, model


def run_site_candidate_v2(output_dir: Path | None = None) -> dict[str, object]:
    output_dir = config.RURAL_PROCESSED_DIR if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cells, place_report = _load_location_cells()
    places, attraction_report = load_places_with_category(config.LISTED_TOURIST_PLACES_PATH)
    baseline_places, _baseline_report = listed_places_for_training(config.LISTED_TOURIST_PLACES_PATH)
    assert len(places) == len(baseline_places), "v2 place dedupe diverged from the baseline place list"
    roads = gpd.read_file(LAGUNA_ROADS_PATH)
    cells = _attach_v2_features(cells, places, roads)
    rural = cells.loc[cells["cell_class"] == RURAL_CELL].copy()
    rural_named = rural.loc[rural["municipality"].notna() & (rural["municipality"].astype(str) != "")]
    coverage = _coverage(rural_named, V2_FEATURES)
    v2_complete = all(item["missing"] == 0 for item in coverage)
    baseline_complete = all(
        item["missing"] == 0 for item in coverage if item["feature"] in PRESENCE_FEATURES
    )
    if v2_complete:
        require_observed_values(rural_named, V2_FEATURES)
    sample = _locked_training_rows(cells)
    require_observed_values(sample, V2_FEATURES)
    assert (sample[ATTRACTION_COUNT] <= sample[LISTED_PLACES_WITHIN_RADIUS]).all()
    replication = _replicate_baseline(sample)
    fitted = _fit_fold_models(sample)
    summary = _summarize_metrics(fitted["metrics"])
    deltas = {
        "v2_logistic_regression": _matched_deltas(summary, "v2_logistic_regression"),
        "v2_random_forest": _matched_deltas(summary, "v2_random_forest"),
    }
    selected_name = _selected_v2_model_name(summary)
    decision_deltas = deltas[selected_name]["delta_auc_fold"]
    decision = research_use_decision(decision_deltas, v2_complete, baseline_complete)
    bootstrap = {
        name: _cluster_bootstrap(
            fitted["oof"],
            "v2_logistic_raw" if name == "v2_logistic_regression" else "v2_random_forest_raw",
        )
        for name in deltas
    }
    scored, _model = _fit_and_score(sample, cells, selected_name)
    empty_scores = scored.loc[scored["zero_current_listings"]]
    band_counts = empty_scores["display_band"].value_counts().to_dict()
    population = {
        "rural_cells": int(len(rural)),
        "positive_rural_cells": int((rural_named["listings_in_cell"] >= 1).sum()),
        "empty_rural_cells": int((rural_named["listings_in_cell"] == 0).sum()),
        "training_rows": int(len(sample)),
        "training_positives": 110,
        "training_negatives": 110,
        "negative_sampling_seed": int(config.RANDOM_STATE),
        "label_definition": "presence = 1 when listings_in_cell >= 1; presence = 0 when listings_in_cell == 0",
        "exclusions": "Urban cells, unclassified cells, and rural cells without a municipality are outside the training population.",
    }
    report = {
        "feature_version": FEATURE_VERSION,
        "interpretation": (
            "Candidate site pattern score: higher values indicate stronger similarity to the "
            "observable location conditions of rural Laguna cells where active STRs are currently "
            "observed. This is a screening measure, not a forecast of revenue, occupancy, "
            "investment return, regulatory suitability, or STR success."
        ),
        "research_use_decision": decision,
        "selected_v2_model": selected_name,
        "adopted_for_map": decision == "deploy v2 as a separate layer",
        "population": population,
        "places": place_report,
        "attraction_places": {
            "categories": list(ATTRACTION_CATEGORIES),
            "dedupe": "first row of each non-empty place_id; blank place_id rows kept",
            **attraction_report,
        },
        "v2_features": list(V2_FEATURES),
        "omitted_features": list(OMITTED_FEATURES),
        "coverage_rural_with_municipality": coverage,
        "baseline_replication": {key: value for key, value in replication.items() if key != "evaluation"},
        "baseline_replication_forest": replication["evaluation"]["forest"],
        "folds": fitted["fold_membership"],
        "validation_scheme": fitted["scheme"],
        "metrics": summary,
        "matched_deltas": deltas,
        "municipality_bootstrap": bootstrap,
        "spearman_flag_threshold": SPEARMAN_FLAG,
        "correlations": fitted["correlations"],
        "permutation_importance": _importance_summary(fitted["importances"]),
        "empty_rural_cells_by_display_band": {str(key): int(value) for key, value in band_counts.items()},
        "display_bands": {
            "rule": "Fixed before scoring: <0.40, 0.40 to <0.60, >=0.60. These bins count cells. They were not tuned on combined out-of-fold predictions.",
            "wording": "Candidate site pattern",
        },
        "probability_note": (
            "The training sample has 110 positives and 110 negatives. The rural population has "
            "110 positives and 931 empty cells. Raw predict_proba values are not the population "
            "probability that a rural cell contains an Airbnb."
        ),
        "youden_note": YOUDEN_NOTE,
        "limitations": [
            "The model represents observational association, not causation.",
            "The negative examples are sampled rather than all empty rural cells.",
            "The positive sample is limited to 110 rural cells.",
            "Generalization across municipalities and geography remains uncertain.",
            "The score does not directly predict revenue, occupancy, costs, legality, utilities, flood risk, building feasibility, or investor return.",
            "A high score means stronger similarity to observed rural STR-location conditions. It does not prove that an STR will succeed.",
        ],
    }
    comparison = _comparison_table(summary, deltas)
    training_columns = [
        CELL_ID_COLUMN,
        "municipality",
        PRESENCE_LABEL,
        "listings_in_cell",
        *V2_FEATURES,
    ]
    sample[training_columns].to_parquet(output_dir / "site_candidate_v2_training.parquet", index=False)
    scored.to_parquet(output_dir / "site_candidate_v2_scores.parquet", index=False)
    fitted["oof"].to_parquet(output_dir / "site_candidate_v2_oof_predictions.parquet", index=False)
    comparison.to_csv(output_dir / "site_candidate_v2_comparison.csv", index=False)
    _feature_dictionary().to_csv(output_dir / "site_candidate_v2_feature_dictionary.csv", index=False)
    grid_crs = str(cells.crs) if getattr(cells, "crs", None) is not None else "unknown"
    _source_registry(grid_crs).to_csv(output_dir / "site_candidate_v2_source_registry.csv", index=False)
    ready = _json_ready(report)
    (output_dir / "site_candidate_v2_report.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
    (output_dir / "site_candidate_v2_validation.md").write_text(_validation_markdown(ready), encoding="utf-8")
    return ready
