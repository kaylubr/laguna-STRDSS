from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
BOUNDARY_DIR = PROJECT_ROOT / "assets" / "boundaries"
MODEL_DIR = PROJECT_ROOT / "models"

AIRROI_LISTINGS_PATH = DATA_DIR / "laguna_listings.json"
AIRROI_HISTORY_DIR = DATA_DIR / "history"
PSA_PATH = DATA_DIR / "PSA.json"

BARANGAY_POLYGONS_PATH = DATA_DIR / "laguna.geojson"
BARANGAY_CLASSIFICATION_PATH = DATA_DIR / "barangay_with_classification.json"
RURAL_URBAN_LISTINGS_PATH = INTERIM_DIR / "airoi_rural_urban.csv"

PROJECTED_CRS = "EPSG:32651"
GEOGRAPHIC_CRS = "EPSG:4326"

PRIMARY_CELL_SIZE_M = 1000
SENSITIVITY_CELL_SIZES_M = (500, 2000)
CELL_SIZES_M = (PRIMARY_CELL_SIZE_M, *SENSITIVITY_CELL_SIZES_M)
MIN_LAND_COVERAGE = 0.5
EXPECTED_CELL_COUNT = 1928

TTM_WINDOW_START = "2025-08"
TTM_WINDOW_END = "2026-07"

PROVINCE_NAME = "Laguna"
EXPECTED_MUNICIPALITY_COUNT = 30

RURAL_AREA_SHARE_THRESHOLD = 0.5
RURAL_PROCESSED_DIR = PROCESSED_DIR / "rural"
NEARBY_TRAINING_THRESHOLD_KM = 5.0
NEIGHBORHOOD_RADIUS_KM = NEARBY_TRAINING_THRESHOLD_KM
LISTED_TOURIST_PLACES_PATH = DATA_DIR / "poi_laguna.json"

TOURIST_ACCESS_BLOC = 1.0 / 3.0
NEARBY_MARKET_BLOC = 1.0 / 3.0
LOCAL_COMPETITION_BLOC = 1.0 / 3.0

SITE_SCORE_DIRECTIONS = {
    "distance_to_listed_tourist_place": "negative",
    "nearby_mean_revenue": "positive",
    "nearby_mean_occupancy": "positive",
    "competition_listing_count": "negative",
}

SITE_SCORE_WEIGHTS = {
    "distance_to_listed_tourist_place": TOURIST_ACCESS_BLOC,
    "nearby_mean_revenue": NEARBY_MARKET_BLOC / 2.0,
    "nearby_mean_occupancy": NEARBY_MARKET_BLOC / 2.0,
    "competition_listing_count": LOCAL_COMPETITION_BLOC,
}

TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42
BLEND_RATIO = 0.5

PARAM_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", 0.5],
}

REVENUE_TARGET = "ttm_revenue"
OCCUPANCY_TARGET = "ttm_occupancy"

SUITABILITY_INDICATOR_DIRECTIONS = {
    "predicted_revenue": "positive",
    "predicted_occupancy": "positive",
    "poi_density_total": "positive",
    "distance_to_nearest_tourist_attraction": "negative",
    "distance_to_nearest_transportation_facility": "negative",
}

SUITABILITY_INDICATORS = tuple(SUITABILITY_INDICATOR_DIRECTIONS)

POI_BLOC_INDICATORS = (
    "poi_density_restaurants",
    "poi_density_commercial",
    "poi_density_transportation",
    "poi_density_recreation",
    "poi_density_tourist_attraction",
    "poi_density_other_facilities",
)

COMPOSITE_INDICATOR_DIRECTIONS = {
    "poi_density_restaurants": "positive",
    "poi_density_commercial": "positive",
    "poi_density_transportation": "positive",
    "poi_density_recreation": "positive",
    "poi_density_tourist_attraction": "positive",
    "poi_density_other_facilities": "positive",
    "distance_to_nearest_transportation_facility": "negative",
    "distance_to_nearest_tourist_attraction": "negative",
    "distance_to_laguna_de_bay": "negative",
    "distance_to_other_water": "negative",
    "distance_to_poblacion": "negative",
    "predicted_revenue": "positive",
    "predicted_occupancy": "positive",
}

COMPOSITE_INDICATORS = tuple(COMPOSITE_INDICATOR_DIRECTIONS)

EWM_WEIGHTING = "ewm"
HYBRID_WEIGHTING = "hybrid"

EARTH_RADIUS_KM = 6371.0
