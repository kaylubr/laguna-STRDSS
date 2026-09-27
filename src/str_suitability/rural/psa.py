import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from str_suitability.config import PROJECTED_CRS
from str_suitability.spatial.join import normalize_psgc

RURAL = "R"
URBAN = "U"
URBAN_RURAL_COLUMN = "urban_rural"
PSGC_COLUMN = "psgc"
POLYGON_PSGC_COLUMN = "adm4_pcode"
POLYGON_NAME_COLUMN = "adm4_name"
MUNICIPALITY_NAME_COLUMN = "adm3_name"
MUNICIPALITY_PSGC_COLUMN = "adm3_pcode"
POLYGON_LABEL_COLUMNS = (MUNICIPALITY_NAME_COLUMN, MUNICIPALITY_PSGC_COLUMN)


def load_barangay_classification(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    records = payload["results"]
    frame = pd.DataFrame(
        {
            PSGC_COLUMN: [normalize_psgc(record["code"]) for record in records],
            "barangay_name": [record["area_name"] for record in records],
            URBAN_RURAL_COLUMN: [record[URBAN_RURAL_COLUMN] for record in records],
        }
    )
    assert frame[PSGC_COLUMN].notna().all(), "every PSA barangay must carry a PSGC code"
    assert frame[PSGC_COLUMN].is_unique, "PSA barangay codes must be unique"
    assert set(frame[URBAN_RURAL_COLUMN]) <= {RURAL, URBAN}, (
        f"PSA urban_rural must be {RURAL} or {URBAN}, found "
        f"{sorted(set(frame[URBAN_RURAL_COLUMN]) - {RURAL, URBAN})}"
    )
    return frame


def load_barangay_polygons(path: Path) -> gpd.GeoDataFrame:
    polygons = gpd.read_file(path)
    assert POLYGON_PSGC_COLUMN in polygons.columns, (
        f"barangay polygons carry no {POLYGON_PSGC_COLUMN} field to join PSA codes on"
    )
    polygons = polygons.to_crs(PROJECTED_CRS)
    polygons[PSGC_COLUMN] = polygons[POLYGON_PSGC_COLUMN].map(normalize_psgc)
    assert polygons[PSGC_COLUMN].notna().all(), "every barangay polygon must resolve to a PSGC code"
    assert polygons[PSGC_COLUMN].is_unique, "barangay polygon PSGC codes must be unique"
    carried = [
        PSGC_COLUMN,
        POLYGON_NAME_COLUMN,
        *[column for column in POLYGON_LABEL_COLUMNS if column in polygons.columns],
        "geometry",
    ]
    return gpd.GeoDataFrame(
        polygons[carried],
        geometry="geometry",
        crs=PROJECTED_CRS,
    )


def classify_barangay_polygons(
    polygons: gpd.GeoDataFrame, classification: pd.DataFrame
) -> gpd.GeoDataFrame:
    lookup = dict(zip(classification[PSGC_COLUMN], classification[URBAN_RURAL_COLUMN]))
    classified = polygons.assign(**{URBAN_RURAL_COLUMN: polygons[PSGC_COLUMN].map(lookup)})
    return gpd.GeoDataFrame(classified, geometry="geometry", crs=polygons.crs)


def report_barangay_classification(
    polygons: gpd.GeoDataFrame, classification: pd.DataFrame
) -> dict[str, object]:
    classified = polygons[URBAN_RURAL_COLUMN].notna()
    return {
        "psa_barangays": int(len(classification)),
        "psa_rural": int((classification[URBAN_RURAL_COLUMN] == RURAL).sum()),
        "psa_urban": int((classification[URBAN_RURAL_COLUMN] == URBAN).sum()),
        "polygons": int(len(polygons)),
        "polygons_classified": int(classified.sum()),
        "polygons_without_psa_record": int((~classified).sum()),
        "psa_codes_without_polygon": int(
            (~classification[PSGC_COLUMN].isin(set(polygons[PSGC_COLUMN]))).sum()
        ),
    }
