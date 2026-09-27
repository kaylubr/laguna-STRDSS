import json
import unicodedata
from pathlib import Path

import pandas as pd

from str_suitability.reference import PSGC_NAME_OVERRIDES

POPULATION_COLUMN = "population"


def normalize_municipality_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name)
    without_diacritics = "".join(character for character in decomposed if not unicodedata.combining(character))
    return " ".join(without_diacritics.lower().replace("city of", "").split())


def load_psa_population(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["municipalities"])
    frame["municipality"] = frame["name"]
    frame["normalized_name"] = frame["municipality"].map(normalize_municipality_name)
    frame["is_city"] = frame.get("type", pd.Series(index=frame.index, dtype=object)).eq("city")
    return frame[["municipality", "normalized_name", "is_city", POPULATION_COLUMN]]


def load_psa_metadata(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "province": payload["province"],
        "year": payload["year"],
        "source": payload["source"],
        "reference_date": payload["reference_date"],
        "total_population": payload["total_population"],
    }


def validate_psa_population(frame: pd.DataFrame, expected_total: int) -> dict[str, object]:
    return {
        "unique_municipalities": frame["municipality"].is_unique,
        "unique_normalized_names": frame["normalized_name"].is_unique,
        "no_missing_population": not frame[POPULATION_COLUMN].isna().any(),
        "municipalities_sum_to_provincial_total": bool(
            frame[POPULATION_COLUMN].sum() == expected_total
        ),
    }


def reconcile_with_boundaries(
    population: pd.DataFrame, boundary_names: list[str]
) -> dict[str, object]:
    normalized_boundaries = {normalize_municipality_name(name) for name in boundary_names}
    normalized_population = set(population["normalized_name"])
    return {
        "missing_from_boundaries": sorted(normalized_population - normalized_boundaries),
        "missing_from_population": sorted(normalized_boundaries - normalized_population),
        "matched": sorted(normalized_population & normalized_boundaries),
    }


def fill_missing_psgc_refs(municipalities: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    normalized = municipalities["name"].map(normalize_municipality_name)
    overrides = normalized.map(
        lambda name: PSGC_NAME_OVERRIDES.get(name, {}).get("psgc_ref")
    )
    filled = municipalities["psgc_ref"].fillna(overrides)
    report = {
        "municipalities": int(len(municipalities)),
        "from_osm": int(municipalities["psgc_ref"].notna().sum()),
        "from_reference": int((municipalities["psgc_ref"].isna() & filled.notna()).sum()),
        "still_missing": municipalities.loc[filled.isna(), "name"].tolist(),
    }
    return municipalities.assign(psgc_ref=filled), report


def attach_psgc_codes(
    population: pd.DataFrame, municipalities: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    complete, _ = fill_missing_psgc_refs(municipalities)
    lookup = dict(
        zip(complete["name"].map(normalize_municipality_name), complete["psgc_ref"])
    )
    codes = population["normalized_name"].map(lookup)
    report = {
        "attached": int(codes.notna().sum()),
        "total": int(len(population)),
        "unmatched": population.loc[codes.isna(), "municipality"].tolist(),
    }
    return population.assign(psgc_ref=codes), report
