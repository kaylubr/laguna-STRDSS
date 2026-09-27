import json
from pathlib import Path

import pandas as pd

LISTING_FIELDS = (
    "listing_id",
    "municipality",
    "psgc_code",
    "latitude",
    "longitude",
    "rating",
    "number_of_reviews",
    "superhost",
    "ttm_revenue",
    "ttm_occupancy",
)


def load_airroi_listings(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    listings = payload["listings"]
    frame = pd.DataFrame(listings)
    frame["listing_id"] = frame["listing_id"].astype(str)
    return frame


def load_listing_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["metadata"]


def load_history_panel(history_dir: Path, window: tuple[str, str]) -> pd.DataFrame:
    start, end = window
    records = []
    for path in sorted(history_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        listing_id = str(payload["listing_id"])
        for month in payload.get("response_data", {}).get("results", []):
            if start <= month["date"] <= end:
                records.append(
                    {
                        "listing_id": listing_id,
                        "month": month["date"],
                        "revenue": month.get("revenue"),
                        "occupancy": month.get("occupancy"),
                        "average_daily_rate": month.get("average_daily_rate"),
                        "months_requested": payload.get("months_available"),
                    }
                )
    return pd.DataFrame.from_records(records)


def load_full_history(history_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(history_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        listing_id = str(payload["listing_id"])
        for month in payload.get("response_data", {}).get("results", []):
            records.append(
                {
                    "listing_id": listing_id,
                    "month": month["date"],
                    "revenue": month.get("revenue"),
                    "occupancy": month.get("occupancy"),
                }
            )
    return pd.DataFrame.from_records(records)
