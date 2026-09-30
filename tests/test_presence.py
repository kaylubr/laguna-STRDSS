import pandas as pd
import pytest

from str_suitability.modeling.presence import (
    EXCLUDED_MARKET_FEATURES,
    PRESENCE_FEATURES,
    earnings_per_bedroom,
    select_training_rows,
)


def _square(cell_id: str, cell_class: str, municipality: str, listings: int) -> dict:
    return {
        "cell_id": cell_id,
        "cell_class": cell_class,
        "municipality": municipality,
        "listings_in_cell": listings,
        "listed_places_within_radius": 1,
        "distance_to_listed_tourist_place": 2.0,
        "road_distance_km": 0.2,
    }


def test_training_rows_balance_listed_and_unlisted_rural_squares():
    rows = [
        _square("listed-a", "rural", "A", 2),
        _square("listed-b", "rural", "B", 1),
        _square("empty-a", "rural", "A", 0),
        _square("empty-b", "rural", "B", 0),
        _square("empty-c", "rural", "C", 0),
        _square("urban", "urban", "A", 4),
        _square("no-town", "rural", "", 1),
    ]
    sample, report = select_training_rows(pd.DataFrame(rows))
    assert set(PRESENCE_FEATURES).isdisjoint(EXCLUDED_MARKET_FEATURES)
    assert report["listed_rural_squares"] == 2
    assert report["unlisted_rural_squares_sampled"] == 2
    assert "urban" not in set(sample["cell_id"])
    assert "no-town" not in set(sample["cell_id"])
    assert int(sample["presence"].sum()) == 2
    assert int((sample["presence"] == 0).sum()) == 2
    again, _report = select_training_rows(pd.DataFrame(rows))
    assert sample["cell_id"].tolist() == again["cell_id"].tolist()


def test_earnings_per_bedroom_needs_three_listings_and_ignores_zero_bedrooms():
    listings = pd.DataFrame(
        [
            {"cell_id": "many", "bedrooms": 2, "ttm_revenue": 100.0},
            {"cell_id": "many", "bedrooms": 0, "ttm_revenue": 50.0},
            {"cell_id": "many", "bedrooms": 4, "ttm_revenue": 200.0},
            {"cell_id": "many", "bedrooms": 1, "ttm_revenue": 40.0},
            {"cell_id": "few", "bedrooms": 2, "ttm_revenue": 80.0},
            {"cell_id": "few", "bedrooms": 2, "ttm_revenue": 80.0},
        ]
    )
    result = earnings_per_bedroom(listings).set_index("cell_id")
    assert result.loc["many", "listings_with_bedrooms"] == 3
    assert result.loc["many", "earnings_per_bedroom"] == pytest.approx((50 + 50 + 40) / 3)
    assert "few" not in result.index
