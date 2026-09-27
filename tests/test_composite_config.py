from str_suitability import config


def test_rural_area_share_threshold_is_one_half():
    assert config.RURAL_AREA_SHARE_THRESHOLD == 0.5


def test_blend_ratio_default_is_one_half():
    assert config.BLEND_RATIO == 0.5


def test_composite_set_drops_the_aggregate_and_demographics():
    assert "poi_density_total" not in config.COMPOSITE_INDICATORS
    assert "population_density_per_km2" not in config.COMPOSITE_INDICATORS
    assert "tourist_attraction_density" not in config.COMPOSITE_INDICATORS
    assert "tourist_attraction_count" not in config.COMPOSITE_INDICATORS


def test_composite_set_includes_the_poi_bloc_new_indicators_and_predictions():
    for name in config.POI_BLOC_INDICATORS:
        assert name in config.COMPOSITE_INDICATORS
    for name in (
        "distance_to_laguna_de_bay",
        "distance_to_other_water",
        "distance_to_poblacion",
        "predicted_revenue",
        "predicted_occupancy",
    ):
        assert name in config.COMPOSITE_INDICATORS


def test_thesis_indicator_set_is_unchanged():
    assert len(config.SUITABILITY_INDICATORS) == 5
    assert "poi_density_total" in config.SUITABILITY_INDICATORS


def test_nearby_training_threshold_is_positive():
    assert config.NEARBY_TRAINING_THRESHOLD_KM > 0.0
