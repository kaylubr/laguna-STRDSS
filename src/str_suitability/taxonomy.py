from enum import StrEnum


class PoiCategory(StrEnum):
    RESTAURANTS = "restaurants"
    COMMERCIAL = "commercial"
    TRANSPORTATION = "transportation"
    RECREATION = "recreation"
    TOURIST_ATTRACTION = "tourist_attraction"
    OTHER_FACILITIES = "other_facilities"


POI_CATEGORY_PRECEDENCE = (
    PoiCategory.TRANSPORTATION,
    PoiCategory.TOURIST_ATTRACTION,
    PoiCategory.RESTAURANTS,
    PoiCategory.RECREATION,
    PoiCategory.COMMERCIAL,
    PoiCategory.OTHER_FACILITIES,
)

TOURIST_ATTRACTION_TAGS = {
    "tourism": frozenset(
        {"attraction", "museum", "viewpoint", "theme_park", "zoo", "gallery", "aquarium", "artwork"}
    ),
    "historic": frozenset(
        {"castle", "fort", "monument", "memorial", "ruins", "archaeological_site", "city_gate"}
    ),
}

ACCOMMODATION_TAGS = {
    "tourism": frozenset(
        {
            "hotel",
            "motel",
            "guest_house",
            "hostel",
            "chalet",
            "apartment",
            "camp_site",
            "caravan_site",
            "wilderness_hut",
            "alpine_hut",
        }
    ),
}

VISITOR_AMENITY_TAGS = {
    "tourism": frozenset({"information", "picnic_site"}),
}

TRANSPORT_FACILITY_TAGS = {
    "amenity": frozenset({"bus_station", "ferry_terminal"}),
    "public_transport": frozenset({"station"}),
    "railway": frozenset({"station", "halt", "tram_stop"}),
    "aeroway": frozenset({"aerodrome", "terminal", "helipad"}),
}

OTHER_TRANSPORTATION_POI_TAGS = {
    "amenity": frozenset({"taxi", "fuel", "parking", "car_rental", "bicycle_rental", "charging_station"}),
    "highway": frozenset({"bus_stop", "platform"}),
}

RESTAURANT_TAGS = {
    "amenity": frozenset(
        {"restaurant", "cafe", "fast_food", "food_court", "bar", "pub", "biergarten", "ice_cream"}
    ),
    "shop": frozenset({"bakery", "confectionery", "coffee"}),
}

RECREATION_TAGS = {
    "leisure": frozenset(
        {
            "park",
            "garden",
            "playground",
            "sports_centre",
            "pitch",
            "stadium",
            "swimming_pool",
            "fitness_centre",
            "golf_course",
            "marina",
            "water_park",
            "dog_park",
            "track",
        }
    ),
    "amenity": frozenset({"cinema", "theatre", "nightclub", "casino", "arts_centre"}),
}

COMMERCIAL_TAGS = {
    "shop": frozenset({"*"}),
    "office": frozenset({"*"}),
    "amenity": frozenset({"bank", "atm", "bureau_de_change", "marketplace"}),
}

OTHER_FACILITY_TAGS = {
    "amenity": frozenset(
        {
            "hospital",
            "clinic",
            "doctors",
            "dentist",
            "pharmacy",
            "veterinary",
            "police",
            "fire_station",
            "post_office",
            "townhall",
            "courthouse",
            "library",
            "place_of_worship",
            "school",
            "college",
            "university",
            "kindergarten",
            "community_centre",
            "social_facility",
            "toilets",
            "drinking_water",
            "shelter",
            "recycling",
            "waste_basket",
        }
    ),
    "leisure": frozenset({"beach_resort"}),
}

POI_TAG_UNIVERSE = (
    "amenity",
    "shop",
    "tourism",
    "leisure",
    "office",
    "historic",
    "craft",
    "public_transport",
    "railway",
    "aeroway",
)

POI_HIGHWAY_VALUES = frozenset({"bus_stop", "platform"})

CATEGORY_TAG_RULES = {
    PoiCategory.TOURIST_ATTRACTION: TOURIST_ATTRACTION_TAGS,
    PoiCategory.TRANSPORTATION: {
        **TRANSPORT_FACILITY_TAGS,
        **OTHER_TRANSPORTATION_POI_TAGS,
    },
    PoiCategory.RESTAURANTS: RESTAURANT_TAGS,
    PoiCategory.RECREATION: RECREATION_TAGS,
    PoiCategory.COMMERCIAL: COMMERCIAL_TAGS,
    PoiCategory.OTHER_FACILITIES: OTHER_FACILITY_TAGS,
}


def matches_tag_rule(tags: dict[str, str], rule: dict[str, frozenset[str]]) -> bool:
    for key, allowed_values in rule.items():
        value = tags.get(key)
        if value is None:
            continue
        if "*" in allowed_values or value in allowed_values:
            return True
    return False


def is_poi(tags: dict[str, str]) -> bool:
    if any(key in tags for key in POI_TAG_UNIVERSE):
        return True
    return tags.get("highway") in POI_HIGHWAY_VALUES


def categorize_poi(tags: dict[str, str]) -> PoiCategory | None:
    if not is_poi(tags):
        return None
    for category in POI_CATEGORY_PRECEDENCE:
        if matches_tag_rule(tags, CATEGORY_TAG_RULES[category]):
            return category
    return PoiCategory.OTHER_FACILITIES


def is_tourist_attraction(tags: dict[str, str]) -> bool:
    return matches_tag_rule(tags, TOURIST_ATTRACTION_TAGS)


def is_transport_facility(tags: dict[str, str]) -> bool:
    return matches_tag_rule(tags, TRANSPORT_FACILITY_TAGS)
