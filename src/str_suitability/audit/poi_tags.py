import pandas as pd

from str_suitability.taxonomy import ACCOMMODATION_TAGS, POI_TAG_UNIVERSE, VISITOR_AMENITY_TAGS

CATEGORY_COLUMN = "category"
TAG_KEY_COLUMN = "poi_tag_key"
TAG_VALUE_COLUMN = "poi_tag_value"
CATCH_ALL_COLUMN = "is_catch_all"
TAG_COLUMN = "tag"

STRONG = "strong"
WEAK = "weak"
NONE = "none"

TOURIST_ACTIVITY = "tourist activity"
VISITOR_AMENITIES = "visitor amenities"
RECREATION = "recreation"
ACCESSIBILITY = "accessibility"
STR_DEMAND = "STR demand"

RELEVANCE_AXES = (
    TOURIST_ACTIVITY,
    VISITOR_AMENITIES,
    RECREATION,
    ACCESSIBILITY,
    STR_DEMAND,
)

TAG_RELEVANCE = {
    "tourism=attraction": (STRONG, "named visitor attraction", (TOURIST_ACTIVITY, STR_DEMAND)),
    "tourism=museum": (STRONG, "cultural attraction", (TOURIST_ACTIVITY, STR_DEMAND)),
    "tourism=viewpoint": (STRONG, "scenic viewpoint", (TOURIST_ACTIVITY, RECREATION)),
    "tourism=theme_park": (
        STRONG,
        "purpose-built attraction drawing overnight stays",
        (TOURIST_ACTIVITY, RECREATION, STR_DEMAND),
    ),
    "tourism=zoo": (
        STRONG,
        "purpose-built attraction drawing overnight stays",
        (TOURIST_ACTIVITY, RECREATION, STR_DEMAND),
    ),
    "tourism=gallery": (STRONG, "cultural attraction", (TOURIST_ACTIVITY, STR_DEMAND)),
    "tourism=aquarium": (STRONG, "purpose-built attraction", (TOURIST_ACTIVITY, STR_DEMAND)),
    "tourism=artwork": (
        WEAK,
        "public art, pass-through rather than a destination",
        (TOURIST_ACTIVITY,),
    ),
    "historic=castle": (STRONG, "heritage landmark", (TOURIST_ACTIVITY, STR_DEMAND)),
    "historic=fort": (STRONG, "heritage landmark", (TOURIST_ACTIVITY, STR_DEMAND)),
    "historic=monument": (STRONG, "heritage landmark", (TOURIST_ACTIVITY, STR_DEMAND)),
    "historic=memorial": (WEAK, "commemorative marker, rarely a trip purpose", (TOURIST_ACTIVITY,)),
    "historic=ruins": (STRONG, "heritage landmark", (TOURIST_ACTIVITY, STR_DEMAND)),
    "historic=archaeological_site": (STRONG, "heritage landmark", (TOURIST_ACTIVITY, STR_DEMAND)),
    "historic=city_gate": (WEAK, "heritage marker, rarely a trip purpose", (TOURIST_ACTIVITY,)),
    "amenity=bus_station": (STRONG, "public transport interchange", (ACCESSIBILITY,)),
    "amenity=ferry_terminal": (STRONG, "water access point", (ACCESSIBILITY, VISITOR_AMENITIES)),
    "public_transport=station": (STRONG, "public transport interchange", (ACCESSIBILITY,)),
    "railway=station": (STRONG, "public transport interchange", (ACCESSIBILITY,)),
    "railway=halt": (WEAK, "minor rail stop", (ACCESSIBILITY,)),
    "railway=tram_stop": (WEAK, "minor rail stop", (ACCESSIBILITY,)),
    "aeroway=aerodrome": (STRONG, "air access point", (ACCESSIBILITY,)),
    "aeroway=terminal": (STRONG, "air access point", (ACCESSIBILITY,)),
    "aeroway=helipad": (WEAK, "minor air access point", (ACCESSIBILITY,)),
    "amenity=taxi": (WEAK, "local transport service", (ACCESSIBILITY,)),
    "amenity=fuel": (WEAK, "road-trip stopping point", (ACCESSIBILITY,)),
    "amenity=parking": (WEAK, "car-borne visitor amenity", (ACCESSIBILITY, VISITOR_AMENITIES)),
    "amenity=car_rental": (WEAK, "tourist mobility service", (ACCESSIBILITY, VISITOR_AMENITIES)),
    "amenity=bicycle_rental": (WEAK, "recreational mobility service", (ACCESSIBILITY, RECREATION)),
    "amenity=charging_station": (NONE, "vehicle servicing with no visitor interpretation", ()),
    "highway=bus_stop": (WEAK, "local transport stop", (ACCESSIBILITY,)),
    "highway=platform": (WEAK, "local transport stop", (ACCESSIBILITY,)),
    "amenity=restaurant": (STRONG, "food service", (VISITOR_AMENITIES, STR_DEMAND)),
    "amenity=cafe": (STRONG, "food and drink service", (VISITOR_AMENITIES, STR_DEMAND)),
    "amenity=bar": (STRONG, "evening economy", (VISITOR_AMENITIES,)),
    "amenity=pub": (STRONG, "evening economy", (VISITOR_AMENITIES,)),
    "amenity=biergarten": (STRONG, "evening economy", (VISITOR_AMENITIES,)),
    "amenity=fast_food": (
        WEAK,
        "convenience food serving residents as much as visitors",
        (VISITOR_AMENITIES,),
    ),
    "amenity=food_court": (WEAK, "convenience food", (VISITOR_AMENITIES,)),
    "amenity=ice_cream": (WEAK, "incidental food service", (VISITOR_AMENITIES,)),
    "shop=bakery": (NONE, "retail food outlet serving residents", ()),
    "shop=confectionery": (NONE, "retail food outlet serving residents", ()),
    "shop=coffee": (WEAK, "coffee retail counter", (VISITOR_AMENITIES,)),
    "leisure=park": (STRONG, "open-space recreation", (RECREATION, VISITOR_AMENITIES)),
    "leisure=garden": (WEAK, "ornamental open space", (RECREATION,)),
    "leisure=playground": (NONE, "child play facility serving residents", ()),
    "leisure=sports_centre": (WEAK, "organised sport", (RECREATION,)),
    "leisure=pitch": (WEAK, "local sports field", (RECREATION,)),
    "leisure=stadium": (WEAK, "event venue", (RECREATION, TOURIST_ACTIVITY)),
    "leisure=swimming_pool": (WEAK, "recreational facility", (RECREATION,)),
    "leisure=fitness_centre": (NONE, "gymnasium serving residents", ()),
    "leisure=golf_course": (
        STRONG,
        "destination recreation",
        (RECREATION, TOURIST_ACTIVITY, STR_DEMAND),
    ),
    "leisure=marina": (
        STRONG,
        "berthing and waterfront recreation",
        (RECREATION, VISITOR_AMENITIES, STR_DEMAND),
    ),
    "leisure=water_park": (
        STRONG,
        "destination recreation",
        (RECREATION, TOURIST_ACTIVITY, STR_DEMAND),
    ),
    "leisure=dog_park": (NONE, "facility serving residents", ()),
    "leisure=track": (NONE, "athletics facility serving residents", ()),
    "amenity=cinema": (WEAK, "commercial entertainment", (RECREATION,)),
    "amenity=theatre": (WEAK, "cultural entertainment", (RECREATION, TOURIST_ACTIVITY)),
    "amenity=nightclub": (WEAK, "evening economy", (VISITOR_AMENITIES,)),
    "amenity=casino": (STRONG, "destination entertainment", (RECREATION, TOURIST_ACTIVITY, STR_DEMAND)),
    "amenity=arts_centre": (WEAK, "cultural venue", (RECREATION, VISITOR_AMENITIES)),
    "amenity=bank": (NONE, "resident banking service", ()),
    "amenity=atm": (WEAK, "cash access for visitors", (VISITOR_AMENITIES,)),
    "amenity=bureau_de_change": (WEAK, "foreign visitor service", (VISITOR_AMENITIES,)),
    "amenity=marketplace": (
        WEAK,
        "public market, partly visitor-facing",
        (VISITOR_AMENITIES, TOURIST_ACTIVITY),
    ),
    "shop=souvenir": (STRONG, "visitor purchase", (VISITOR_AMENITIES, TOURIST_ACTIVITY)),
    "shop=gift": (WEAK, "visitor purchase", (VISITOR_AMENITIES,)),
    "shop=art": (WEAK, "visitor purchase", (VISITOR_AMENITIES, TOURIST_ACTIVITY)),
    "shop=travel_agency": (WEAK, "visitor service", (VISITOR_AMENITIES,)),
    "amenity=hospital": (NONE, "resident health service", ()),
    "amenity=clinic": (NONE, "resident health service", ()),
    "amenity=doctors": (NONE, "resident health service", ()),
    "amenity=dentist": (NONE, "resident health service", ()),
    "amenity=pharmacy": (WEAK, "health service useful to visitors", (VISITOR_AMENITIES,)),
    "amenity=veterinary": (NONE, "resident animal health service", ()),
    "amenity=police": (NONE, "civil order service", ()),
    "amenity=fire_station": (NONE, "emergency service", ()),
    "amenity=post_office": (NONE, "resident service", ()),
    "amenity=townhall": (NONE, "government administration", ()),
    "amenity=courthouse": (NONE, "government administration", ()),
    "amenity=library": (NONE, "resident service", ()),
    "amenity=place_of_worship": (
        WEAK,
        "heritage places of worship draw visitors, generic worship is local",
        (TOURIST_ACTIVITY,),
    ),
    "amenity=school": (NONE, "resident education facility", ()),
    "amenity=college": (NONE, "resident education facility", ()),
    "amenity=university": (NONE, "resident education facility", ()),
    "amenity=kindergarten": (NONE, "resident education facility", ()),
    "amenity=community_centre": (NONE, "resident community facility", ()),
    "amenity=social_facility": (NONE, "resident social service", ()),
    "amenity=toilets": (WEAK, "visitor amenity", (VISITOR_AMENITIES,)),
    "amenity=drinking_water": (WEAK, "visitor amenity", (VISITOR_AMENITIES,)),
    "amenity=shelter": (WEAK, "visitor amenity", (VISITOR_AMENITIES,)),
    "amenity=recycling": (NONE, "waste facility with no visitor interpretation", ()),
    "amenity=waste_basket": (NONE, "street furniture with no visitor interpretation", ()),
    "leisure=beach_resort": (
        STRONG,
        "waterfront resort",
        (TOURIST_ACTIVITY, VISITOR_AMENITIES, STR_DEMAND),
    ),
    "tourism=picnic_site": (WEAK, "picnic area", (RECREATION, VISITOR_AMENITIES)),
}

TAG_KEY_FALLBACK = {
    "shop": (NONE, "retail or service outlet serving residents", ()),
    "office": (NONE, "administrative or professional office", ()),
    "craft": (NONE, "workshop or craft trade", ()),
    "amenity": (WEAK, "amenity with no visitor interpretation in the taxonomy", (VISITOR_AMENITIES,)),
    "leisure": (
        WEAK,
        "leisure facility with no visitor interpretation in the taxonomy",
        (RECREATION,),
    ),
    "historic": (
        WEAK,
        "heritage object with no visitor interpretation in the taxonomy",
        (TOURIST_ACTIVITY,),
    ),
    "public_transport": (WEAK, "public transport element", (ACCESSIBILITY,)),
    "railway": (WEAK, "rail element", (ACCESSIBILITY,)),
    "aeroway": (WEAK, "air transport element", (ACCESSIBILITY,)),
    "highway": (WEAK, "street element", (ACCESSIBILITY,)),
}

ACCOMMODATION_VALUES = ACCOMMODATION_TAGS["tourism"]
VISITOR_AMENITY_VALUES = VISITOR_AMENITY_TAGS["tourism"]

VISITOR_DENSITY_CATEGORIES = ("tourist_attraction", "restaurants", "recreation")


def tag_relevance(key: str, value: str) -> tuple[str, str, tuple[str, ...]]:
    entry = TAG_RELEVANCE.get(f"{key}={value}")
    if entry is not None:
        return entry
    if key == "tourism":
        if value in ACCOMMODATION_VALUES:
            return (
                STRONG,
                "overnight accommodation, the supply side of STR demand",
                (STR_DEMAND, VISITOR_AMENITIES),
            )
        if value in VISITOR_AMENITY_VALUES:
            return (STRONG, "visitor information or picnic provision", (VISITOR_AMENITIES, TOURIST_ACTIVITY))
        return (
            WEAK,
            "tourism object that is neither attraction nor accommodation",
            (TOURIST_ACTIVITY,),
        )
    return TAG_KEY_FALLBACK.get(key, (NONE, f"unrecognised {key} value", ()))


def tag_inventory(pois: pd.DataFrame, total: int | None = None) -> pd.DataFrame:
    denominator = int(len(pois)) if total is None else int(total)
    grouped = (
        pois.groupby([CATEGORY_COLUMN, TAG_KEY_COLUMN, TAG_VALUE_COLUMN], dropna=False)
        .agg(count=("osm_id", "size"), catch_all=(CATCH_ALL_COLUMN, "all"))
        .reset_index()
    )
    grouped["share_of_total"] = grouped["count"] / denominator
    grouped["share_of_category"] = grouped["count"] / grouped.groupby(CATEGORY_COLUMN)["count"].transform("sum")

    verdicts = [
        tag_relevance(key, value)
        for key, value in zip(grouped[TAG_KEY_COLUMN], grouped[TAG_VALUE_COLUMN])
    ]
    grouped["relevance_tier"] = [verdict[0] for verdict in verdicts]
    grouped["relevance_reason"] = [verdict[1] for verdict in verdicts]
    grouped["relevance_axes"] = ["; ".join(verdict[2]) for verdict in verdicts]
    return grouped.sort_values(["count", CATEGORY_COLUMN], ascending=[False, True]).reset_index(drop=True)


def catch_all_breakdown(pois: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    other = pois[pois[CATEGORY_COLUMN] == "other_facilities"]
    catch_all = other[other[CATCH_ALL_COLUMN]]
    explicit = other[~other[CATCH_ALL_COLUMN]]

    ranked = (
        catch_all.groupby([TAG_KEY_COLUMN, TAG_VALUE_COLUMN], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    tiers = [
        tag_relevance(key, value)[0]
        for key, value in zip(ranked[TAG_KEY_COLUMN], ranked[TAG_VALUE_COLUMN])
    ]
    ranked["relevance_tier"] = tiers
    ranked["share_of_other_facilities"] = ranked["count"] / len(other)

    strongly_relevant = ranked.loc[ranked["relevance_tier"] == STRONG, "count"].sum()
    summary = {
        "other_facilities": int(len(other)),
        "other_facilities_share_of_total": float(len(other) / len(pois)),
        "explicit_list": int(len(explicit)),
        "catch_all": int(len(catch_all)),
        "catch_all_share_of_other_facilities": float(len(catch_all) / len(other)),
        "catch_all_strongly_relevant": int(strongly_relevant),
    }
    return ranked, summary


def missing_universe_keys(pois: pd.DataFrame) -> list[str]:
    present = set(pois[TAG_KEY_COLUMN].dropna())
    return [key for key in POI_TAG_UNIVERSE if key not in present]


def tag_value_mask(pois: pd.DataFrame, key: str, values) -> pd.Series:
    return (pois[TAG_KEY_COLUMN] == key) & pois[TAG_VALUE_COLUMN].isin(set(values))
