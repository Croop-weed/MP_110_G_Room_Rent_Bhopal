"""Synthetic tests for preprocessing and unknown-category handling."""

import pandas as pd

from pg_rent.preprocessing import build_preprocessor, features_and_target


def test_preprocessing_imputes_and_encodes_synthetic_data():
    df = pd.DataFrame(
        {
            "distance_to_campus_km": [1.0, None, 3.0],
            "room_area_sqft": [100.0, 200.0, None],
            "sharing": [1.0, None, 3.0],
            "meals_per_day": [1.0, 2.0, None],
            "building_age_yr": [10.0, None, 20.0],
            "wifi_mbps": [100.0, None, 200.0],
            "furnishing": ["full", None, "semi"],
            "locality": ["Sector East", "Sector West", None],
            "record_id": ["R1", "R2", "R3"],
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "rent_rs": [10000.0, 12000.0, 14000.0],
        }
    )

    X, y = features_and_target(df, "rent_rs")
    assert "record_id" not in X and "date" not in X and "rent_rs" not in X

    transformed = build_preprocessor().fit_transform(X)
    assert transformed.shape[0] == 3
    assert transformed.shape[1] >= 6
    assert y is not None and len(y) == 3


def test_unknown_category_handling_does_not_error():
    train = pd.DataFrame(
        {
            "distance_to_campus_km": [1.0, 2.0],
            "room_area_sqft": [100.0, 200.0],
            "sharing": [1.0, 2.0],
            "meals_per_day": [1.0, 2.0],
            "building_age_yr": [10.0, 20.0],
            "wifi_mbps": [100.0, 200.0],
            "furnishing": ["full", "semi"],
            "locality": ["Sector East", "Sector West"],
        }
    )
    test = train.copy()
    test["furnishing"] = ["brand-new", "unknown"]
    test["locality"] = ["New Place", "Other Place"]

    preprocessor = build_preprocessor().fit(train)
    transformed = preprocessor.transform(test)
    assert transformed.shape[0] == 2
    assert all(not pd.isna(value) for value in transformed.ravel())
