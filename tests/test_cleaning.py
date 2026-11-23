"""Synthetic tests for verified cleaning corrections."""

import pandas as pd

from pg_rent.cleaning import clean_data


def test_cleaning_applies_unit_sentinel_target_and_spelling_corrections():
    df = pd.DataFrame(
        {
            "record_id": ["R00001", "R00224", "R00224", "R99999"],
            "locality": ["Sector West", "Market Side", "Market Side", "Sector East"],
            "building_age_yr": [120.0, 20.0, 20.0, 30.0],
            "sharing": [999, 2, 2, 3],
            "rent_rs": [10000, 88680, 88680, 12000],
            "furnishing": ["FUL.", " full ", " full ", "SEMI"],
        }
    )

    cleaned = clean_data(df)

    assert cleaned is not df
    assert cleaned.loc[0, "building_age_yr"] == 10.0
    assert pd.isna(cleaned.loc[0, "sharing"])
    assert cleaned.loc[1, "rent_rs"] == 8868.0
    assert cleaned["furnishing"].tolist() == ["full", "full", "semi"]


def test_cleaning_handles_duplicate_removal_and_missing_target_safely():
    df = pd.DataFrame(
        {
            "record_id": ["R1", "R1", "R2"],
            "locality": ["Market Side", "Market Side", "Sector West"],
            "building_age_yr": [10.0, 10.0, 24.0],
            "sharing": [1.0, 1.0, None],
            "furnishing": [None, None, "FULL"],
        }
    )
    cleaned = clean_data(df)
    assert len(cleaned) == 2
    assert cleaned.loc[2, "building_age_yr"] == 2.0
