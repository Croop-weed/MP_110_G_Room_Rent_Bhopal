"""Reusable scikit-learn preprocessing pipeline."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NUMERIC_FEATURES = [
    "distance_to_campus_km",
    "room_area_sqft",
    "sharing",
    "meals_per_day",
    "building_age_yr",
    "wifi_mbps",
]
CATEGORICAL_FEATURES = ["furnishing", "locality"]


def build_preprocessor() -> ColumnTransformer:
    """Build a train-fitted-only ColumnTransformer for rent features."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def features_and_target(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Separate model features from the target while excluding identifiers."""
    excluded = {"record_id", "date", target}
    features = df[[c for c in df.columns if c not in excluded]].copy()
    y = df[target].copy() if target in df.columns else None
    return features, y
