"""Synthetic configuration and basic training behavior tests."""

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from pg_rent.config import load_config
from pg_rent.preprocessing import build_preprocessor, features_and_target
from pg_rent.training import sha256_file, model_from_config


def test_configuration_loads_required_sections(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text(
        "data:\n  train_csv: data/train.csv\n  target_column: rent_rs\nsplit:\n  validation_ratio: 0.2\n  random_seed: 42\nmlflow:\n  experiment_name: test-exp\n  registered_model_name: test-model\nmodels:\n  random_forest_200:\n    n_estimators: 200\n    max_depth: 10\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config["data"]["target_column"] == "rent_rs"
    assert config["split"]["random_seed"] == 42
    assert config["mlflow"]["experiment_name"] == "test-exp"


def test_basic_pipeline_fits_and_predicts():
    df = pd.DataFrame(
        {
            "distance_to_campus_km": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "room_area_sqft": [100, 150, 200, 250, 300, 350],
            "sharing": [1, 2, 1, 2, 3, 2],
            "meals_per_day": [0, 1, 2, 1, 3, 2],
            "building_age_yr": [5, 10, 15, 20, 25, 30],
            "wifi_mbps": [50, 100, 150, 200, 250, 300],
            "furnishing": [
                "unfurnished",
                "semi",
                "full",
                "semi",
                "full",
                "unfurnished",
            ],
            "locality": [
                "Sector East",
                "Sector West",
                "Market Side",
                "Sector East",
                "Sector West",
                "Market Side",
            ],
            "rent_rs": [10000, 12000, 14000, 16000, 18000, 20000],
        }
    )
    X, y = features_and_target(df, "rent_rs")
    model = model_from_config(
        "random_forest_200", {"n_estimators": 10, "max_depth": 5, "random_state": 42}
    )
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([("preprocessor", build_preprocessor()), ("regressor", model)])
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    pipeline.fit(X_train, y_train)
    assert pipeline.predict(X_test).shape == (2,)


def test_sha256_is_stable(tmp_path):
    file = tmp_path / "sample.csv"
    file.write_bytes(b"a,b\n1,2\n")
    assert sha256_file(file) == hashlib.sha256(b"a,b\n1,2\n").hexdigest()
