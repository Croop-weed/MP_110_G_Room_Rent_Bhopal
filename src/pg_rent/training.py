"""MLflow training and registration for PG rent Stage 2."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .cleaning import clean_data, generate_cleaning_report
from .config import load_config
from .preprocessing import build_preprocessor, features_and_target


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 from original file bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_from_config(name: str, params: dict[str, Any]):
    """Create a configured scikit-learn regression model."""
    if name.startswith("random_forest"):
        return RandomForestRegressor(**params)
    if name.startswith("extra_trees"):
        return ExtraTreesRegressor(**params)
    raise ValueError(f"Unsupported model configuration: {name}")


def train() -> dict[str, Any]:
    """Clean data, run at least three MLflow experiments, and register the champion."""
    config = load_config(ROOT / "params.yaml")
    train_path = ROOT / config["data"]["train_csv"]
    target = config["data"]["target_column"]
    split_ratio = float(config["split"]["validation_ratio"])
    seed = int(config["split"]["random_seed"])
    experiment_name = config["mlflow"]["experiment_name"]
    model_name = config["mlflow"]["registered_model_name"]

    original_hash = sha256_file(train_path)
    original = pd.read_csv(train_path)
    cleaned = clean_data(original)
    generate_cleaning_report(
        original, cleaned, ROOT / "reports" / "cleaning_report.txt"
    )

    X, y = features_and_target(cleaned, target)
    if y is None:
        raise ValueError("Training data must contain the target column")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=split_ratio, random_state=seed
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    run_results: list[dict[str, Any]] = []
    for config_name, model_params in config["models"].items():
        model_params = dict(model_params)
        model_params["random_state"] = seed
        model = model_from_config(config_name, model_params)
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("regressor", model),
            ]
        )

        with mlflow.start_run(run_name=config_name) as run:
            mlflow.log_param("data_sha256", original_hash)
            mlflow.log_param("model_name", config_name)
            for key, value in model_params.items():
                mlflow.log_param(key, value)

            pipeline.fit(X_train, y_train)
            prediction = pipeline.predict(X_val)
            mae = float(mean_absolute_error(y_val, prediction))
            mlflow.log_metric("mae", mae)
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            run_results.append(
                {
                    "config_name": config_name,
                    "run_id": run.info.run_id,
                    "mae": mae,
                    "model_uri": f"runs:/{{{run.info.run_id}}}/model".replace(
                        "{", ""
                    ).replace("}", ""),
                }
            )

    winner = min(run_results, key=lambda item: item["mae"])
    model_uri = f"runs:/{{{winner['run_id']}}}/model".replace("{", "").replace("}", "")

    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "champion", registered.version)

    findings = yaml.safe_load((ROOT / "findings.yaml").read_text(encoding="utf-8"))
    findings["tracking"] = {
        "experiment_name": experiment_name,
        "best_run_id": winner["run_id"],
        "registered_model_name": model_name,
        "champion_version": int(registered.version),
    }
    (ROOT / "findings.yaml").write_text(
        yaml.safe_dump(findings, sort_keys=False), encoding="utf-8"
    )

    print(f"Experiment name: {experiment_name}")
    for result in run_results:
        print(
            f"Run ID: {result['run_id']} MAE: {result['mae']:.6f} model: {result['config_name']}"
        )
    print(f"Winning run ID: {winner['run_id']}")
    print(f"Winning MAE: {winner['mae']:.6f}")
    print(f"Registered model name: {model_name}")
    print(f"Registered version: {registered.version}")
    print("Champion alias: champion")

    return {
        "runs": run_results,
        "winner": winner,
        "registered_model_name": model_name,
        "version": registered.version,
        "data_sha256": original_hash,
    }


if __name__ == "__main__":
    train()
