"""PSI, locality MAE, chronological retraining, promotion, and holdout utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from .cleaning import clean_data
from .preprocessing import build_preprocessor, features_and_target


ROOT = Path(__file__).resolve().parents[2]
NUMERIC_FEATURES = [
    "distance_to_campus_km",
    "room_area_sqft",
    "sharing",
    "meals_per_day",
    "building_age_yr",
    "wifi_mbps",
]


def calculate_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calculate PSI using quantile bins derived only from the reference data."""
    reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    current_values = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if reference_values.size == 0 or current_values.size == 0:
        return float("inf")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference_values, quantiles))
    if len(edges) == 1:
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    elif edges.size < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference_values, bins=edges)
    cur_counts, _ = np.histogram(current_values, bins=edges)
    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def calculate_all_psi(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Calculate 10-bin PSI for every configured numeric feature."""
    rows = []
    for column in NUMERIC_FEATURES:
        if column in reference and column in current:
            rows.append(
                {
                    "feature": column,
                    "psi": calculate_psi(reference[column], current[column], 10),
                }
            )
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def locality_mae(model: Any, later: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Evaluate v1 MAE separately for every locality."""
    rows = []
    X, y = features_and_target(later, "rent_rs")
    if y is None:
        raise ValueError("later.csv must contain rent_rs")
    for locality, indices in later.groupby("locality", sort=True).groups.items():
        local_X = X.loc[indices]
        local_y = y.loc[indices]
        pred = model.predict(local_X)
        rows.append(
            {
                "locality": locality,
                "row_count": len(indices),
                "mae": float(mean_absolute_error(local_y, pred)),
            }
        )
    result = pd.DataFrame(rows).sort_values("mae", ascending=False, ignore_index=True)
    worst = str(result.iloc[0]["locality"])
    return result, worst


def chronological_split(
    later: pd.DataFrame, first_fraction: float = 0.70
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort later data by date and split it chronologically without leakage."""
    sorted_later = later.copy()
    sorted_later["date"] = pd.to_datetime(sorted_later["date"])
    sorted_later = sorted_later.sort_values(
        ["date", "record_id"], kind="stable"
    ).reset_index(drop=True)
    split_index = int(len(sorted_later) * first_fraction)
    split_index = max(1, min(split_index, len(sorted_later) - 1))
    return sorted_later.iloc[:split_index].copy(), sorted_later.iloc[
        split_index:
    ].copy()


def sha256_file(path: str | Path) -> str:
    """Compute a dataset SHA-256 from original bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promotion_decision(
    v1_mae: float, v2_mae: float, threshold: float = 0.15
) -> dict[str, Any]:
    """Apply the assignment's relative 15% improvement promotion rule."""
    relative_improvement = (v1_mae - v2_mae) / v1_mae if v1_mae != 0 else 0.0
    promote = relative_improvement >= threshold
    return {
        "promote": promote,
        "decision": "promote" if promote else "keep",
        "relative_improvement": float(relative_improvement),
    }


def run_monitoring_and_retraining() -> dict[str, Any]:
    """Compute PSI, locality MAE, retrain v2, apply promotion, and write holdout predictions."""
    train_path = ROOT / "data" / "train.csv"
    later_path = ROOT / "data" / "later.csv"
    holdout_path = ROOT / "data" / "holdout_features.csv"

    train_clean = clean_data(pd.read_csv(train_path))
    later_clean = clean_data(pd.read_csv(later_path))
    holdout_original = pd.read_csv(holdout_path)

    psi = calculate_all_psi(train_clean, later_clean)
    psi_path = ROOT / "reports" / "psi_results.csv"
    psi_path.parent.mkdir(parents=True, exist_ok=True)
    psi.to_csv(psi_path, index=False)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    v1 = mlflow.pyfunc.load_model("models:/pg-rent-g187@champion")
    locality, worst = locality_mae(v1, later_clean)
    locality_path = ROOT / "reports" / "locality_mae.csv"
    locality.to_csv(locality_path, index=False)

    first_later, final_later = chronological_split(later_clean, 0.70)
    retrain = pd.concat([train_clean, first_later], ignore_index=True)
    X_retrain, y_retrain = features_and_target(retrain, "rent_rs")
    X_check, y_check = features_and_target(final_later, "rent_rs")

    v2 = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=300, max_depth=15, random_state=42, n_jobs=-1
                ),
            ),
        ]
    )
    v2.fit(X_retrain, y_retrain)
    v1_check = v1.predict(X_check)
    v2_check = v2.predict(X_check)
    v1_mae = float(mean_absolute_error(y_check, v1_check))
    v2_mae = float(mean_absolute_error(y_check, v2_check))
    promotion = promotion_decision(v1_mae, v2_mae)

    mlflow.set_experiment("pg-rent-stage4-g187")
    with mlflow.start_run(run_name="v1-v2-comparison"):
        mlflow.log_param("data_sha256_train", sha256_file(train_path))
        mlflow.log_param("data_sha256_later", sha256_file(later_path))
        mlflow.log_param("later_split_fraction", 0.70)
        mlflow.log_param("promotion_threshold", 0.15)
        mlflow.log_param("n_estimators_v2", 300)
        mlflow.log_param("max_depth_v2", 15)
        mlflow.log_metric("mae_v1_check", v1_mae)
        mlflow.log_metric("mae_v2_check", v2_mae)
        mlflow.log_metric("relative_mae_improvement", promotion["relative_improvement"])

    champion_name = "pg-rent-g187"
    champion_version = 1
    if promotion["promote"]:
        with mlflow.start_run(run_name="v2-promoted") as v2_run:
            mlflow.log_param("data_sha256_train", sha256_file(train_path))
            mlflow.log_param("data_sha256_later", sha256_file(later_path))
            mlflow.log_param("n_estimators", 300)
            mlflow.log_param("max_depth", 15)
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("mae_v2_check", v2_mae)
            mlflow.sklearn.log_model(
                v2, artifact_path="model", serialization_format="cloudpickle"
            )
            model_uri = "runs:/" + v2_run.info.run_id + "/model"
            registered = mlflow.register_model(model_uri=model_uri, name=champion_name)
            champion_version = int(registered.version)
            MlflowClient().set_registered_model_alias(
                champion_name, "champion", champion_version
            )
        champion = mlflow.pyfunc.load_model("models:/pg-rent-g187@champion")
    else:
        champion = v1

    holdout_clean = clean_data(holdout_original)
    holdout_X, _ = features_and_target(holdout_clean, "rent_rs")
    predictions = champion.predict(holdout_X)
    prediction_values = np.asarray(predictions, dtype=float).ravel()
    if len(prediction_values) != len(holdout_original):
        raise RuntimeError("Prediction count differs from holdout row count")
    if not np.isfinite(prediction_values).all():
        raise RuntimeError("Holdout predictions contain missing or infinite values")

    predictions_path = ROOT / "predictions" / "holdout_predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"record_id": holdout_original["record_id"], "prediction": prediction_values}
    ).to_csv(predictions_path, index=False)

    import yaml

    findings = yaml.safe_load((ROOT / "findings.yaml").read_text(encoding="utf-8"))
    findings["monitoring"] = {
        "psi": {row["feature"]: float(row["psi"]) for _, row in psi.iterrows()},
        "most_drifted_feature": str(psi.iloc[0]["feature"]),
        "mae_by_group": {
            str(row["locality"]): float(row["mae"]) for _, row in locality.iterrows()
        },
        "worst_served_group": worst,
        "mae_v1_on_check": v1_mae,
        "mae_v2_on_check": v2_mae,
        "decision": promotion["decision"],
    }
    findings["tracking"]["champion_version"] = champion_version
    (ROOT / "findings.yaml").write_text(
        yaml.safe_dump(findings, sort_keys=False), encoding="utf-8"
    )

    return {
        "psi": psi.to_dict("records"),
        "most_drifted_feature": str(psi.iloc[0]["feature"]),
        "locality_mae": locality.to_dict("records"),
        "worst_served_group": worst,
        "v1_mae": v1_mae,
        "v2_mae": v2_mae,
        "promotion": promotion,
        "champion_version": champion_version,
        "holdout_rows": len(prediction_values),
        "holdout_path": str(predictions_path),
    }


if __name__ == "__main__":
    print(run_monitoring_and_retraining())
