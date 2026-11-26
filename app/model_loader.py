"""Robust MLflow champion model loading."""

from __future__ import annotations

import joblib
import logging
from pathlib import Path
from typing import Any, Protocol

import mlflow


LOGGER = logging.getLogger("pg_rent.application")


class PredictionModel(Protocol):
    def predict(self, frame: Any) -> Any: ...


class ModelLoadingError(RuntimeError):
    """Raised when the registered champion cannot be loaded."""


class ChampionModelLoader:
    """Load the registered champion once and reuse it across requests."""

    def __init__(self, model_uri: str, tracking_uri: str) -> None:
        self.model_uri = model_uri
        self.tracking_uri = tracking_uri
        self._model: PredictionModel | None = None

    def load(self) -> PredictionModel:
        if self._model is not None:
            return self._model
        LOGGER.info("Loading champion model", extra={"model_uri": self.model_uri})
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            self._model = mlflow.pyfunc.load_model(self.model_uri)
        except Exception:
            LOGGER.exception(
                "Champion model loading failed via MLflow; trying packaged artifact"
            )
            try:
                artifact_path = Path(
                    "/app/mlruns/2/models/m-9f40945653bd4f81a6091dc87c2f6ca6/artifacts/model.pkl"
                )
                if not artifact_path.exists():
                    artifact_path = (
                        Path.cwd()
                        / "mlruns/2/models/m-9f40945653bd4f81a6091dc87c2f6ca6/artifacts/model.pkl"
                    )
                self._model = joblib.load(artifact_path)
                LOGGER.info("Packaged champion artifact loaded")
                return self._model
            except Exception as exc:
                LOGGER.exception("Champion model loading failed")
                raise ModelLoadingError(
                    f"Could not load champion model {self.model_uri}"
                ) from exc
        LOGGER.info("Champion model loaded", extra={"model_uri": self.model_uri})
        return self._model
