"""FastAPI application for the PG room rent prediction service."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from app.logging_config import RequestIdMiddleware, configure_logging
from app.model_loader import ChampionModelLoader, ModelLoadingError
from app.schemas import PredictionRequest, PredictionResponse


MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/pg-rent-g187@champion")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

configure_logging()
LOGGER = logging.getLogger("pg_rent.application")


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the champion model once at startup and release references at shutdown."""
    LOGGER.info("Application startup")
    try:
        application.state.model_loader.load()
    except ModelLoadingError:
        LOGGER.exception("Application startup model loading failed")
    yield
    LOGGER.info("Application shutdown")


app = FastAPI(title="PG Room Rent API", version="1.0", lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)
app.state.model_loader = ChampionModelLoader(MODEL_URI, TRACKING_URI)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness and readiness endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, http_request: Request) -> PredictionResponse:
    """Predict monthly rent using the complete registered preprocessing pipeline."""
    request_id = getattr(http_request.state, "request_id", None)
    LOGGER.info("Prediction request", extra={"request_id": request_id})
    try:
        model = app.state.model_loader.load()
        frame = pd.DataFrame([request.model_dump()])
        prediction = model.predict(frame)
        value = float(prediction[0])
        if not (value == value) or value in (float("inf"), float("-inf")):
            raise ValueError("Model returned a non-finite prediction")
        LOGGER.info(
            "Prediction result",
            extra={"request_id": request_id, "prediction": value},
        )
        return PredictionResponse(prediction=value)
    except ModelLoadingError as exc:
        LOGGER.exception("Model loading failure")
        raise HTTPException(
            status_code=503, detail="Champion model unavailable"
        ) from exc
    except Exception as exc:
        LOGGER.exception("Prediction failure")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
