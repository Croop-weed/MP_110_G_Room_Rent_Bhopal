"""Pydantic request and response schemas."""

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Exactly the eight fields accepted by the PG rent prediction service."""

    model_config = ConfigDict(extra="forbid")

    distance_to_campus_km: float = Field(..., ge=0)
    room_area_sqft: float = Field(..., ge=0, le=650)
    sharing: int = Field(..., ge=1)
    meals_per_day: int = Field(..., ge=0)
    building_age_yr: float = Field(..., ge=0)
    wifi_mbps: float = Field(..., ge=0)
    furnishing: str
    locality: str


class PredictionResponse(BaseModel):
    """Monthly rent prediction in rupees."""

    prediction: float
