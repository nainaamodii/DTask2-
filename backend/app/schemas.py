"""
Pydantic request/response schemas for the Traffic Impact Forecasting API.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class EventIntakeRequest(BaseModel):
    event_type: str = Field(..., description="'planned' or 'unplanned'", examples=["unplanned"])
    event_cause: str = Field(..., description="e.g. vehicle_breakdown, accident, construction, protest",
                              examples=["vehicle_breakdown"])
    latitude: float = Field(..., examples=[12.9716])
    longitude: float = Field(..., examples=[77.5946])
    corridor: Optional[str] = Field("Non-corridor", description="Named arterial corridor, or 'Non-corridor'. "
                                     "Drives is_major_corridor for the ML models and the diversion engine.")
    zone: Optional[str] = Field("Unknown", description="Traffic police zone, e.g. 'Central Zone 2'")
    veh_type: Optional[str] = Field(None, description="Vehicle type, for display/logging only — "
                                     "NOT used by the duration/closure models (see notebook_features.py finding #2).")
    event_datetime: Optional[datetime] = Field(
        None, description="When the event starts/was reported. Defaults to now (UTC) if omitted.")
    requires_road_closure_hint: Optional[bool] = Field(
        None, description="Only used by the rule_based method. The ml_model method predicts this itself.")

    # --- Fields required by the notebook-trained duration/closure models ---
    authenticated: Optional[bool] = Field(
        False, description="Whether the report has been verified/authenticated.")
    at_junction: Optional[bool] = Field(False, description="Whether the event is located at a road junction.")
    priority: Optional[str] = Field(
        "High", description="Officer's intake severity assessment: 'High', 'Low', or 'Unknown'. "
        "This is a MODEL INPUT here (not an output) — see README for the reasoning.")
    direction: Optional[str] = Field(
        "north_or_other",
        description="One of: north_east, north_west, south, south_west, unknown, west, "
        "or 'north_or_other' (baseline — the trained model can't distinguish 'north' from 'south_east').")
    cargo_macro: Optional[str] = Field(
        "empty_or_none",
        description="One of: empty_or_none, general_goods, liquid_gas, municipal_waste, perishables, "
        "or 'construction_heavy' (baseline). Only meaningful for vehicle/cargo-related events.")
    breakdown_severity: Optional[str] = Field(
        "not_applicable",
        description="One of: not_applicable, minor_issue, major_failure, or 'general_breakdown' (baseline). "
        "Only meaningful when event_cause is a vehicle breakdown.")
    endlatitude: Optional[float] = Field(
        None, description="If the event spans a stretch of road, its end latitude. Omit for point events.")
    endlongitude: Optional[float] = Field(None, description="End longitude, paired with endlatitude.")


class ResourceRecommendation(BaseModel):
    manpower: int
    barricades: int
    diversions: int


class BarricadePoint(BaseModel):
    latitude: float
    longitude: float
    bearing_deg: float


class DiversionRoute(BaseModel):
    path_coords: list  # list of [lat, lon]
    distance_km: float
    est_minutes: float
    via_corridors: list[str]


class ImpactForecastResponse(BaseModel):
    method: str  # "rule_based" or "ml_model"
    impact_score: float
    severity_band: str

    predicted_duration_minutes: Optional[float] = None    # ml_model only
    predicted_road_closure: bool
    predicted_road_closure_confidence: Optional[float] = None

    # Kept for the rule_based path / backward compatibility with the dashboard's
    # legacy display logic; ml_model leaves this as None since priority is now
    # an INPUT to the models, not a prediction.
    predicted_priority: Optional[str] = None
    predicted_priority_confidence: Optional[float] = None

    recommendation: ResourceRecommendation
    barricade_points: list[BarricadePoint] = []
    diversion_routes: list[DiversionRoute] = []
    notes: Optional[str] = None


class FeedbackLogRequest(BaseModel):
    """Post-event learning loop input (Week 6) — log what actually happened
    after an event was resolved, alongside what was predicted at intake time,
    so accuracy can be tracked and fed into retraining."""
    event_id: str
    actual_road_closure: bool
    actual_duration_minutes: Optional[float] = None
    actual_priority: Optional[str] = None  # legacy / optional, not the primary target anymore
    notes: Optional[str] = None

    predicted_duration_minutes: Optional[float] = None
    predicted_road_closure: Optional[bool] = None
    predicted_road_closure_confidence: Optional[float] = None
    predicted_priority: Optional[str] = None
    predicted_priority_confidence: Optional[float] = None
    method: Optional[str] = None  # "rule_based" or "ml_model" — which forecast this compares to
