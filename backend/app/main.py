"""
Event-Driven Traffic Congestion Forecasting — Backend API
================================================================
Endpoints:
  POST /forecast            -> impact forecast + resource recommendation for one event
  POST /feedback             -> post-event learning loop (Week 6)
  GET  /health                -> liveness check
  GET  /model-info             -> which models are loaded + their offline metrics

By default /forecast uses the notebook-trained ML models (congestion duration
regressor + road closure classifier, see app/ml_engine.py and
app/notebook_features.py). Pass ?method=rule_based to force the Week 2
rule-based baseline instead — useful for side-by-side demos.

Run (from the backend/ directory):
    uvicorn app.main:app --reload --port 8000

Then try:
    http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.schemas import EventIntakeRequest, ImpactForecastResponse, ResourceRecommendation, FeedbackLogRequest, BarricadePoint, DiversionRoute
from app import rule_engine
from app.ml_engine import get_ml_engine
from app.road_graph import get_road_graph_engine
from app import feedback_store

HERE = os.path.dirname(__file__)
METRICS_PATH = os.path.join(HERE, "..", "..", "outputs", "model_metrics.json")

app = FastAPI(
    title="Traffic Congestion Forecasting API",
    description="Event-driven traffic impact forecasting and resource recommendation prototype.",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Feedback persistence now backed by SQLite (see app/feedback_store.py, Week 6).
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI()

STATIC_DIR = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    info = {
        "active_models": {
            "duration_model": "models/notebook/xgboost_duration_model.pkl (XGBRegressor)",
            "closure_model": "models/notebook/xgboost_closure_model.pkl (XGBClassifier)",
        },
        "known_findings": [
            "Duration model's training target had a units bug (seconds mislabeled as "
            "minutes in the source notebook) — corrected at inference time, see "
            "app/notebook_features.py and app/ml_engine.py docstrings.",
            "veh_type and priority_High are not used by the fitted models, despite "
            "appearing in earlier notebook cells.",
            "direction dummies only cover 6 of 8 real-world directions; 'north' and "
            "'south_east' fall back to the model's baseline category.",
        ],
    }
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            info["legacy_week3_metrics"] = json.load(f)
    return info


def _build_spatial_recs(event: EventIntakeRequest, rec: dict):
    """Compute barricade containment points + diversion routes for a forecasted event.
    Always attempted; degrades gracefully (empty lists) if the corridor isn't
    represented in the road graph (e.g. 'Non-corridor' events, or very sparse corridors)."""
    road_engine = get_road_graph_engine()

    barricades = road_engine.barricade_points(
        event.latitude, event.longitude, n_barricades=rec["barricades"], radius_km=0.3)

    diversions = []
    corridor = event.corridor or "Non-corridor"
    if corridor != "Non-corridor" and rec["diversions"] > 0:
        raw = road_engine.diversions(event.latitude, event.longitude, corridor,
                                      top_n=min(rec["diversions"], 3))
        for d in raw:
            diversions.append(DiversionRoute(
                path_coords=[list(c) for c in d["path_coords"]],
                distance_km=d["distance_km"],
                est_minutes=d["est_minutes"],
                via_corridors=d["via_corridors"],
            ))

    barricade_pts = [BarricadePoint(**b) for b in barricades]
    return barricade_pts, diversions


@app.post("/forecast", response_model=ImpactForecastResponse)
def forecast(event: EventIntakeRequest, method: str = Query("ml_model", enum=["ml_model", "rule_based"])):
    if method == "rule_based":
        time_window = rule_engine.time_multiplier  # not used directly; compute below
        # Need a time_window string for the rule engine - derive simply from hour if provided
        hour = (event.event_datetime.hour if event.event_datetime else 12)
        if 8 <= hour <= 11:
            tw = "morning_peak"
        elif 17 <= hour <= 21:
            tw = "evening_peak"
        elif hour >= 22 or hour <= 5:
            tw = "night"
        else:
            tw = "off_peak"

        is_corridor_road = (event.corridor or "Non-corridor") != "Non-corridor"
        is_weekend = False
        requires_closure = bool(event.requires_road_closure_hint) if event.requires_road_closure_hint is not None else False

        result = rule_engine.score_event(
            event_cause=event.event_cause,
            is_corridor_road=is_corridor_road,
            time_window=tw,
            is_weekend=is_weekend,
            requires_road_closure=requires_closure,
        )
        rec = {"manpower": result.manpower, "barricades": result.barricades, "diversions": result.diversions}
        barricade_pts, diversion_routes = _build_spatial_recs(event, rec)

        return ImpactForecastResponse(
            method="rule_based",
            impact_score=result.impact_score,
            severity_band=result.severity_band,
            predicted_priority="High" if result.impact_score >= 6 else "Low",
            predicted_road_closure=requires_closure,
            recommendation=ResourceRecommendation(**rec),
            barricade_points=barricade_pts,
            diversion_routes=diversion_routes,
            notes="Computed via Week 2 weighted rule engine (no trained model).",
        )

    # --- ML model path (default) — notebook-trained duration + closure models ---
    engine = get_ml_engine()
    pred = engine.predict(event)
    rec = rule_engine.recommend_resources(pred["impact_score"], pred["predicted_closure"])
    barricade_pts, diversion_routes = _build_spatial_recs(event, rec)

    return ImpactForecastResponse(
        method="ml_model",
        impact_score=pred["impact_score"],
        severity_band=pred["severity_band"],
        predicted_duration_minutes=pred["duration_minutes"],
        predicted_road_closure=pred["predicted_closure"],
        predicted_road_closure_confidence=round(pred["closure_proba"], 3),
        recommendation=ResourceRecommendation(**rec),
        barricade_points=barricade_pts,
        diversion_routes=diversion_routes,
        notes=(f"Predicted congestion duration: {pred['duration_minutes']} min "
               f"(model raw output {pred['raw_duration_seconds']}s, corrected from a units bug "
               f"in the source notebook — see notebook_features.py). "
               f"Computed via xgboost_duration_model.pkl + xgboost_closure_model.pkl."),
    )


@app.post("/feedback")
def feedback(entry: FeedbackLogRequest):
    """Week 6 post-event learning loop: logs actuals (predicted vs actual) to SQLite
    for the monthly retraining batch (see scripts/07_retrain_with_feedback.py)."""
    row_id = feedback_store.insert_feedback(entry.model_dump())
    return {"status": "logged", "id": row_id, "total_feedback_entries": feedback_store.count_feedback()}


@app.get("/feedback")
def list_feedback():
    return {"entries": feedback_store.list_feedback()}


@app.get("/feedback/stats")
def feedback_stats():
    """Quick view of how predictions are tracking against logged actuals — this is
    the same delta the monthly retraining job consumes."""
    return feedback_store.summary_stats()


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

@app.get("/{catchall:path}")
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "index.html not found"}