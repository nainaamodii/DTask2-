"""
ML inference engine (v2) — wraps the notebook-trained models:
  - xgboost_duration_model.pkl  (XGBRegressor -> predicted congestion duration, in SECONDS, see notebook_features.py finding #1)
  - xgboost_closure_model.pkl   (XGBClassifier -> predicted road closure probability)

This replaces the Week 3 ASTRAM-trained priority/closure classifiers
(archived in ml_engine_legacy.py) with the user-supplied models. The framing
flips slightly: previously `priority` was a model OUTPUT; now it's an INPUT
(the officer's intake assessment), and the model instead predicts the thing
the original plan actually asked for — `peak_delay_minutes` (here:
predicted congestion duration) and whether the road needs to close.
"""

import os
import pickle
import pandas as pd

from app.notebook_features import RawEventInput, build_feature_row

HERE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(HERE, "..", "..", "models", "notebook")

# Duration -> severity band thresholds (minutes). Calibrated loosely around the
# duration distribution seen in earlier ASTRAM EDA (median ~46min, p75 ~85min
# for events that had a logged resolution time).
DURATION_BAND_THRESHOLDS = {
    "LOW": 20,        # < 20 min
    "MEDIUM": 50,      # 20-50 min
    "HIGH": 100,        # 50-100 min
    # >= 100 min -> CRITICAL
}


class NotebookMLEngine:
    def __init__(self):
        with open(os.path.join(MODEL_DIR, "xgboost_duration_model.pkl"), "rb") as f:
            self.duration_model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "xgboost_closure_model.pkl"), "rb") as f:
            self.closure_model = pickle.load(f)

    def predict(self, event) -> dict:
        """`event` is a schemas.EventIntakeRequest (or duck-typed equivalent)."""
        raw = RawEventInput(
            event_type=event.event_type,
            latitude=event.latitude,
            longitude=event.longitude,
            event_cause=event.event_cause,
            authenticated=bool(getattr(event, "authenticated", False)),
            is_major_corridor=(getattr(event, "corridor", "Non-corridor") or "Non-corridor") != "Non-corridor",
            at_junction=bool(getattr(event, "at_junction", False)),
            priority=getattr(event, "priority", "High") or "High",
            direction=getattr(event, "direction", "north_or_other") or "north_or_other",
            cargo_macro=getattr(event, "cargo_macro", "empty_or_none") or "empty_or_none",
            breakdown_severity=getattr(event, "breakdown_severity", "not_applicable") or "not_applicable",
            endlatitude=getattr(event, "endlatitude", None),
            endlongitude=getattr(event, "endlongitude", None),
            veh_type=getattr(event, "veh_type", None),
        )
        X = build_feature_row(raw)

        raw_duration_seconds = float(self.duration_model.predict(X)[0])
        raw_duration_seconds = max(raw_duration_seconds, 0.0)  # guard against tiny negative regressor noise
        duration_minutes = round(raw_duration_seconds / 60.0, 1)

        closure_proba = float(self.closure_model.predict_proba(X)[0, 1])
        predicted_closure = closure_proba >= 0.5

        band = self._duration_band(duration_minutes)
        # impact_score kept on the same 0-10 scale as the rest of the system
        # (rule engine / resource lookup table) for continuity, derived from duration.
        impact_score = min(round((duration_minutes / 120.0) * 10, 2), 10.0)
        if predicted_closure:
            impact_score = min(round(impact_score * 1.15, 2), 10.0)

        return {
            "duration_minutes": duration_minutes,
            "raw_duration_seconds": round(raw_duration_seconds, 1),
            "closure_proba": closure_proba,
            "predicted_closure": predicted_closure,
            "severity_band": band,
            "impact_score": impact_score,
        }

    @staticmethod
    def _duration_band(minutes: float) -> str:
        if minutes < DURATION_BAND_THRESHOLDS["LOW"]:
            return "LOW"
        if minutes < DURATION_BAND_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        if minutes < DURATION_BAND_THRESHOLDS["HIGH"]:
            return "HIGH"
        return "CRITICAL"


_engine_singleton = None


def get_ml_engine() -> NotebookMLEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = NotebookMLEngine()
    return _engine_singleton
