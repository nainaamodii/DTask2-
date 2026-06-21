"""
ML inference engine — loads the trained XGBoost models + encoders/lookup
tables from Week 3 (scripts/05_train_ml_model.py) and turns a single new
EventIntakeRequest into priority / road-closure predictions.

Feature engineering here intentionally mirrors scripts/01_preprocessing.py so
training and inference stay consistent.
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timezone

HERE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(HERE, "..", "..", "models")

MORNING_PEAK = (8, 11)
EVENING_PEAK = (17, 21)


class MLEngine:
    def __init__(self):
        self.encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))
        self.lookups = joblib.load(os.path.join(MODEL_DIR, "lookup_tables.joblib"))
        self.feature_lists = joblib.load(os.path.join(MODEL_DIR, "model_feature_lists.joblib"))

        self.priority_model = xgb.XGBClassifier()
        self.priority_model.load_model(os.path.join(MODEL_DIR, "priority_model.json"))

        self.closure_model = xgb.XGBClassifier()
        self.closure_model.load_model(os.path.join(MODEL_DIR, "road_closure_model.json"))

    @staticmethod
    def _time_window(hour: int) -> str:
        if MORNING_PEAK[0] <= hour <= MORNING_PEAK[1]:
            return "morning_peak"
        if EVENING_PEAK[0] <= hour <= EVENING_PEAK[1]:
            return "evening_peak"
        if hour >= 22 or hour <= 5:
            return "night"
        return "off_peak"

    def _build_feature_row(self, req) -> dict:
        dt = req.event_datetime or datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local_dt = dt.astimezone(timezone.utc) + pd.Timedelta(hours=5, minutes=30)

        hour_of_day = local_dt.hour
        day_of_week = local_dt.weekday()
        is_weekend = int(day_of_week in (5, 6))
        month = local_dt.month
        time_window = self._time_window(hour_of_day)
        is_peak_hour = int(time_window in ("morning_peak", "evening_peak"))

        corridor = req.corridor or "Non-corridor"
        is_corridor_road = int(corridor != "Non-corridor")

        corridor_event_share = self.lookups["corridor_event_share"].get(
            corridor, self.lookups["corridor_event_share_default"])
        cause_historical_high_rate = self.lookups["cause_historical_high_rate"].get(
            req.event_cause, self.lookups["cause_historical_high_rate_default"])

        row = {
            "event_type": req.event_type,
            "event_cause": req.event_cause,
            "zone": req.zone or "Unknown",
            "corridor": corridor,
            "veh_type": req.veh_type or "not_applicable",
            "time_window": time_window,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "month": month,
            "is_peak_hour": is_peak_hour,
            "is_corridor_road": is_corridor_road,
            "corridor_event_share": corridor_event_share,
            "cause_historical_high_rate": cause_historical_high_rate,
        }
        return row

    def _encode(self, row: dict, feature_list: list) -> pd.DataFrame:
        encoded = {}
        for col in feature_list:
            if col in self.encoders:  # categorical
                mapping = self.encoders[col]
                encoded[col] = mapping.get(str(row[col]), 0)
            else:
                encoded[col] = row[col]
        return pd.DataFrame([encoded])[feature_list]

    def predict(self, req) -> dict:
        row = self._build_feature_row(req)

        priority_features = self.feature_lists["priority_model"]
        closure_features = self.feature_lists["road_closure_model"]

        X_priority = self._encode(row, priority_features)
        X_closure = self._encode(row, closure_features)

        priority_proba = float(self.priority_model.predict_proba(X_priority)[0, 1])
        closure_proba = float(self.closure_model.predict_proba(X_closure)[0, 1])

        predicted_priority = "High" if priority_proba >= 0.5 else "Low"
        predicted_closure = closure_proba >= 0.5

        # Translate model confidence into a 0-10 impact score consistent with the
        # rule-based engine's scale, so downstream resource lookup is shared.
        impact_score = round(priority_proba * 10, 2)
        if predicted_closure:
            impact_score = min(round(impact_score * 1.15, 2), 10.0)

        return {
            "row": row,
            "priority_proba": priority_proba,
            "closure_proba": closure_proba,
            "predicted_priority": predicted_priority,
            "predicted_closure": predicted_closure,
            "impact_score": impact_score,
        }


_engine_singleton = None


def get_ml_engine() -> MLEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = MLEngine()
    return _engine_singleton
