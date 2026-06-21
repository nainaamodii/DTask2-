"""
Week 6 - Monthly Retraining Job (Post-Event Learning Loop)
===============================================================
Closes the loop described in the original plan:

    Event occurs -> officers log actual congestion (POST /feedback)
                 -> system compares predicted vs actual (GET /feedback/stats)
                 -> delta stored as training signal (data/feedback.db)
                 -> monthly retraining batch updates model weights  <- this script
                 -> recommendation thresholds auto-calibrate

What this script does:
  1. Reads every row from data/feedback.db (logged via the API's /feedback
     endpoint after real events were resolved).
  2. Converts each feedback row into a training row in the SAME feature
     schema as data/processed_events.csv (reusing scripts/01_preprocessing's
     feature logic for consistency).
  3. Appends these to the historical dataset and retrains both models exactly
     as scripts/05_train_ml_model.py does, but reports a *before vs after*
     metric comparison so you can see whether the feedback genuinely helped.
  4. Versions the resulting models (models/v2/, models/v3/, ...) rather than
     overwriting v1, so a bad retraining batch can be rolled back.

In production this would run on a schedule (e.g. a monthly cron / Airflow
job); here it's invoked manually for demo purposes.

Run:
    python scripts/07_retrain_with_feedback.py
"""

import os
import sqlite3
import json
import shutil
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "processed_events.csv")
FEEDBACK_DB = os.path.join(HERE, "..", "data", "feedback.db")
MODEL_DIR = os.path.join(HERE, "..", "models")
OUT_DIR = os.path.join(HERE, "..", "outputs")

CATEGORICAL_FEATURES = ["event_type", "event_cause", "zone", "corridor", "veh_type", "time_window"]
NUMERIC_FEATURES = ["hour_of_day", "day_of_week", "is_weekend", "month", "is_peak_hour",
                     "is_corridor_road", "corridor_event_share", "cause_historical_high_rate"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_feedback_rows() -> pd.DataFrame:
    if not os.path.exists(FEEDBACK_DB):
        print("No feedback.db found yet — nothing to retrain on. "
              "(Log some /feedback entries via the API first.)")
        return pd.DataFrame()
    conn = sqlite3.connect(FEEDBACK_DB)
    df = pd.read_sql_query("SELECT * FROM feedback", conn)
    conn.close()
    return df


def feedback_to_training_rows(fb: pd.DataFrame) -> pd.DataFrame:
    """
    NOTE on a real limitation: the /feedback payload only captures the outcome
    (actual_priority, actual_road_closure) and links back to an event_id — it
    does NOT carry the original intake features (cause, corridor, time, etc.)
    because those live in whatever system created the event, not in this
    feedback record. A production version would join on event_id against an
    `events` table populated at /forecast time. For this prototype, we
    simulate that join: if a `notes` field on the feedback contains the
    original intake JSON (the demo client is expected to pass it through),
    we parse it; otherwise the row is skipped with a warning, since we can't
    safely fabricate features for a real retraining set.
    """
    rows = []
    skipped = 0
    for _, r in fb.iterrows():
        intake = None
        if r.get("notes"):
            try:
                parsed = json.loads(r["notes"])
                if isinstance(parsed, dict) and "event_cause" in parsed:
                    intake = parsed
            except (json.JSONDecodeError, TypeError):
                pass
        if intake is None:
            skipped += 1
            continue
        rows.append({**intake,
                     "priority": r["actual_priority"],
                     "requires_road_closure": bool(r["actual_road_closure"])})
    print(f"Parsed {len(rows)} usable feedback rows with intake features attached "
          f"({skipped} skipped — no intake JSON in `notes`).")
    return pd.DataFrame(rows)


def engineer_minimal(df: pd.DataFrame, lookups: dict) -> pd.DataFrame:
    """Lightweight re-implementation of the feature engineering needed here
    (mirrors scripts/01_preprocessing.py / backend/app/ml_engine.py)."""
    df = df.copy()
    df["is_corridor_road"] = (df["corridor"].fillna("Non-corridor") != "Non-corridor").astype(int)
    df["corridor_event_share"] = df["corridor"].map(lookups["corridor_event_share"]).fillna(
        lookups["corridor_event_share_default"])
    df["cause_historical_high_rate"] = df["event_cause"].map(lookups["cause_historical_high_rate"]).fillna(
        lookups["cause_historical_high_rate_default"])
    df["target_priority"] = (df["priority"] == "High").astype(int)
    df["target_road_closure"] = df["requires_road_closure"].astype(int)
    for col in ["zone", "veh_type", "time_window", "event_type"]:
        if col not in df.columns:
            df[col] = "Unknown"
    for col in ["hour_of_day", "day_of_week", "is_weekend", "month", "is_peak_hour"]:
        if col not in df.columns:
            df[col] = 0
    return df


def train_eval(df_enc: pd.DataFrame, target_col: str, feature_list: list):
    X, y = df_enc[feature_list], df_enc[target_col]
    if y.nunique() < 2:
        return None, None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.value_counts().min() > 1 else None)
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, scale_pos_weight=pos_weight,
                               eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "n_train": len(X_train), "n_test": len(X_test),
    }
    try:
        metrics["roc_auc"] = round(roc_auc_score(y_test, proba), 4)
    except ValueError:
        metrics["roc_auc"] = None
    return model, metrics


def next_version_dir() -> str:
    existing = [d for d in os.listdir(MODEL_DIR) if d.startswith("v") and d[1:].isdigit()] \
        if os.path.exists(MODEL_DIR) else []
    n = max([int(d[1:]) for d in existing], default=1) + 1
    return os.path.join(MODEL_DIR, f"v{n}")


def main():
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))
    lookups = joblib.load(os.path.join(MODEL_DIR, "lookup_tables.joblib"))

    base_df = pd.read_csv(DATA_PATH)
    fb_raw = load_feedback_rows()
    fb_df = feedback_to_training_rows(fb_raw) if len(fb_raw) else pd.DataFrame()

    if len(fb_df) == 0:
        print("\nNo usable feedback rows yet. Run the demo seeding step (see README "
              "'Week 6' section) to simulate some, or wait for real /feedback traffic.")
        print("Retraining on historical data only, for a clean before/after comparison baseline.")

    fb_df_engineered = engineer_minimal(fb_df, lookups) if len(fb_df) else pd.DataFrame()
    combined = pd.concat([base_df, fb_df_engineered], ignore_index=True, sort=False) if len(fb_df) else base_df

    def encode(df):
        df = df.copy()
        for col in CATEGORICAL_FEATURES:
            mapping = encoders[col]
            df[col] = df[col].astype(str).map(lambda v: mapping.get(v, 0))
        return df

    df_enc = encode(combined)

    priority_features = [f for f in ALL_FEATURES if f != "cause_historical_high_rate"]
    closure_features = ALL_FEATURES.copy()

    print(f"\nTraining set size: {len(combined)} rows ({len(base_df)} historical + {len(fb_df)} from feedback)")

    new_priority_model, priority_metrics = train_eval(df_enc, "target_priority", priority_features)
    new_closure_model, closure_metrics = train_eval(df_enc, "target_road_closure", closure_features)

    # Compare against the currently-deployed (v1) metrics for a before/after view
    old_metrics_path = os.path.join(OUT_DIR, "model_metrics.json")
    old_metrics = json.load(open(old_metrics_path)) if os.path.exists(old_metrics_path) else {}

    print("\n=== Retrained priority_model ===")
    print(json.dumps(priority_metrics, indent=2))
    print("vs previous deployed version:",
          json.dumps(old_metrics.get("priority_model", {}), indent=2))

    print("\n=== Retrained road_closure_model ===")
    print(json.dumps(closure_metrics, indent=2))
    print("vs previous deployed version:",
          json.dumps(old_metrics.get("road_closure_model", {}), indent=2))

    version_dir = next_version_dir()
    os.makedirs(version_dir, exist_ok=True)
    new_priority_model.save_model(os.path.join(version_dir, "priority_model.json"))
    new_closure_model.save_model(os.path.join(version_dir, "road_closure_model.json"))

    report = {
        "version_dir": version_dir,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_historical_rows": len(base_df),
        "n_feedback_rows_used": len(fb_df),
        "priority_model": priority_metrics,
        "road_closure_model": closure_metrics,
        "previous_priority_model": old_metrics.get("priority_model", {}),
        "previous_road_closure_model": old_metrics.get("road_closure_model", {}),
    }
    with open(os.path.join(version_dir, "retrain_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nNew model version saved to {version_dir}")
    print("This prototype does NOT auto-promote the new version to production "
          "(models/priority_model.json, models/road_closure_model.json) — that's "
          "an intentional safety gate. To promote: copy the files from "
          f"{version_dir} over the ones in {MODEL_DIR} after reviewing retrain_report.json.")


if __name__ == "__main__":
    main()
