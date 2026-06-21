"""
Week 3 - ML Model Training (XGBoost)
=======================================
Trains two gradient-boosted classifiers on the processed event dataset:

1. priority_model      -> predicts target_priority (High=1 / Low=0)
                           This is the primary "congestion severity" signal.
2. road_closure_model  -> predicts target_road_closure (1/0)
                           Feeds directly into the diversion-planning module.

Both models only use features that are knowable AT INTAKE TIME (cause, location,
time, vehicle type) — never the target columns themselves or post-hoc fields like
duration/status that wouldn't exist yet for a brand-new incoming report.

Outputs:
  models/priority_model.json
  models/road_closure_model.json
  models/encoders.joblib       (category -> int mappings, needed by the API)
  outputs/model_metrics.json   (eval numbers for both models + comparison vs rule baseline)

Run:
    python scripts/05_train_ml_model.py
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              classification_report, precision_score, recall_score)
import xgboost as xgb

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "processed_events.csv")
MODEL_DIR = os.path.join(HERE, "..", "models")
OUT_DIR = os.path.join(HERE, "..", "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORICAL_FEATURES = ["event_type", "event_cause", "zone", "corridor", "veh_type", "time_window"]
NUMERIC_FEATURES = ["hour_of_day", "day_of_week", "is_weekend", "month", "is_peak_hour",
                     "is_corridor_road", "corridor_event_share", "cause_historical_high_rate"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_encoders(df: pd.DataFrame) -> dict:
    """category -> {value: int} mapping, with an 'UNK' bucket for unseen values at inference time."""
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        cats = sorted(df[col].astype(str).unique().tolist())
        mapping = {v: i + 1 for i, v in enumerate(cats)}  # 0 reserved for unknown
        mapping["__UNK__"] = 0
        encoders[col] = mapping
    return encoders


def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        mapping = encoders[col]
        df[col] = df[col].astype(str).map(lambda v: mapping.get(v, 0))
    return df


def train_one_target(df_enc: pd.DataFrame, target_col: str, model_name: str):
    X = df_enc[ALL_FEATURES]
    y = df_enc[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Note: cause_historical_high_rate is computed from priority on the FULL dataset in
    # preprocessing — for the priority model this leaks a small amount of target signal
    # for the cause column specifically. We keep it for the road_closure model (different
    # target) but exclude it for the priority model to keep that evaluation honest.
    features = ALL_FEATURES.copy()
    if target_col == "target_priority":
        features = [f for f in features if f != "cause_historical_high_rate"]
        X_train, X_test = X_train[features], X_test[features]

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features_used": features,
    }

    print(f"\n=== {model_name} ===")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, pred))

    # Feature importance
    importances = dict(zip(features, model.feature_importances_.round(4).tolist()))
    importances = dict(sorted(importances.items(), key=lambda kv: -kv[1]))
    metrics["feature_importance"] = importances

    model.save_model(os.path.join(MODEL_DIR, f"{model_name}.json"))
    return metrics, features


def main():
    df = pd.read_csv(DATA_PATH)

    encoders = build_encoders(df)
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.joblib"))

    # Lookup tables needed to reconstruct corridor_event_share / cause_historical_high_rate
    # for a brand-new incoming event at inference time (same logic as scripts/01_preprocessing.py)
    corridor_share_lookup = df["corridor"].value_counts(normalize=True).to_dict()
    cause_high_rate_lookup = df.groupby("event_cause")["priority"].apply(lambda s: (s == "High").mean()).to_dict()
    joblib.dump(
        {"corridor_event_share": corridor_share_lookup, "cause_historical_high_rate": cause_high_rate_lookup,
         "corridor_event_share_default": 0.0, "cause_historical_high_rate_default": df["target_priority"].mean()},
        os.path.join(MODEL_DIR, "lookup_tables.joblib"))

    df_enc = apply_encoders(df, encoders)

    priority_metrics, priority_features = train_one_target(df_enc, "target_priority", "priority_model")

    # --- IMPORTANT DATA FINDING ---
    # is_corridor_road / corridor / corridor_event_share turn out to almost perfectly
    # determine `priority` in the source ASTRAM data (Non-corridor -> ~100% Low,
    # On-corridor -> ~99.6% High). That's not a feature-engineering bug — corridor is
    # genuinely known at intake time — but it means the historical "priority" label is,
    # in practice, mostly a re-statement of an existing administrative rule rather than
    # an independently assessed severity judgement. We report a second, ablated model
    # that excludes corridor-identity features, to show what the model can predict from
    # cause/time/zone/vehicle-type alone — the genuinely "learned" signal a forecasting
    # tool would need for events where corridor tagging is ambiguous or not yet assigned.
    ablated_features = [f for f in ALL_FEATURES
                         if f not in ("is_corridor_road", "corridor", "corridor_event_share",
                                      "cause_historical_high_rate")]
    X = df_enc[ablated_features]
    y = df_enc["target_priority"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    ablated_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss", random_state=42)
    ablated_model.fit(X_train, y_train)
    pred = ablated_model.predict(X_test)
    proba = ablated_model.predict_proba(X_test)[:, 1]
    ablated_metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "features_used": ablated_features,
        "note": "Excludes corridor-identity features to isolate genuine learned signal "
                "from cause/time/zone/vehicle-type, since corridor alone near-perfectly "
                "determines priority in this dataset (see comment in script).",
    }
    print("\n=== priority_model (ablated - no corridor identity features) ===")
    print(json.dumps(ablated_metrics, indent=2))
    ablated_model.save_model(os.path.join(MODEL_DIR, "priority_model_ablated.json"))

    closure_metrics, closure_features = train_one_target(df_enc, "target_road_closure", "road_closure_model")

    # Save which feature list each model expects (priority model drops one leaky feature)
    joblib.dump({"priority_model": priority_features, "road_closure_model": closure_features},
                os.path.join(MODEL_DIR, "model_feature_lists.joblib"))

    all_metrics = {
        "priority_model": priority_metrics,
        "priority_model_ablated_no_corridor": ablated_metrics,
        "road_closure_model": closure_metrics,
        "comparison_note": (
            "Rule-based baseline (Week 2) accuracy on full historical data: ~0.66 at best "
            "threshold, AUC ~0.667. Compare against priority_model roc_auc above — the "
            "trained model should meaningfully beat that ceiling."
        ),
        "data_finding": (
            "priority_model's near-1.0 AUC is driven almost entirely by corridor identity "
            "(Non-corridor events are ~100% Low priority, on-corridor events are ~99.6% High "
            "in the historical data) — this looks like an existing administrative labeling "
            "rule in ASTRAM, not something the model 'discovered'. The ablated model (no "
            "corridor-identity features) shows the genuine signal available from cause/time/"
            "zone/vehicle-type alone, which is the realistic operating point for new/ambiguous "
            "event reports."
        ),
    }
    with open(os.path.join(OUT_DIR, "model_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved models -> {MODEL_DIR}")
    print(f"Saved metrics -> {os.path.join(OUT_DIR, 'model_metrics.json')}")


if __name__ == "__main__":
    main()
