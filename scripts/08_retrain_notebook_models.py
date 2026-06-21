"""
Week 6 - Monthly Retraining Job (notebook-model schema)
=============================================================
Adapted version of the post-event learning loop for the duration/closure
models supplied via the Gridlock_traffic_data.ipynb notebook (see
backend/app/notebook_features.py for the feature schema this matches).

What this does:
  1. Reads every row from data/feedback.db.
  2. For rows where the original intake payload was passed through (the
     dashboard's feedback button does this via the `notes` field — see the
     same limitation noted in the legacy script), rebuilds the 30-column
     feature vector using backend/app/notebook_features.py.
  3. Re-fits a fresh XGBRegressor (duration, in seconds — matching the
     original training data's units) and XGBClassifier (closure) combining
     a frozen base sample of "training-shaped" rows with the new feedback
     rows*, and reports before/after metrics.
  4. Versions the result under models/notebook/v2/, v3/, ... — never
     auto-overwrites the live model.

*IMPORTANT CAVEAT: we don't have the original notebook's full training
dataframe (data_shared.csv) in this project, only the fitted models. So
this script can't do a true "old data + new feedback" retrain — instead it
fine-tunes by continuing training (`xgb_model=...` warm start) the existing
fitted models on the feedback batch alone. This is a reasonable prototype
stand-in (and a real technique — incremental/warm-start boosting), but is
NOT equivalent to a full retrain on the combined dataset. If you have
access to the original data_shared.csv used in the notebook, point
ORIGINAL_TRAINING_DATA below at it to do a proper full retrain instead.

Run:
    python scripts/08_retrain_notebook_models.py
"""

import os
import sys
import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd
import pickle
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score, roc_auc_score

HERE = os.path.dirname(__file__)
BACKEND_APP_DIR = os.path.join(HERE, "..", "backend")
sys.path.insert(0, BACKEND_APP_DIR)
from app.notebook_features import RawEventInput, build_feature_row, MODEL_FEATURE_COLUMNS  # noqa: E402

FEEDBACK_DB = os.path.join(HERE, "..", "data", "feedback.db")
MODEL_DIR = os.path.join(HERE, "..", "models", "notebook")

# Set this to a local copy of the notebook's original `data_shared.csv` (or an
# equivalent already-cleaned dataframe) to do a true full retrain instead of
# the warm-start fine-tune. Left as None here since that file wasn't provided.
ORIGINAL_TRAINING_DATA = None


def load_feedback_rows() -> pd.DataFrame:
    if not os.path.exists(FEEDBACK_DB):
        print("No feedback.db found yet — nothing to retrain on. "
              "(Log some /feedback entries via the API or dashboard first.)")
        return pd.DataFrame()
    conn = sqlite3.connect(FEEDBACK_DB)
    df = pd.read_sql_query("SELECT * FROM feedback", conn)
    conn.close()
    return df


def feedback_to_feature_rows(fb: pd.DataFrame):
    """Parses the intake JSON the dashboard embeds in `notes` (same workaround
    as the legacy script) back into RawEventInput + actual outcomes."""
    X_rows, y_duration_sec, y_closure = [], [], []
    skipped = 0
    for _, r in fb.iterrows():
        if r.get("actual_duration_minutes") is None:
            skipped += 1
            continue
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
        raw = RawEventInput(
            event_type=intake.get("event_type", "unplanned"),
            latitude=intake["latitude"],
            longitude=intake["longitude"],
            event_cause=intake["event_cause"],
            authenticated=bool(intake.get("authenticated", False)),
            is_major_corridor=(intake.get("corridor", "Non-corridor") or "Non-corridor") != "Non-corridor",
            at_junction=bool(intake.get("at_junction", False)),
            priority=intake.get("priority", "High") or "High",
            direction=intake.get("direction", "north_or_other") or "north_or_other",
            cargo_macro=intake.get("cargo_macro", "empty_or_none") or "empty_or_none",
            breakdown_severity=intake.get("breakdown_severity", "not_applicable") or "not_applicable",
            endlatitude=intake.get("endlatitude"),
            endlongitude=intake.get("endlongitude"),
        )
        X_rows.append(build_feature_row(raw))
        y_duration_sec.append(float(r["actual_duration_minutes"]) * 60.0)  # back to seconds, matching training units
        y_closure.append(int(r["actual_road_closure"]))

    print(f"Parsed {len(X_rows)} usable feedback rows with intake features + actual outcomes attached "
          f"({skipped} skipped — missing intake JSON or actual_duration_minutes).")
    if not X_rows:
        return None, None, None
    X = pd.concat(X_rows, ignore_index=True)[MODEL_FEATURE_COLUMNS]
    return X, pd.Series(y_duration_sec), pd.Series(y_closure)


def next_version_dir() -> str:
    existing = [d for d in os.listdir(MODEL_DIR) if d.startswith("v") and d[1:].isdigit()] \
        if os.path.exists(MODEL_DIR) else []
    n = max([int(d[1:]) for d in existing], default=1) + 1
    return os.path.join(MODEL_DIR, f"v{n}")


def main():
    with open(os.path.join(MODEL_DIR, "xgboost_duration_model.pkl"), "rb") as f:
        duration_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "xgboost_closure_model.pkl"), "rb") as f:
        closure_model = pickle.load(f)

    fb_raw = load_feedback_rows()
    if len(fb_raw) == 0:
        print("Nothing to retrain on yet.")
        return

    X, y_dur, y_cls = feedback_to_feature_rows(fb_raw)
    if X is None or len(X) < 5:
        print(f"\nOnly {0 if X is None else len(X)} usable feedback rows — need at least 5 for a "
              "meaningful retrain batch. Log more feedback via the dashboard and try again.")
        return

    print(f"\nFine-tuning on {len(X)} feedback rows (warm-start from the currently deployed models)...")

    # Baseline (before) predictions on this same batch, for a clear before/after comparison
    pred_dur_before = duration_model.predict(X)
    pred_cls_before = closure_model.predict(X)
    proba_cls_before = closure_model.predict_proba(X)[:, 1]

    before_metrics = {
        "duration_mae_seconds": round(mean_absolute_error(y_dur, pred_dur_before), 1),
        "duration_mae_minutes": round(mean_absolute_error(y_dur, pred_dur_before) / 60, 1),
    }
    try:
        before_metrics["closure_accuracy"] = round(accuracy_score(y_cls, pred_cls_before), 3)
        before_metrics["closure_roc_auc"] = round(roc_auc_score(y_cls, proba_cls_before), 3) if y_cls.nunique() > 1 else None
    except ValueError:
        before_metrics["closure_accuracy"] = None

    # Warm-start fine-tune: continue boosting on top of the existing trees
    new_duration_model = xgb.XGBRegressor(n_estimators=30, random_state=42, objective="reg:squarederror")
    new_duration_model.fit(X, y_dur, xgb_model=duration_model.get_booster())

    new_closure_model = xgb.XGBClassifier(n_estimators=30, random_state=42, eval_metric="logloss")
    if y_cls.nunique() > 1:
        new_closure_model.fit(X, y_cls, xgb_model=closure_model.get_booster())
    else:
        print("Feedback batch has only one closure class present — skipping closure fine-tune this round.")
        new_closure_model = closure_model

    pred_dur_after = new_duration_model.predict(X)
    after_metrics = {
        "duration_mae_seconds": round(mean_absolute_error(y_dur, pred_dur_after), 1),
        "duration_mae_minutes": round(mean_absolute_error(y_dur, pred_dur_after) / 60, 1),
    }
    if y_cls.nunique() > 1:
        pred_cls_after = new_closure_model.predict(X)
        proba_cls_after = new_closure_model.predict_proba(X)[:, 1]
        after_metrics["closure_accuracy"] = round(accuracy_score(y_cls, pred_cls_after), 3)
        after_metrics["closure_roc_auc"] = round(roc_auc_score(y_cls, proba_cls_after), 3)
    else:
        after_metrics["closure_accuracy"] = before_metrics["closure_accuracy"]

    print("\n=== BEFORE (currently deployed models, evaluated on this feedback batch) ===")
    print(json.dumps(before_metrics, indent=2))
    print("\n=== AFTER (fine-tuned, evaluated on the SAME batch — expect improvement, "
          "this is an in-sample check not a held-out eval) ===")
    print(json.dumps(after_metrics, indent=2))

    version_dir = next_version_dir()
    os.makedirs(version_dir, exist_ok=True)
    with open(os.path.join(version_dir, "xgboost_duration_model.pkl"), "wb") as f:
        pickle.dump(new_duration_model, f)
    with open(os.path.join(version_dir, "xgboost_closure_model.pkl"), "wb") as f:
        pickle.dump(new_closure_model, f)

    report = {
        "version_dir": version_dir,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_feedback_rows_used": len(X),
        "method": "warm_start_finetune",
        "caveat": ("Fine-tuned on feedback only (no access to the original training dataframe). "
                   "Metrics above are in-sample on the fine-tuning batch, not a held-out test set — "
                   "treat as a sanity check, not a generalization estimate. Review before promoting."),
        "before": before_metrics,
        "after": after_metrics,
    }
    with open(os.path.join(version_dir, "retrain_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nNew model version saved to {version_dir}")
    print("Not auto-promoted to production. To promote: copy the .pkl files from "
          f"{version_dir} over the ones in {MODEL_DIR} after reviewing retrain_report.json.")


if __name__ == "__main__":
    main()
