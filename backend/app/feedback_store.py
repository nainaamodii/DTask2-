"""
Week 6 - Post-Event Learning Loop: Feedback Persistence
============================================================
SQLite-backed feedback table so feedback survives server restarts and can be
consumed by the monthly retraining job (scripts/08_retrain_notebook_models.py).

Table: feedback
  id, event_id, actual_road_closure, actual_duration_minutes, actual_priority,
  notes, logged_at, predicted_duration_minutes, predicted_road_closure,
  predicted_road_closure_confidence, predicted_priority, predicted_priority_confidence, method

actual_priority / predicted_priority are kept as legacy/optional columns —
duration + closure are the primary targets for the current models.

The predicted_* columns are optional and only populated when the caller
includes them — letting the API log predicted-vs-actual deltas in one place
even though /forecast and /feedback are separate calls in this prototype (a
production system would link them via event_id end-to-end).
"""

import sqlite3
import os
from datetime import datetime, timezone

HERE = os.path.dirname(__file__)
DB_PATH = os.path.join(HERE, "..", "..", "data", "feedback.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    actual_road_closure INTEGER NOT NULL,
    actual_duration_minutes REAL,
    actual_priority TEXT,
    notes TEXT,
    predicted_duration_minutes REAL,
    predicted_road_closure INTEGER,
    predicted_road_closure_confidence REAL,
    predicted_priority TEXT,
    predicted_priority_confidence REAL,
    method TEXT,
    logged_at TEXT NOT NULL
);
"""


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn


def insert_feedback(entry: dict) -> int:
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO feedback
           (event_id, actual_road_closure, actual_duration_minutes, actual_priority, notes,
            predicted_duration_minutes, predicted_road_closure, predicted_road_closure_confidence,
            predicted_priority, predicted_priority_confidence, method, logged_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            entry["event_id"],
            int(entry["actual_road_closure"]),
            entry.get("actual_duration_minutes"),
            entry.get("actual_priority"),
            entry.get("notes"),
            entry.get("predicted_duration_minutes"),
            int(entry["predicted_road_closure"]) if entry.get("predicted_road_closure") is not None else None,
            entry.get("predicted_road_closure_confidence"),
            entry.get("predicted_priority"),
            entry.get("predicted_priority_confidence"),
            entry.get("method"),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def list_feedback() -> list:
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM feedback ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_feedback() -> int:
    conn = _get_conn()
    n = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    conn.close()
    return n


def summary_stats() -> dict:
    """How predictions are tracking against actuals, for entries where a
    prediction was logged alongside the actual outcome."""
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM feedback WHERE predicted_duration_minutes IS NOT NULL "
        "OR predicted_road_closure IS NOT NULL").fetchall()
    conn.close()
    rows = [dict(r) for r in rows]
    n = len(rows)
    if n == 0:
        return {"n_total_feedback": count_feedback(), "n_with_prediction_logged": 0,
                "duration_mae_minutes": None, "closure_accuracy": None}

    dur_rows = [r for r in rows if r["predicted_duration_minutes"] is not None and r["actual_duration_minutes"] is not None]
    duration_mae = (sum(abs(r["predicted_duration_minutes"] - r["actual_duration_minutes"]) for r in dur_rows) / len(dur_rows)
                     if dur_rows else None)

    closure_rows = [r for r in rows if r["predicted_road_closure"] is not None]
    closure_correct = sum(1 for r in closure_rows
                           if int(r["predicted_road_closure"]) == int(r["actual_road_closure"]))

    return {
        "n_total_feedback": count_feedback(),
        "n_with_prediction_logged": n,
        "duration_mae_minutes": round(duration_mae, 1) if duration_mae is not None else None,
        "n_duration_comparisons": len(dur_rows),
        "closure_accuracy": round(closure_correct / len(closure_rows), 3) if closure_rows else None,
        "n_closure_comparisons": len(closure_rows),
    }
