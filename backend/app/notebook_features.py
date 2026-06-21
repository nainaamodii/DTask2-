"""
Feature engineering for the notebook-trained models (Gridlock_traffic_data.ipynb)
=====================================================================================
This is the single source of truth for turning a friendly event-intake payload
into the exact 30-column feature vector that xgboost_duration_model.pkl and
xgboost_closure_model.pkl expect (their fitted `.feature_names_in_`).

Reverse-engineered + verified against the supplied notebook and the actual
fitted models (see notes below for the discrepancies found between them).

--- Important findings, documented here rather than hidden in the encoding ---

1. UNIT BUG IN THE DURATION TARGET: the notebook computes
   `congestion_time_minutes = (resolved_datetime - start_datetime).dt.total_seconds()`
   — that's `.total_seconds()`, never divided by 60. So despite the column's name,
   the duration model was trained on SECONDS. Verified empirically: raw model
   output for representative inputs comes out as 18,000-36,000 (i.e. 5-10 hours),
   which only makes sense as seconds. `predict_duration_minutes()` below divides
   the raw model output by 60 to correct this.

2. `veh_type` and `priority_High` are NOT in the final fitted models' feature
   list, even though the notebook's earlier cells construct them. The trained
   `.feature_names_in_` (30 cols) doesn't include any `veh_type__*` or
   `priority_High` column — they must have been dropped in a step that isn't
   in the notebook as supplied. Practical effect: vehicle type cannot influence
   either model's prediction (kept here for display/logging only), and "High"
   priority is represented implicitly as `priority_Low=0, priority_Unknown=0`.

3. `direction` dummies in the fitted model only cover 6 of the real dataset's
   8 directions (north_east, north_west, south, south_west, unknown, west —
   no `north` or `south_east` columns exist). Selecting "north" or "south_east"
   is therefore indistinguishable from the dropped baseline to this model. The
   UI/API exposes this as an explicit "North / Other (baseline)" option rather
   than silently picking one.

4. `cargo_macro` baseline (all dummies = 0) is `construction_heavy` — a `drop_first`
   artifact of alphabetical ordering, not a deliberate "default" choice. We default
   the UI to `empty_or_none` instead (which has its own explicit dummy column) since
   that's the closest semantic match to "no cargo / not applicable".
"""

from dataclasses import dataclass, field
import pandas as pd

# Macro cause mapping — copied verbatim from the notebook's cause_mapping dict.
# Note 'others' appears twice in the original dict literal; the second
# assignment ('others' -> 'congestion') wins in Python, so that's what's used here.
CAUSE_MACRO_MAPPING = {
    "public_event": "Planned & Public Events",
    "vip_movement": "Planned & Public Events",
    "procession": "Planned & Public Events",
    "protest": "Planned & Public Events",
    "vehicle_breakdown": "Accidents & Hazards",
    "accident": "Accidents & Hazards",
    "tree_fall": "Accidents & Hazards",
    "debris": "Accidents & Hazards",
    "construction": "Infrastructure Issues",
    "pot_holes": "Infrastructure Issues",
    "road_conditions": "Infrastructure Issues",
    "water_logging": "Weather Events",
    "fog_low_visibility": "Weather Events",
    "others": "congestion",
}
CAUSE_MACRO_DEFAULT = "Other / Unspecified"  # fillna() result for unmapped causes

# event_cause_macro dummy columns present in the fitted model (baseline = "Accidents & Hazards")
EVENT_CAUSE_MACRO_DUMMY_VALUES = [
    "Infrastructure Issues", "Other / Unspecified", "Planned & Public Events",
    "Weather Events", "congestion",
]

# direction dummy columns present in the fitted model (baseline = north / other, see finding #3)
DIRECTION_DUMMY_VALUES = ["north_east", "north_west", "south", "south_west", "unknown", "west"]
DIRECTION_BASELINE_LABEL = "north_or_other"

# cargo_macro dummy columns present in the fitted model (baseline = construction_heavy, see finding #4)
CARGO_MACRO_DUMMY_VALUES = ["empty_or_none", "general_goods", "liquid_gas", "municipal_waste", "perishables"]
CARGO_MACRO_BASELINE_LABEL = "construction_heavy"

# breakdown_severity dummy columns present in the fitted model (baseline = general_breakdown)
BREAKDOWN_SEVERITY_DUMMY_VALUES = ["major_failure", "minor_issue", "not_applicable"]
BREAKDOWN_SEVERITY_BASELINE_LABEL = "general_breakdown"

# priority dummy columns present in the fitted model (baseline = High, see finding #2)
PRIORITY_DUMMY_VALUES = ["Low", "Unknown"]

# The exact 30 columns + order both fitted models expect.
MODEL_FEATURE_COLUMNS = [
    "event_type", "latitude", "longitude", "endlatitude", "endlongitude",
    "authenticated", "is_stretch_event", "is_major_corridor", "at_junction",
    "priority_Low", "priority_Unknown",
    "cargo_macro_empty_or_none", "cargo_macro_general_goods", "cargo_macro_liquid_gas",
    "cargo_macro_municipal_waste", "cargo_macro_perishables",
    "breakdown_severity_major_failure", "breakdown_severity_minor_issue", "breakdown_severity_not_applicable",
    "direction_north_east", "direction_north_west", "direction_south", "direction_south_west",
    "direction_unknown", "direction_west",
    "event_cause_macro_Infrastructure Issues", "event_cause_macro_Other / Unspecified",
    "event_cause_macro_Planned & Public Events", "event_cause_macro_Weather Events",
    "event_cause_macro_congestion",
]


@dataclass
class RawEventInput:
    """Friendly, UI-facing event description. build_feature_row() turns this
    into the model's expected 30-column vector."""
    event_type: str                  # "planned" | "unplanned"
    latitude: float
    longitude: float
    event_cause: str                 # granular cause, e.g. "vehicle_breakdown" -> mapped to macro
    authenticated: bool = False
    is_major_corridor: bool = False
    at_junction: bool = False
    priority: str = "High"           # "High" | "Low" | "Unknown" — officer's intake assessment
    direction: str = "north_or_other"  # one of DIRECTION_DUMMY_VALUES, or "north_or_other" baseline
    cargo_macro: str = "empty_or_none"  # one of CARGO_MACRO_DUMMY_VALUES, or "construction_heavy" baseline
    breakdown_severity: str = "not_applicable"  # one of BREAKDOWN_SEVERITY_DUMMY_VALUES, or "general_breakdown" baseline
    endlatitude: float = None         # if event spans a stretch of road
    endlongitude: float = None
    veh_type: str = None              # captured for display only — NOT used by these models (finding #2)


def event_cause_to_macro(event_cause: str) -> str:
    return CAUSE_MACRO_MAPPING.get(event_cause.strip().lower(), CAUSE_MACRO_DEFAULT)


def build_feature_row(raw: RawEventInput) -> pd.DataFrame:
    """Returns a single-row DataFrame with exactly MODEL_FEATURE_COLUMNS, ready
    to pass to xgboost_duration_model.predict() / xgboost_closure_model.predict_proba()."""
    row = {c: 0 for c in MODEL_FEATURE_COLUMNS}

    row["event_type"] = 1 if raw.event_type == "planned" else 0
    row["latitude"] = raw.latitude
    row["longitude"] = raw.longitude

    is_stretch = raw.endlatitude is not None and raw.endlongitude is not None
    row["is_stretch_event"] = 1 if is_stretch else 0
    row["endlatitude"] = raw.endlatitude if is_stretch else raw.latitude
    row["endlongitude"] = raw.endlongitude if is_stretch else raw.longitude

    row["authenticated"] = 1 if raw.authenticated else 0
    row["is_major_corridor"] = 1 if raw.is_major_corridor else 0
    row["at_junction"] = 1 if raw.at_junction else 0

    if raw.priority == "Low":
        row["priority_Low"] = 1
    elif raw.priority == "Unknown":
        row["priority_Unknown"] = 1
    # else "High": both stay 0 (implicit baseline, see finding #2)

    if raw.cargo_macro in CARGO_MACRO_DUMMY_VALUES:
        row[f"cargo_macro_{raw.cargo_macro}"] = 1
    # else "construction_heavy" baseline: all cargo_macro_* stay 0

    if raw.breakdown_severity in BREAKDOWN_SEVERITY_DUMMY_VALUES:
        row[f"breakdown_severity_{raw.breakdown_severity}"] = 1
    # else "general_breakdown" baseline: all breakdown_severity_* stay 0

    if raw.direction in DIRECTION_DUMMY_VALUES:
        row[f"direction_{raw.direction}"] = 1
    # else "north_or_other" baseline: all direction_* stay 0

    macro = event_cause_to_macro(raw.event_cause)
    if macro in EVENT_CAUSE_MACRO_DUMMY_VALUES:
        row[f"event_cause_macro_{macro}"] = 1
    # else "Accidents & Hazards" baseline: all event_cause_macro_* stay 0

    return pd.DataFrame([row])[MODEL_FEATURE_COLUMNS]
