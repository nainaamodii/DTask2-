"""
Week 2 - Rule-Based Baseline Impact Scoring Engine
=====================================================
This is the "gets something demonstrable immediately" baseline from the plan.
It runs WITHOUT any trained model — pure weighted lookup tables derived from
the EDA in Week 1 (data/processed_events.csv -> historical high-priority rate
and closure rate per cause/corridor/time-window).

Impact Score (0-10) = base_weight[event_cause]
                     * corridor_multiplier
                     * peak_hour_multiplier
                     * weekend_multiplier

Resource recommendation is then derived from thresholds on that score.

This module is imported by both:
 - scripts/03_rule_based_eval.py (offline evaluation against historical priority)
 - backend/app/rule_engine.py (live FastAPI use - re-exported there)

Run standalone for a quick demo:
    python scripts/03_rule_based_baseline.py
"""

from dataclasses import dataclass

# Base weight per event cause (~1-5 scale before multipliers), calibrated from historical
# high-priority rate seen in EDA (scripts/02_eda.py output). Multipliers below stack on top
# of this, so headroom is intentionally left for peak/corridor/closure factors to push the
# final 0-10 score up — a cause only hits CRITICAL when several risk factors align.
BASE_WEIGHT = {
    "vehicle_breakdown": 4.0,
    "construction": 4.0,
    "water_logging": 3.8,
    "others": 3.8,
    "pot_holes": 3.4,
    "road_conditions": 3.4,
    "congestion": 4.4,
    "debris": 4.0,
    "accident": 5.0,        # lower historical %High but high real-world severity when it occurs
    "public_event": 3.8,
    "protest": 4.4,
    "vip_movement": 3.4,
    "procession": 3.0,
    "tree_fall": 2.5,
    "fog_low_visibility": 3.0,
    "test_demo": 0.5,
}
DEFAULT_BASE_WEIGHT = 3.2

CORRIDOR_MULTIPLIER_ON = 1.15      # event sits on a named arterial corridor
CORRIDOR_MULTIPLIER_OFF = 1.0      # non-corridor / local road

PEAK_HOUR_MULTIPLIER = 1.2         # morning/evening peak window
OFF_PEAK_MULTIPLIER = 1.0
NIGHT_MULTIPLIER = 0.75

WEEKEND_DAMPENER = 0.9             # slightly lower base traffic on weekends
ROAD_CLOSURE_BOOST = 1.3           # event explicitly requires closing the road


@dataclass
class ImpactResult:
    impact_score: float          # 0-10
    severity_band: str           # LOW / MEDIUM / HIGH / CRITICAL
    manpower: int
    barricades: int
    diversions: int


def time_multiplier(time_window: str) -> float:
    if time_window in ("morning_peak", "evening_peak"):
        return PEAK_HOUR_MULTIPLIER
    if time_window == "night":
        return NIGHT_MULTIPLIER
    return OFF_PEAK_MULTIPLIER


def compute_impact_score(event_cause: str, is_corridor_road: bool, time_window: str,
                          is_weekend: bool, requires_road_closure: bool = False) -> float:
    base = BASE_WEIGHT.get(event_cause, DEFAULT_BASE_WEIGHT)
    score = base
    score *= CORRIDOR_MULTIPLIER_ON if is_corridor_road else CORRIDOR_MULTIPLIER_OFF
    score *= time_multiplier(time_window)
    score *= WEEKEND_DAMPENER if is_weekend else 1.0
    if requires_road_closure:
        score *= ROAD_CLOSURE_BOOST
    return min(round(score, 2), 10.0)


def severity_band(score: float) -> str:
    if score >= 8:
        return "CRITICAL"
    if score >= 6:
        return "HIGH"
    if score >= 4:
        return "MEDIUM"
    return "LOW"


def recommend_resources(score: float, requires_road_closure: bool = False) -> dict:
    """Lookup-table style resource recommendation, scaled by impact score."""
    band = severity_band(score)

    table = {
        "CRITICAL": dict(manpower=24, barricades=14, diversions=3),
        "HIGH":     dict(manpower=14, barricades=8,  diversions=2),
        "MEDIUM":   dict(manpower=6,  barricades=4,  diversions=1),
        "LOW":      dict(manpower=2,  barricades=1,  diversions=0),
    }
    rec = table[band].copy()
    if requires_road_closure and band in ("LOW", "MEDIUM"):
        # any explicit closure needs at least minimal containment, even if base score is low
        rec["barricades"] = max(rec["barricades"], 4)
        rec["manpower"] = max(rec["manpower"], 6)
    return rec


def score_event(event_cause: str, is_corridor_road: bool, time_window: str,
                 is_weekend: bool, requires_road_closure: bool = False) -> ImpactResult:
    score = compute_impact_score(event_cause, is_corridor_road, time_window, is_weekend, requires_road_closure)
    band = severity_band(score)
    rec = recommend_resources(score, requires_road_closure)
    return ImpactResult(impact_score=score, severity_band=band, **rec)


if __name__ == "__main__":
    demo_events = [
        dict(event_cause="vehicle_breakdown", is_corridor_road=True, time_window="evening_peak",
             is_weekend=False, requires_road_closure=False),
        dict(event_cause="accident", is_corridor_road=True, time_window="morning_peak",
             is_weekend=False, requires_road_closure=True),
        dict(event_cause="tree_fall", is_corridor_road=False, time_window="night",
             is_weekend=True, requires_road_closure=True),
        dict(event_cause="protest", is_corridor_road=True, time_window="evening_peak",
             is_weekend=False, requires_road_closure=True),
    ]
    for e in demo_events:
        r = score_event(**e)
        print(f"{e} -> {r}")
