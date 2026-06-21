"""
Week 1 - Data Collection, Cleaning, Feature Engineering
=========================================================
Source: Astram traffic event data (Bengaluru Traffic Police / ASTRAM system).

This script:
1. Loads the raw anonymized event log.
2. Cleans datatypes (datetimes, booleans, categorical normalization).
3. Engineers the features the downstream rule-based and ML models use.
4. Saves a clean, model-ready CSV to data/processed_events.csv

Run:
    python scripts/01_preprocessing.py
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "astram_event_data.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed_events.csv")

# Peak traffic windows for Bengaluru (24h local clock)
MORNING_PEAK = (8, 11)   # 8:00 - 10:59
EVENING_PEAK = (17, 21)  # 17:00 - 20:59


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Normalize categorical text (some values have inconsistent casing, e.g. 'Debris' vs 'debris') ---
    df["event_cause"] = df["event_cause"].str.strip().str.lower()
    df["event_cause"] = df["event_cause"].replace({
        "fog / low visibility": "fog_low_visibility",
    })
    df["event_type"] = df["event_type"].str.strip().str.lower()
    df["priority"] = df["priority"].str.strip().str.title()
    df["status"] = df["status"].str.strip().str.lower()
    df["corridor"] = df["corridor"].fillna("Unknown")
    df["zone"] = df["zone"].fillna("Unknown")
    df["veh_type"] = df["veh_type"].fillna("not_applicable")

    # --- Datetime parsing ---
    for col in ["start_datetime", "end_datetime", "closed_datetime", "resolved_datetime", "created_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Drop rows with no usable start time or no event cause - can't engineer features for these
    before = len(df)
    df = df.dropna(subset=["start_datetime", "event_cause"])
    dropped = before - len(df)
    print(f"Dropped {dropped} rows with missing start_datetime/event_cause ({dropped/before:.1%})")

    # --- Resolve duplicate/typo categories ---
    df["requires_road_closure"] = df["requires_road_closure"].astype(bool)

    # --- Drop rows where priority is missing (target) ---
    df = df.dropna(subset=["priority"])
    df = df[df["priority"].isin(["High", "Low"])]

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Time-based features (convert UTC -> IST = UTC+5:30 for local relevance)
    local_dt = df["start_datetime"] + pd.Timedelta(hours=5, minutes=30)
    df["hour_of_day"] = local_dt.dt.hour
    df["day_of_week"] = local_dt.dt.dayofweek  # 0=Mon
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = local_dt.dt.month

    def peak_flag(h):
        if MORNING_PEAK[0] <= h <= MORNING_PEAK[1]:
            return "morning_peak"
        if EVENING_PEAK[0] <= h <= EVENING_PEAK[1]:
            return "evening_peak"
        if 22 <= h or h <= 5:
            return "night"
        return "off_peak"

    df["time_window"] = df["hour_of_day"].apply(peak_flag)
    df["is_peak_hour"] = df["time_window"].isin(["morning_peak", "evening_peak"]).astype(int)

    # Corridor type - arterial/corridor roads carry more through-traffic than local roads
    df["is_corridor_road"] = (df["corridor"] != "Non-corridor").astype(int)

    # Historical congestion index per corridor: how often a corridor shows up historically
    # (proxy for "how busy/important this stretch of road normally is")
    corridor_counts = df["corridor"].value_counts(normalize=True)
    df["corridor_event_share"] = df["corridor"].map(corridor_counts).fillna(0)

    # Historical severity rate per event_cause - what fraction of this cause type are usually High priority
    cause_high_rate = df.groupby("event_cause")["priority"].apply(lambda s: (s == "High").mean())
    df["cause_historical_high_rate"] = df["event_cause"].map(cause_high_rate)

    # Duration (only available for ~39% of closed/resolved events; kept as analysis field, NOT a clean model
    # target because of heavy-tailed/likely-erroneous values e.g. records left open for weeks)
    end_dt = df["closed_datetime"].fillna(df["resolved_datetime"])
    duration_min = (end_dt - df["start_datetime"]).dt.total_seconds() / 60
    df["duration_minutes"] = duration_min.where((duration_min >= 0) & (duration_min <= 1440))  # cap at 24h, drop garbage

    # Final label encodings used by the model
    df["target_priority"] = (df["priority"] == "High").astype(int)
    df["target_road_closure"] = df["requires_road_closure"].astype(int)

    return df


FEATURE_COLUMNS = [
    "event_type", "event_cause", "zone", "corridor", "veh_type",
    "hour_of_day", "day_of_week", "is_weekend", "month",
    "time_window", "is_peak_hour", "is_corridor_road",
    "corridor_event_share", "cause_historical_high_rate",
]

TARGET_COLUMNS = ["target_priority", "target_road_closure"]

KEEP_COLUMNS = ["id", "latitude", "longitude", "start_datetime", "duration_minutes",
                "priority", "requires_road_closure", "police_station"] + FEATURE_COLUMNS + TARGET_COLUMNS


def main():
    print("Loading raw data...")
    df = load_raw()
    print(f"Raw shape: {df.shape}")

    print("\nCleaning...")
    df = clean(df)
    print(f"Shape after cleaning: {df.shape}")

    print("\nEngineering features...")
    df = engineer_features(df)

    df_out = df[KEEP_COLUMNS]
    df_out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved processed dataset -> {OUT_PATH}")
    print(f"Final shape: {df_out.shape}")
    print(f"\nTarget balance (priority High=1): \n{df_out['target_priority'].value_counts(normalize=True)}")
    print(f"\nTarget balance (road_closure=1): \n{df_out['target_road_closure'].value_counts(normalize=True)}")


if __name__ == "__main__":
    main()
