"""
Week 1 - Exploratory Data Analysis
====================================
Generates summary statistics and charts from the processed event dataset.
Outputs land in outputs/eda/

Run:
    python scripts/02_eda.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed_events.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "eda")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["figure.dpi"] = 110


def main():
    df = pd.read_csv(DATA_PATH)
    lines = []
    lines.append("# EDA Summary — Astram Traffic Event Data\n")
    lines.append(f"Total cleaned events: **{len(df)}**\n")

    # 1. Event cause distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    df["event_cause"].value_counts().plot(kind="barh", ax=ax, color="#3b6fd4")
    ax.set_title("Event Count by Cause")
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_event_cause_distribution.png"))
    plt.close(fig)

    # 2. Priority by event cause (stacked %)
    ct = pd.crosstab(df["event_cause"], df["priority"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="barh", stacked=True, ax=ax, color=["#e0763a", "#3b6fd4"])
    ax.set_title("Priority Split (%) by Event Cause")
    ax.set_xlabel("% of events")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_priority_by_cause.png"))
    plt.close(fig)

    # 3. Events by hour of day
    fig, ax = plt.subplots(figsize=(8, 4))
    df["hour_of_day"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#3b6fd4")
    ax.set_title("Events by Hour of Day (IST)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_events_by_hour.png"))
    plt.close(fig)

    # 4. Events by day of week
    fig, ax = plt.subplots(figsize=(7, 4))
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    counts = df["day_of_week"].value_counts().sort_index()
    ax.bar([dow_labels[i] for i in counts.index], counts.values, color="#3b6fd4")
    ax.set_title("Events by Day of Week")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "04_events_by_dow.png"))
    plt.close(fig)

    # 5. Top corridors by event count
    fig, ax = plt.subplots(figsize=(8, 5))
    df[df["corridor"] != "Non-corridor"]["corridor"].value_counts().head(15).plot(
        kind="barh", ax=ax, color="#3b6fd4")
    ax.set_title("Top 15 Corridors by Event Count")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "05_top_corridors.png"))
    plt.close(fig)

    # 6. Road closure rate by time window
    fig, ax = plt.subplots(figsize=(6, 4))
    df.groupby("time_window")["target_road_closure"].mean().sort_values().plot(
        kind="bar", ax=ax, color="#e0763a")
    ax.set_title("Road Closure Rate by Time Window")
    ax.set_ylabel("Closure rate")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "06_closure_rate_by_window.png"))
    plt.close(fig)

    # Text summary stats
    lines.append("## Event type split\n")
    lines.append(df["event_type"].value_counts().to_string() + "\n")
    lines.append("\n## Priority split\n")
    lines.append(df["priority"].value_counts(normalize=True).round(3).to_string() + "\n")
    lines.append("\n## Road closure rate overall\n")
    lines.append(f"{df['target_road_closure'].mean():.3f}\n")
    lines.append("\n## High-priority rate by cause (top causes)\n")
    top_causes = df["event_cause"].value_counts().head(8).index
    lines.append(df[df['event_cause'].isin(top_causes)].groupby("event_cause")["target_priority"].mean()
                  .sort_values(ascending=False).round(3).to_string() + "\n")
    lines.append("\n## Duration (minutes) — only for events with a logged close/resolve time, capped at 24h\n")
    lines.append(df["duration_minutes"].describe().round(1).to_string() + "\n")

    with open(os.path.join(OUT_DIR, "eda_summary.md"), "w") as f:
        f.write("\n".join(lines))

    print("EDA complete. Charts + summary written to:", OUT_DIR)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
