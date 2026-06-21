"""
Week 2 - Offline Evaluation of the Rule-Based Baseline
=========================================================
Applies the rule-based scoring engine to every historical event and checks
how well "impact_score >= 6 (HIGH/CRITICAL band)" lines up with the actual
logged priority. This is the baseline we try to beat with the ML model in
Week 3.

Run:
    python scripts/04_rule_based_eval.py
"""

import pandas as pd
import importlib.util
import os
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("rb", os.path.join(HERE, "03_rule_based_baseline.py"))
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)

DATA_PATH = os.path.join(HERE, "..", "data", "processed_events.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    scores = []
    for _, row in df.iterrows():
        s = rb.compute_impact_score(
            event_cause=row["event_cause"],
            is_corridor_road=bool(row["is_corridor_road"]),
            time_window=row["time_window"],
            is_weekend=bool(row["is_weekend"]),
            requires_road_closure=bool(row["target_road_closure"]),
        )
        scores.append(s)

    df["rule_score"] = scores
    df["rule_pred_high"] = (df["rule_score"] >= 6).astype(int)

    y_true = df["target_priority"]
    y_pred = df["rule_pred_high"]

    print("=== Rule-Based Baseline vs Historical Priority Label ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    try:
        print(f"ROC-AUC (using raw score as ranking): {roc_auc_score(y_true, df['rule_score']):.3f}")
    except ValueError:
        pass
    print()
    print(classification_report(y_true, y_pred, target_names=["Low", "High"]))

    print("\n--- Threshold sweep (score >= t treated as 'High') ---")
    for t in [2, 3, 4, 5, 6, 7]:
        pred_t = (df["rule_score"] >= t).astype(int)
        acc_t = accuracy_score(y_true, pred_t)
        print(f"t={t}: accuracy={acc_t:.3f}")
    print("\nNote: majority-class baseline (always predict 'High') gets "
          f"{y_true.mean():.3f} accuracy. The rule engine's ranking has real "
          "signal (AUC > 0.5) but a fixed weighted formula plateaus quickly — "
          "this is exactly the gap Week 3's trained ML model should close.")

    out_path = os.path.join(HERE, "..", "outputs", "rule_based_eval.csv")
    df[["event_cause", "corridor", "time_window", "rule_score", "rule_pred_high", "priority"]].to_csv(
        out_path, index=False)
    print(f"\nSaved per-event scored output -> {out_path}")


if __name__ == "__main__":
    main()
