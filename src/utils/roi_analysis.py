"""
ROI vs threshold sweep.
Previously orphaned (never called) — now wired into app.py as an
interactive chart helping users pick the optimal decision threshold.
"""
import numpy as np
import pandas as pd


def roi_vs_threshold(
    df: pd.DataFrame,
    retention_cost: float,
    success_rate: float,
    months_lost: int,
) -> pd.DataFrame:
    """
    Sweep decision thresholds 0.1 → 0.9 and compute net ROI at each.
    Returns a DataFrame suitable for Plotly line chart.
    """
    rows = []
    for t in np.arange(0.1, 0.95, 0.05):
        subset = df[df["Churn_Probability"] >= t]
        if subset.empty:
            continue
        n         = len(subset)
        loss      = subset["MonthlyCharges"].sum() * months_lost
        saved     = loss * success_rate
        spend     = n * retention_cost
        net       = saved - spend
        precision = n / max(len(df), 1)
        rows.append({
            "Threshold":         round(float(t), 2),
            "Customers_Targeted": n,
            "Net_ROI":           round(net, 0),
            "Retention_Spend":   round(spend, 0),
            "Revenue_Saved":     round(saved, 0),
            "Targeting_Rate_%":  round(precision * 100, 1),
        })
    return pd.DataFrame(rows)
