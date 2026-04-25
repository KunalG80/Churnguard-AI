"""
Build the executive summary dict consumed by PDF and PPT exporters.
"""

import pandas as pd


def executive_summary(seg_df: pd.DataFrame) -> dict:
    """
    Extract key headline numbers from the segment ROI table.

    Returns a flat dict of human-readable KPIs.
    """

    def _get(tier: str, col: str, default=0):
        row = seg_df[seg_df["Risk_Tier"] == tier]
        return float(row[col].values[0]) if not row.empty else default

    total_risk        = float(seg_df["Revenue_at_Risk"].sum())
    total_recoverable = float(seg_df["Recoverable_Revenue"].sum())
    total_investment  = float(seg_df["Retention_Investment"].sum())
    total_net         = float(seg_df["Net_Value_Created"].sum())
    overall_roi       = (total_net / total_investment) if total_investment > 0 else 0

    summary = {
        "Revenue at Risk":           total_risk,
        "Total Recoverable Revenue": total_recoverable,
        "Total Retention Investment": total_investment,
        "Net Value Created":         total_net,
        "Overall ROI":               overall_roi,
        "Low Risk ROI":              _get("Low",    "Net_ROI"),
        "Medium Risk ROI":           _get("Medium", "Net_ROI"),
        "High Risk ROI":             _get("High",   "Net_ROI"),
        "Low Risk Net Value":        _get("Low",    "Net_Value_Created"),
        "Medium Risk Net Value":     _get("Medium", "Net_Value_Created"),
        "High Risk Net Value":       _get("High",   "Net_Value_Created"),
    }
    return summary
