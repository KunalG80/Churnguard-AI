import pandas as pd


def executive_summary(seg_df: pd.DataFrame) -> dict:
    """
    Compute summary KPIs from segment ROI dataframe.

    Fixed: now returns ALL keys expected by ppt_export._add_kpi_slide.
    Previous version returned only 3 keys; PPT showed ₹0 for 3 of 4 KPIs.
    """
    if seg_df.empty:
        return {
            "Revenue at Risk":           0,
            "Total Recoverable Revenue": 0,
            "Net Value Created":         0,
            "Overall ROI":               0,
            "Medium Risk ROI":           0,
            "High Risk ROI":             0,
        }

    def tier_net_roi(tier: str) -> float:
        row = seg_df[seg_df["Risk_Tier"] == tier]
        return float(row["Net_ROI"].iloc[0]) if not row.empty else 0.0

    return {
        "Revenue at Risk":           float(seg_df["Revenue_at_Risk"].sum()),
        "Total Recoverable Revenue": float(seg_df["Recoverable_Revenue"].sum()),
        "Net Value Created":         float(seg_df["Net_Value_Created"].sum()),
        "Overall ROI":               float(seg_df["Net_ROI"].mean()),
        "Medium Risk ROI":           tier_net_roi("Medium"),
        "High Risk ROI":             tier_net_roi("High"),
    }
