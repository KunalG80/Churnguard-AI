import pandas as pd


def segment_roi_table(seg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format segment ROI dataframe for display.
    Fixed: Net ROI column was labelled '₹' — ROI is a ratio, not currency.
    """
    if seg_df.empty:
        return seg_df

    display = seg_df[[c for c in [
        "Risk_Tier", "Customers", "Revenue_at_Risk",
        "Retention_Investment", "Recoverable_Revenue",
        "Net_Value_Created", "Net_ROI", "Breakeven_Success_%",
        "Recommendation",
    ] if c in seg_df.columns]].copy()

    # Format currency columns
    for col in ["Revenue_at_Risk", "Retention_Investment",
                "Recoverable_Revenue", "Net_Value_Created"]:
        if col in display.columns:
            display[col] = display[col].map(lambda v: f"₹{v:,.0f}")

    # Net ROI as percentage (was mislabelled ₹)
    if "Net_ROI" in display.columns:
        display["Net_ROI"] = display["Net_ROI"].map(lambda v: f"{v*100:.1f}%")

    if "Breakeven_Success_%" in display.columns:
        display["Breakeven_Success_%"] = display["Breakeven_Success_%"].map(
            lambda v: f"{v:.1f}%"
        )

    return display.rename(columns={
        "Risk_Tier":           "Risk Tier",
        "Revenue_at_Risk":     "Revenue Exposure",
        "Retention_Investment":"Retention Spend",
        "Recoverable_Revenue": "Recoverable Revenue",
        "Net_Value_Created":   "Net Value Created",
        "Net_ROI":             "Net ROI",           # no ₹ symbol — it's a ratio
        "Breakeven_Success_%": "Breakeven Rate",
    })
