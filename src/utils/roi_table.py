def segment_roi_table(seg_df):

    rename_map = {

        "Customers": "Customers",
        "Revenue_at_Risk": "Revenue Exposure (₹)",
        "Retention_Investment": "Retention Spend (₹)",
        "Recoverable_Revenue": "Recoverable Revenue (₹)",
        "Net_ROI": "Net ROI (₹)"
    }

    cols = [
        "Risk_Tier",
        "Customers",
        "Revenue_at_Risk",
        "Retention_Investment",
        "Recoverable_Revenue",
        "Net_ROI"
    ]

    available_cols = [
        c for c in cols if c in seg_df.columns
    ]

    table = seg_df[available_cols].rename(
        columns=rename_map
    )

    return table