import pandas as pd
import numpy as np

def segment_roi_analysis(df, retention_cost, months_lost, success_rate):

    # ------------------------------
    # 1. RISK TIERS
    # ------------------------------
    df["Risk_Tier"] = pd.cut(
        df["Churn_Probability"],
        bins=[0,0.4,0.7,1],
        labels=["Low","Medium","High"]
    )

    # ------------------------------
    # 2. AGGREGATION
    # ------------------------------
    seg = (
        df.groupby("Risk_Tier", observed=True)
        .agg(
            Customers=("Churn_Prediction","count"),
            Avg_Probability=("Churn_Probability","mean"),
            Avg_Monthly_Charges=("MonthlyCharges","mean")
        )
        .reset_index()
    )

    # ------------------------------
    # 3. REVENUE AT RISK
    # ------------------------------
    seg["Revenue_at_Risk"] = (
        seg["Customers"]
        * seg["Avg_Probability"]
        * seg["Avg_Monthly_Charges"]
        * months_lost
    )

    # ------------------------------
    # 4. INVESTMENT REQUIRED
    # ------------------------------
    seg["Retention_Investment"] = (
        seg["Customers"]
        * retention_cost
    )

    # ------------------------------
    # 5. RECOVERABLE VALUE
    # ------------------------------
    seg["Recoverable_Revenue"] = (
        seg["Revenue_at_Risk"]
        * success_rate
    )

    # ------------------------------
    # 6. NET VALUE CREATED ₹
    # ------------------------------
    seg["Net_Value_Created"] = (
        seg["Recoverable_Revenue"]
        - seg["Retention_Investment"]
    )

    # ------------------------------
    # 7. ROI %
    # ------------------------------
    seg["Net_ROI"] = (
        seg["Net_Value_Created"]
        /
        seg["Retention_Investment"]
    ).replace([np.inf,-np.inf],0).fillna(0)

    # ------------------------------
    # 8. BREAKEVEN SUCCESS %
    # ------------------------------
    seg["Breakeven_Success_%"] = (
        seg["Retention_Investment"]
        /
        seg["Revenue_at_Risk"]
    ).clip(upper=1) * 100

    # ------------------------------
    # 9. CFO DECISION LAYER
    # ------------------------------
    seg["Recommendation"] = np.where(
        seg["Net_Value_Created"] > 0,
        "INVEST",
        "AVOID"
    )

    return seg