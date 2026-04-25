import numpy as np
import pandas as pd


def segment_roi_analysis(df: pd.DataFrame, retention_cost: float,
                          months_lost: int, success_rate: float) -> pd.DataFrame:
    """
    ARCH FIX: operates only on predicted churners (Churn_Prediction == 1).
    Previously ran on all customers, inflating investment and distorting KPIs.
    """
    # Filter to predicted churners only
    churners = df[df["Churn_Prediction"] == 1].copy()

    if churners.empty:
        return pd.DataFrame(columns=[
            "Risk_Tier", "Customers", "Avg_Probability",
            "Avg_Monthly_Charges", "Revenue_at_Risk", "Retention_Investment",
            "Recoverable_Revenue", "Net_Value_Created", "Net_ROI",
            "Breakeven_Success_%", "Recommendation",
        ])

    # Risk tiers based on churn probability
    churners["Risk_Tier"] = pd.cut(
        churners["Churn_Probability"],
        bins=[0, 0.4, 0.7, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    seg = (
        churners.groupby("Risk_Tier", observed=True)
        .agg(
            Customers=("Churn_Prediction", "count"),
            Avg_Probability=("Churn_Probability", "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
        )
        .reset_index()
    )

    seg["Revenue_at_Risk"]     = seg["Customers"] * seg["Avg_Probability"] * seg["Avg_Monthly_Charges"] * months_lost
    seg["Retention_Investment"]= seg["Customers"] * retention_cost
    seg["Recoverable_Revenue"] = seg["Revenue_at_Risk"] * success_rate
    seg["Net_Value_Created"]   = seg["Recoverable_Revenue"] - seg["Retention_Investment"]
    seg["Net_ROI"]             = (
        seg["Net_Value_Created"] / seg["Retention_Investment"]
    ).replace([np.inf, -np.inf], 0).fillna(0)
    seg["Breakeven_Success_%"] = (
        (seg["Retention_Investment"] / seg["Revenue_at_Risk"]).clip(upper=1) * 100
    ).fillna(0)
    seg["Recommendation"] = np.where(seg["Net_Value_Created"] > 0, "INVEST", "AVOID")

    return seg


def budget_constrained_targets(df: pd.DataFrame, retention_cost: float,
                                months_lost: int, success_rate: float,
                                total_budget: float) -> pd.DataFrame:
    """
    NEW: Rank predicted churners by ROI and greedily assign budget top-down.
    Returns only the customers reachable within budget, with cumulative spend.
    """
    churners = df[df["Churn_Prediction"] == 1].copy()
    if churners.empty:
        return churners

    churners["Expected_Revenue_Saved"] = (
        churners["Churn_Probability"]
        * churners["MonthlyCharges"]
        * months_lost
        * success_rate
    )
    churners["Net_Value"] = churners["Expected_Revenue_Saved"] - retention_cost
    churners = churners.sort_values("Net_Value", ascending=False).reset_index(drop=True)

    churners["Cumulative_Spend"] = (np.arange(len(churners)) + 1) * retention_cost
    reachable = churners[churners["Cumulative_Spend"] <= total_budget].copy()

    return reachable
