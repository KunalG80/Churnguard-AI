import pandas as pd
import numpy as np
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.utils.segment_roi import segment_roi_analysis, budget_constrained_targets


def make_df(n=100, churn_prob=0.8):
    np.random.seed(42)
    return pd.DataFrame({
        "Churn_Prediction":  [1] * n,
        "Churn_Probability": np.random.uniform(0.4, 1.0, n),
        "MonthlyCharges":    np.random.uniform(30, 100, n),
    })


def test_invest_avoid_labels():
    df  = make_df()
    seg = segment_roi_analysis(df, retention_cost=500, months_lost=12, success_rate=0.5)
    assert set(seg["Recommendation"]).issubset({"INVEST", "AVOID"})


def test_net_value_sign_matches_recommendation():
    df  = make_df()
    seg = segment_roi_analysis(df, retention_cost=500, months_lost=12, success_rate=0.5)
    invest_rows = seg[seg["Recommendation"] == "INVEST"]
    assert (invest_rows["Net_Value_Created"] > 0).all()


def test_only_churners_included():
    """ARCH FIX: ROI should run on Churn_Prediction==1 only."""
    df = make_df(50)
    df.loc[:24, "Churn_Prediction"] = 0   # half are retained
    seg = segment_roi_analysis(df, 500, 12, 0.3)
    assert seg["Customers"].sum() == 25   # only churners counted


def test_budget_constrained_respects_budget():
    df = make_df(200)
    budget = 10_000
    result = budget_constrained_targets(df, retention_cost=500, months_lost=12,
                                         success_rate=0.3, total_budget=budget)
    if not result.empty:
        assert result["Cumulative_Spend"].max() <= budget
