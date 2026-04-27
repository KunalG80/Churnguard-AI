import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.segment_roi import segment_roi_analysis, budget_constrained_targets
from src.utils.roi_analysis import roi_vs_threshold

def _df(n=80, all_churn=True):
    np.random.seed(42)
    return pd.DataFrame({
        "Churn_Prediction":  [1] * n if all_churn else [1]*(n//2) + [0]*(n//2),
        "Churn_Probability": np.random.uniform(0.4, 1.0, n),
        "MonthlyCharges":    np.random.uniform(30, 100, n),
    })

def test_invest_avoid_labels():
    seg = segment_roi_analysis(_df(), 500, 12, 0.5)
    assert set(seg["Recommendation"]).issubset({"INVEST", "AVOID"})

def test_net_value_sign_matches_recommendation():
    seg = segment_roi_analysis(_df(), 500, 12, 0.5)
    assert (seg[seg["Recommendation"] == "INVEST"]["Net_Value_Created"] > 0).all()

def test_only_churners_included():
    """ARCH FIX: ROI must run on Churn_Prediction==1 only."""
    df  = _df(80, all_churn=False)
    seg = segment_roi_analysis(df, 500, 12, 0.3)
    assert seg["Customers"].sum() == 40

def test_budget_constrained_respects_budget():
    df     = _df(200)
    result = budget_constrained_targets(df, 500, 12, 0.3, total_budget=10_000)
    if not result.empty:
        assert result["Cumulative_Spend"].max() <= 10_000

def test_roi_sweep_returns_dataframe():
    df = _df(100)
    result = roi_vs_threshold(df, 500, 0.3, 12)
    assert isinstance(result, pd.DataFrame)
    assert "Net_ROI" in result.columns
    assert "Threshold" in result.columns
