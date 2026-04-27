import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.executive_summary import executive_summary

SEG = pd.DataFrame({
    "Risk_Tier":            ["Low", "Medium", "High"],
    "Revenue_at_Risk":      [10000.0, 50000.0, 80000.0],
    "Recoverable_Revenue":  [3000.0,  15000.0, 24000.0],
    "Net_Value_Created":    [1000.0,  8000.0,  10000.0],
    "Net_ROI":              [0.1, 0.5, 0.8],
    "Recommendation":       ["INVEST", "INVEST", "INVEST"],
})

REQUIRED_KEYS = [
    "Revenue at Risk",
    "Total Recoverable Revenue",
    "Net Value Created",
    "Overall ROI",
    "Medium Risk ROI",
    "High Risk ROI",
]

def test_returns_all_ppt_keys():
    """FIX: PPT KPI slide was showing ₹0 for 3 of 4 metrics due to missing keys."""
    summary = executive_summary(SEG)
    for key in REQUIRED_KEYS:
        assert key in summary, f"Missing key: {key}"

def test_empty_seg_no_crash():
    empty = pd.DataFrame(columns=SEG.columns)
    summary = executive_summary(empty)
    assert summary["Revenue at Risk"] == 0

def test_revenue_at_risk_totals():
    summary = executive_summary(SEG)
    assert summary["Revenue at Risk"] == SEG["Revenue_at_Risk"].sum()
