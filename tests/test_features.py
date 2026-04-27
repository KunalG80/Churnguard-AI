import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.build_features import build_features
from src.config import TENURE_LABELS

BASE = pd.DataFrame({
    "tenure":          [1, 12, 36, 61],
    "MonthlyCharges":  [30.0, 55.0, 75.0, 99.0],
    "TotalCharges":    [30.0, 660.0, 2700.0, 6039.0],
    "PhoneService":    ["Yes", "No", "Yes", "Yes"],
    "OnlineSecurity":  ["No", "Yes", "Yes", "Yes"],
    "StreamingTV":     ["No", "No", "Yes", "Yes"],
})

def test_service_count_created():
    assert "service_count" in build_features(BASE).columns

def test_charge_per_service_created():
    assert "charge_per_service" in build_features(BASE).columns

def test_avg_monthly_spend_created():
    assert "avg_monthly_spend" in build_features(BASE).columns

def test_is_new_customer_flag():
    out = build_features(BASE)
    assert out.loc[out["tenure"] == 1,  "is_new_customer"].iloc[0] == 1
    assert out.loc[out["tenure"] == 36, "is_new_customer"].iloc[0] == 0

def test_tenure_group_labels_match_config():
    """BUG FIX: was '0-1yr' etc — mismatch with preprocess.py."""
    out = build_features(BASE)
    assert set(out["tenure_group"].unique()).issubset(set(TENURE_LABELS))
