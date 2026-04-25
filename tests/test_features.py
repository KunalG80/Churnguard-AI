import pandas as pd
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.features.build_features import build_features


BASE = pd.DataFrame({
    "tenure":           [1, 12, 36, 61],
    "MonthlyCharges":   [30.0, 55.0, 75.0, 99.0],
    "TotalCharges":     [30.0, 660.0, 2700.0, 6039.0],
    "PhoneService":     ["Yes", "No", "Yes", "Yes"],
    "OnlineSecurity":   ["No", "Yes", "Yes", "Yes"],
    "StreamingTV":      ["No", "No", "Yes", "Yes"],
})


def test_service_count_created():
    out = build_features(BASE)
    assert "service_count" in out.columns


def test_charge_per_service_created():
    out = build_features(BASE)
    assert "charge_per_service" in out.columns


def test_avg_monthly_spend_created():
    out = build_features(BASE)
    assert "avg_monthly_spend" in out.columns


def test_is_new_customer_flag():
    out = build_features(BASE)
    assert "is_new_customer" in out.columns
    assert out.loc[out["tenure"] == 1, "is_new_customer"].iloc[0] == 1
    assert out.loc[out["tenure"] == 36, "is_new_customer"].iloc[0] == 0


def test_tenure_group_labels_match_config():
    """BUG FIX: labels must match TENURE_LABELS in config, not a different set."""
    from src.config import TENURE_LABELS
    out = build_features(BASE)
    assert set(out["tenure_group"].unique()).issubset(set(TENURE_LABELS))
