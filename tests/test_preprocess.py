import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocess import clean_data
from src.config import TENURE_LABELS

SAMPLE = pd.DataFrame({
    "customerID":    ["X1", "X2", "X3"],
    "gender":        ["Male", "Female", None],
    "SeniorCitizen": [0, 1, 0],
    "tenure":        [1, 24, 60],
    "MonthlyCharges":[29.85, 56.95, 99.0],
    "TotalCharges":  [29.85, "1889.50", " "],
    "Contract":      ["Month-to-month", "One year", "Two year"],
})


def test_row_count_preserved():
    assert len(clean_data(SAMPLE)) == len(SAMPLE)

def test_customer_id_dropped():
    assert "customerID" not in clean_data(SAMPLE).columns

def test_total_charges_numeric():
    assert pd.api.types.is_float_dtype(clean_data(SAMPLE)["TotalCharges"])

def test_tenure_group_created():
    """BUG FIX: was dead code after return."""
    assert "tenure_group" in clean_data(SAMPLE).columns

def test_tenure_group_uses_config_labels():
    """Labels must come from config — not a hardcoded set."""
    out = clean_data(SAMPLE)
    assert set(out["tenure_group"].unique()).issubset(set(TENURE_LABELS))

def test_tenure_group_first_bin():
    out = clean_data(SAMPLE)
    assert out.loc[out["tenure"] == 1, "tenure_group"].iloc[0] == "0-12m"

def test_no_null_categoricals():
    out = clean_data(SAMPLE)
    cat_cols = out.select_dtypes(include=["object", "string"]).columns
    assert out[cat_cols].isnull().sum().sum() == 0
