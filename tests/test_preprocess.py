import pandas as pd
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.data.preprocess import clean_data


SAMPLE = pd.DataFrame({
    "customerID":    ["X1", "X2", "X3"],
    "gender":        ["Male", "Female", None],
    "SeniorCitizen": [0, 1, 0],
    "tenure":        [1, 24, 60],
    "MonthlyCharges":[29.85, 56.95, 99.0],
    "TotalCharges":  [29.85, "1889.50", " "],   # mixed types
    "Contract":      ["Month-to-month", "One year", "Two year"],
})


def test_row_count_preserved():
    out = clean_data(SAMPLE)
    assert len(out) == len(SAMPLE)


def test_customer_id_dropped():
    out = clean_data(SAMPLE)
    assert "customerID" not in out.columns


def test_total_charges_numeric():
    out = clean_data(SAMPLE)
    assert pd.api.types.is_float_dtype(out["TotalCharges"])


def test_tenure_group_created():
    """BUG FIX: was dead code after return — now must be present."""
    out = clean_data(SAMPLE)
    assert "tenure_group" in out.columns


def test_tenure_group_labels():
    out = clean_data(SAMPLE)
    assert out["tenure_group"].notna().all()
    assert out.loc[out["tenure"] == 1, "tenure_group"].iloc[0] == "0-12m"


def test_no_null_categoricals():
    out = clean_data(SAMPLE)
    cat_cols = out.select_dtypes(include="object").columns
    assert out[cat_cols].isnull().sum().sum() == 0
