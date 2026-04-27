import logging
import pandas as pd
from src.config import TENURE_BINS, TENURE_LABELS

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data — safe for both training and inference.

    Fixes vs original:
      - tenure_group created BEFORE return (was unreachable dead code)
      - fillna is type-aware (blanket fillna(0) corrupted categoricals)
      - select_dtypes uses ["object","string"] for pandas 3.x compat
      - TENURE_LABELS imported from config (was hardcoded, mismatched)
    """
    df = df.copy()

    # Replace blank strings with NaN
    df = df.replace(r"^\s*$", None, regex=True)

    # TotalCharges: often arrives as string with spaces on new-customer rows
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # SeniorCitizen: treat as categorical (0/1 strings) for consistent OHE
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # Drop ID column — not a feature
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Type-aware fills
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include=["object", "string"]).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    # ── tenure_group (FIXED: was dead code after return statement) ────────────
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=TENURE_BINS,
            labels=TENURE_LABELS,
            include_lowest=True,
        ).astype(str)

    logger.debug("clean_data: output shape %s", df.shape)
    return df
