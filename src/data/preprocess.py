import pandas as pd
from src.config import TENURE_BINS, TENURE_LABELS


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data.  Safe for both training and inference.
    BUG FIX: tenure_group now created BEFORE return (was dead code after return).
    BUG FIX: fillna uses type-aware fill (not blanket fillna(0)).
    """
    df = df.copy()

    # Replace blank strings with NaN
    df = df.replace(r'^\s*$', None, regex=True)

    # Fix TotalCharges (often comes as string with spaces)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # SeniorCitizen: coerce to string for consistent OHE treatment
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # Drop customer ID - not a feature
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Type-aware fills (BUG FIX: not blanket fillna(0) which corrupts categoricals)
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    # BUG FIX: tenure_group created HERE (was after return → dead code)
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=TENURE_BINS,
            labels=TENURE_LABELS,
            include_lowest=True,
        ).astype(str)

    return df
