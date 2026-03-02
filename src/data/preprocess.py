import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean uploaded or training data.
    Pandas 3.0 safe version.
    """
    df = df.copy()
    # Replace blank strings
    df = df.replace(r'^\s*$', None, regex=True)

    # Fix numeric column
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    # Fix type mismatch
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # Drop ID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"],axis=1)

    # Fill numeric NA
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(
        df[numeric_cols].median()
    )

    # Fill categorical NA (SAFE WAY)
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df

# -----------------------------------------
# CREATE tenure_group (TRAINING FEATURE)
# -----------------------------------------

    if "tenure" in df.columns:

        df["tenure"] = pd.to_numeric(
            df["tenure"],
            errors="coerce"
        )

        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0,12,24,48,60,100],
            labels=[
                "0-12",
                "12-24",
                "24-48",
                "48-60",
                "60+"
            ]
        )

        df["tenure_group"] = df["tenure_group"].astype(str)