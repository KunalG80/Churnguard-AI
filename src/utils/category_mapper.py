import pandas as pd


def normalize_categories(df):

    for col in df.select_dtypes(include="object").columns:

        df[col] = df[col].astype(str).str.strip().str.lower()

        if "gender" in col.lower():
            df[col] = df[col].replace({
                "m": "male",
                "f": "female"
            })

        if "yes" in df[col].unique():
            df[col] = df[col].replace({
                "y": "yes",
                "1": "yes"
            })

        if "no" in df[col].unique():
            df[col] = df[col].replace({
                "n": "no",
                "0": "no"
            })

    return df