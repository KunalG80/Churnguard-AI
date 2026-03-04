def clean_binary(df):

    binary_cols = [
        "Partner","Dependents","PhoneService",
        "MultipleLines","OnlineSecurity",
        "OnlineBackup","DeviceProtection",
        "TechSupport","StreamingTV",
        "StreamingMovies","PaperlessBilling"
    ]

    for col in binary_cols:

        if col in df.columns:

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            df[col] = df[col].replace({
                "yes":"Yes",
                "no":"No",
                "true":"Yes",
                "false":"No",
                "1":"Yes",
                "0":"No",
                " ":"No",
                "nan":"No"
            })

    return df