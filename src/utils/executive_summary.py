import pandas as pd

def executive_summary(seg_df):

    def get_roi(tier):

        row = seg_df[
            seg_df["Risk_Tier"] == tier
        ]

        if row.empty:
            return 0

        return row["Net_ROI"].values[0]

    med_roi = get_roi("Medium")
    high_roi = get_roi("High")

    summary = {

        "Revenue at Risk":
        seg_df["Revenue_at_Risk"].sum(),

        "Medium Risk ROI":
        med_roi,

        "High Risk ROI":
        high_roi

    }

    return summary