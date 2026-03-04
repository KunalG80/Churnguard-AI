import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.config import *
from src.data.load_data import load_data
from src.data.preprocess import clean_data
from src.features.build_features import build_features

from src.utils.segment_roi import segment_roi_analysis
from src.utils.executive_summary import executive_summary
from src.utils.roi_table import segment_roi_table
from src.utils.report_bundle import build_report_bundle
from src.utils.pdf_export import export_pdf
from src.utils.ppt_export import export_ppt


# -------------------------
# CREATE REPORT FOLDER
# -------------------------
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# -------------------------
# LOAD DATA
# -------------------------
raw_df = load_data(DATA_RAW_PATH)

ml_df = raw_df.copy()
ml_df = clean_data(ml_df)
ml_df = build_features(ml_df)

ml_df.to_csv(
    "data/processed/processed.csv",
    index=False
)

X = ml_df.drop(columns=[TARGET_COL])
y = ml_df[TARGET_COL]

# -------------------------
# CLEAN CATEGORICALS
# -------------------------
cat_cols = X.select_dtypes(include="object").columns

for col in cat_cols:

    X[col] = (
        X[col]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

# -------------------------
# SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

train_df = X_train.copy()
train_df[TARGET_COL] = y_train

test_df = X_test.copy()
test_df[TARGET_COL] = y_test

train_df.to_csv(
    "data/processed/train.csv",
    index=False
)

test_df.to_csv(
    "data/processed/test.csv",
    index=False
)


# -------------------------
# PREPROCESSOR
# -------------------------

# --------------------------------------------------
# FORCE NUMERIC CONVERSION BEFORE SPLIT
# --------------------------------------------------
num_force = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen"
]

for col in num_force:
    if col in X.columns:
        X[col] = (
            X[col]
            .astype(str)
            .str.replace(r"[^\d\.]", "", regex=True)
        )
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )
        
# --------------------------------------------------
# SEPARATE NUMERIC VS CATEGORICAL FIRST
# --------------------------------------------------
num_cols = X.select_dtypes(include=["int64","float64"]).columns
cat_cols = X.select_dtypes(include=["object","category"]).columns

# --------------------------------------------------
# NUMERIC FILL
# --------------------------------------------------
X[num_cols] = X[num_cols].fillna(0)

# --------------------------------------------------
# CATEGORICAL FILL
# --------------------------------------------------
for col in cat_cols:
    X[col] = X[col].astype(str).fillna("Unknown")

# --------------------------------------------------
# NOW SPLIT NUMERIC VS CATEGORICAL
# --------------------------------------------------
num_cols = X.select_dtypes(include=["int64","float64"]).columns
cat_cols = X.select_dtypes(include="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=int
            ),
            cat_cols
        ),
        ("num", StandardScaler(), num_cols)
    ]
)

# -------------------------
# MODEL PIPELINE
# -------------------------
pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ]
)

pipeline.fit(X_train, y_train)


# -------------------------
# SAVE MODEL
# -------------------------
joblib.dump(
    pipeline,
    "models/churn_model.pkl"
)


# -------------------------
# PREDICT FULL DATA
# -------------------------
preds = pipeline.predict(X)
probs = pipeline.predict_proba(X)[:,1]

ml_df["Churn_Prediction"] = pd.Series(preds, index=ml_df.index)
ml_df["Churn_Probability"] = pd.Series(probs, index=ml_df.index)

ml_df["Churn_Prediction"] = pd.to_numeric(
    ml_df["Churn_Prediction"],
    errors="coerce"
)

ml_df["Churn_Probability"] = pd.to_numeric(
    ml_df["Churn_Probability"],
    errors="coerce"
)


# -------------------------
# KPI GRAPH
# -------------------------
sns.countplot(
    data=ml_df,
    x="Churn_Prediction"
)
plt.savefig(
    "reports/figures/churn_distribution.png"
)
plt.clf()


# -------------------------
# GENDER CHURN RATE
# -------------------------
gender_rate = (
    ml_df.groupby("gender")["Churn_Prediction"]
    .mean()
    .reset_index()
)

gender_rate["Churn_Rate_%"] = (
    gender_rate["Churn_Prediction"] * 100
)

sns.barplot(
    data=gender_rate,
    x="gender",
    y="Churn_Rate_%"
)
plt.savefig(
    "reports/figures/gender_churn.png"
)
plt.clf()


# -------------------------
# SCATTER
# -------------------------
sns.scatterplot(
    data=ml_df,
    x="MonthlyCharges",
    y="Churn_Probability"
)
plt.savefig(
    "reports/figures/spend_vs_churn.png"
)
plt.clf()


# -------------------------
# FINANCE ASSUMPTIONS
# -------------------------
raw_df["Churn_Prediction"] = ml_df["Churn_Prediction"]
raw_df["Churn_Probability"] = ml_df["Churn_Probability"]

retention_cost = 500
months_lost = 12
success_rate = 0.3
total_budget = 100000


# -------------------------
# SEGMENT ROI
# -------------------------
seg_df = segment_roi_analysis(
    ml_df,
    retention_cost,
    months_lost,
    success_rate
)

summary = executive_summary(seg_df)

# -------------------------
# SEGMENT REVENUE GRAPH
# -------------------------
sns.barplot(
    data=seg_df,
    x="Risk_Tier",
    y="Revenue_at_Risk"
)

plt.title("Revenue by Risk Segment")

plt.savefig(
    "reports/figures/revenue_by_segment.png"
)

plt.clf()

roi_tbl = segment_roi_table(seg_df)

# -------------------------
# BUILD BUNDLE
# -------------------------
bundle = build_report_bundle(
    seg_df,
    summary,
    retention_cost,
    months_lost,
    success_rate,
    total_budget
)


# -------------------------
# GENERATE REPORTS
# -------------------------
export_pdf(bundle)
export_ppt(bundle)

print("CFO Report + PPT Generated Successfully")