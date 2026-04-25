"""
ChurnGuard AI — Training Pipeline
Fixes applied:
  - Uses XGBoost (was LogisticRegression despite README saying XGBoost)
  - TreeExplainer for SHAP (was LinearExplainer)
  - Evaluates on X_test only (was predicting on full X)
  - 5-fold stratified CV added
  - PR curve + optimal threshold logged
  - Model metadata saved to JSON on every run
  - All paths via config (pathlib, cwd-safe)
  - Explicit imports (no wildcard)
  - logging replaces print()
"""
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DATA_RAW_PATH, DATA_PROCESSED_DIR, PROCESSED_CSV, TRAIN_CSV, TEST_CSV,
    MODEL_PATH, FIGURES_DIR, TARGET_COL, TEST_SIZE, RANDOM_STATE, CV_FOLDS,
)
from src.data.load_data import load_data
from src.data.preprocess import clean_data
from src.features.build_features import build_features
from src.models.train_xgboost import train_xgboost
from src.models.evaluate import evaluate_model
from src.models.shap_explain import shap_explain
from src.utils.segment_roi import segment_roi_analysis
from src.utils.executive_summary import executive_summary
from src.utils.roi_table import segment_roi_table
from src.utils.report_bundle import build_report_bundle
from src.utils.pdf_export import export_pdf
from src.utils.ppt_export import export_ppt

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Directories ───────────────────────────────────────────────────────────────
for d in [DATA_PROCESSED_DIR, MODEL_PATH.parent, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Load + clean + feature engineer ──────────────────────────────────────────
logger.info("Loading data from %s", DATA_RAW_PATH)
raw_df = load_data(str(DATA_RAW_PATH))

ml_df = clean_data(raw_df.copy())
ml_df = build_features(ml_df)
ml_df.to_csv(PROCESSED_CSV, index=False)
logger.info("Processed data saved → %s  (%d rows, %d cols)", PROCESSED_CSV, *ml_df.shape)

# ── Target encoding ───────────────────────────────────────────────────────────
ml_df[TARGET_COL] = ml_df[TARGET_COL].map({"Yes": 1, "No": 0}).fillna(ml_df[TARGET_COL])
ml_df[TARGET_COL] = pd.to_numeric(ml_df[TARGET_COL], errors="coerce").fillna(0).astype(int)

X = ml_df.drop(columns=[TARGET_COL])
y = ml_df[TARGET_COL]

# ── Type-safe column prep ─────────────────────────────────────────────────────
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

X[num_cols] = X[num_cols].fillna(0)
for col in cat_cols:
    X[col] = X[col].astype(str).fillna("Unknown").replace("", "Unknown")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Persist splits
X_train.assign(**{TARGET_COL: y_train}).to_csv(TRAIN_CSV, index=False)
X_test.assign(**{TARGET_COL: y_test}).to_csv(TEST_CSV, index=False)
logger.info("Train=%d  Test=%d", len(X_train), len(X_test))

# ── Preprocessor ──────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int), cat_cols),
        ("num", StandardScaler(), num_cols),
    ],
    remainder="drop",
)

# ── 5-fold cross-validation ───────────────────────────────────────────────────
logger.info("Running %d-fold stratified CV …", CV_FOLDS)
from src.models.train_xgboost import train_xgboost as _train_fn
from xgboost import XGBClassifier
from src.config import XGB_PARAMS

# We need a full pipeline object for CV
_pos = int((y == 1).sum()); _neg = int((y == 0).sum())
_cv_model = XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": _neg / _pos})
_cv_pipe = Pipeline([("preprocess", preprocessor), ("model", _cv_model)])

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(_cv_pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
logger.info("CV AUC: %.4f ± %.4f  (folds: %s)",
            cv_scores.mean(), cv_scores.std(),
            [f"{s:.3f}" for s in cv_scores])

# ── Train final model on full training set ────────────────────────────────────
logger.info("Training final XGBoost model …")

# Fit preprocessor on train, transform both splits
preprocessor.fit(X_train)
X_train_t = preprocessor.transform(X_train)
X_test_t  = preprocessor.transform(X_test)

xgb_model = train_xgboost(X_train_t, y_train, X_test_t, y_test)

# Wrap in sklearn Pipeline so app.py / schema_handler can use .named_steps
pipeline = Pipeline([("preprocess", preprocessor), ("model", xgb_model)])
# Re-fit pipeline (needed so feature_names_in_ is set on preprocessor)
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, MODEL_PATH)
logger.info("Model saved → %s", MODEL_PATH)

# ── Evaluate on TEST SET ONLY ──────────────────────────────────────────────────
logger.info("Evaluating on held-out test set …")
metrics = evaluate_model(xgb_model, X_test_t, y_test)
logger.info("Test AUC: %.4f | Optimal threshold: %.3f",
            metrics["roc_auc"], metrics["optimal_threshold"])

# ── SHAP explanations ─────────────────────────────────────────────────────────
logger.info("Generating SHAP explanations …")
feature_names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
shap_explain(xgb_model, X_test_t, feature_names)

# ── Visualisations ────────────────────────────────────────────────────────────
ml_df["Churn_Prediction"] = pipeline.predict(X)
ml_df["Churn_Probability"] = pipeline.predict_proba(X)[:, 1]

fig, ax = plt.subplots(); sns.countplot(data=ml_df, x="Churn_Prediction", ax=ax)
fig.savefig(FIGURES_DIR / "churn_distribution.png"); plt.close(fig)

gender_rate = ml_df.groupby("gender")["Churn_Prediction"].mean().reset_index()
gender_rate["Churn_Rate_%"] = gender_rate["Churn_Prediction"] * 100
fig, ax = plt.subplots(); sns.barplot(data=gender_rate, x="gender", y="Churn_Rate_%", ax=ax)
fig.savefig(FIGURES_DIR / "gender_churn.png"); plt.close(fig)

fig, ax = plt.subplots()
sns.scatterplot(data=ml_df, x="MonthlyCharges", y="Churn_Probability", ax=ax, alpha=0.4)
fig.savefig(FIGURES_DIR / "spend_vs_churn.png"); plt.close(fig)

# ── ROI analysis (on churners only) ──────────────────────────────────────────
retention_cost = 500; months_lost = 12; success_rate = 0.3; total_budget = 100_000

seg_df  = segment_roi_analysis(ml_df, retention_cost, months_lost, success_rate)
summary = executive_summary(seg_df)
roi_tbl = segment_roi_table(seg_df)

fig, ax = plt.subplots()
sns.barplot(data=seg_df, x="Risk_Tier", y="Revenue_at_Risk", ax=ax)
ax.set_title("Revenue by Risk Segment")
fig.savefig(FIGURES_DIR / "revenue_by_segment.png"); plt.close(fig)

bundle = build_report_bundle(seg_df, summary, retention_cost, months_lost, success_rate, total_budget)
export_pdf(bundle)
export_ppt(bundle)

logger.info("Pipeline complete — CV AUC %.4f | Test AUC %.4f", cv_scores.mean(), metrics["roc_auc"])
logger.info("Reports generated in %s", str(MODEL_PATH.parent.parent / "reports"))
