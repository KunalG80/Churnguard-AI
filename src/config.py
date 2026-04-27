from pathlib import Path

# ── Project root (works regardless of cwd) ───────────────────────────────────
ROOT = Path(__file__).parent.parent

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_RAW_PATH      = ROOT / "data" / "raw" / "Churn.csv"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_CSV      = DATA_PROCESSED_DIR / "processed.csv"
TRAIN_CSV          = DATA_PROCESSED_DIR / "train.csv"
TEST_CSV           = DATA_PROCESSED_DIR / "test.csv"
MODEL_PATH         = ROOT / "models" / "churn_model.pkl"
METADATA_PATH      = ROOT / "models" / "metadata.json"
SCHEMA_PATH        = ROOT / "models" / "training_schema.json"
REPORTS_DIR        = ROOT / "reports"
FIGURES_DIR        = REPORTS_DIR / "figures"
FONT_PATH          = ROOT / "assets" / "fonts" / "DejaVuSans.ttf"

# ── Model settings ────────────────────────────────────────────────────────────
TARGET_COL   = "Churn"
TEST_SIZE    = 0.2
RANDOM_STATE = 42
CV_FOLDS     = 5
THRESHOLD    = 0.35

# ── Feature engineering ───────────────────────────────────────────────────────
TENURE_LABELS = ["0-12m", "12-24m", "24-48m", "48-60m", "60+m"]
TENURE_BINS   = [0, 12, 24, 48, 60, 100]

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":      300,
    "max_depth":         4,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  3,
    "gamma":             0.1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "eval_metric":  "logloss",
    "random_state": RANDOM_STATE,
    # use_label_encoder removed — deprecated in XGBoost ≥ 1.6, causes UserWarning
}

# ── Report output paths ───────────────────────────────────────────────────────
PDF_PATH    = REPORTS_DIR / "churnguard_report.pdf"
PPT_PATH    = REPORTS_DIR / "churnguard_report.pptx"
BUNDLE_PATH = REPORTS_DIR / "Retention_Strategy_Pack.zip"
