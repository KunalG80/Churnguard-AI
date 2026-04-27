"""
Threshold tuning utility.
Run: python -m src.models.threshold_tuner
"""
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from src.config import MODEL_PATH, DATA_RAW_PATH, TARGET_COL
from src.data.load_data import load_data
from src.data.preprocess import clean_data
from src.features.build_features import build_features
from sklearn.model_selection import train_test_split
from src.config import TEST_SIZE, RANDOM_STATE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s — %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RETENTION_COST = 100    # email/SMS/voucher campaign — not a personal visit
MONTHS_LOST    = 12
SUCCESS_RATE   = 0.30


def threshold_table(y_true, y_prob, avg_monthly: float = 65.0) -> pd.DataFrame:
    rows = []
    for t in np.arange(0.30, 0.91, 0.05):
        y_pred   = (y_prob >= t).astype(int)
        targeted = int(y_pred.sum())
        if targeted == 0:
            continue
        tp    = int(((y_pred == 1) & (y_true == 1)).sum())
        prec  = precision_score(y_true, y_pred, zero_division=0)
        rec   = recall_score(y_true, y_pred, zero_division=0)
        f1    = f1_score(y_true, y_pred, zero_division=0)
        spend = targeted * RETENTION_COST
        saved = tp * avg_monthly * MONTHS_LOST * SUCCESS_RATE
        net   = saved - spend
        rows.append({
            "Threshold":     round(t, 2),
            "Targeted":      targeted,
            "True Churners": tp,
            "Precision %":   round(prec * 100, 1),
            "Recall %":      round(rec  * 100, 1),
            "F1":            round(f1, 3),
            "Spend ₹":       f"₹{spend:,.0f}",
            "Net Value ₹":   f"₹{net:,.0f}",
            "Verdict":       "✅ PROFIT" if net > 0 else "❌ LOSS",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logger.info("Loading + preprocessing raw data (same path as main.py) …")

    # ── Reproduce exact same preprocessing as main.py ─────────────────────────
    raw   = load_data(str(DATA_RAW_PATH))
    ml_df = build_features(clean_data(raw.copy()))

    ml_df[TARGET_COL] = (
        ml_df[TARGET_COL].map({"Yes": 1, "No": 0})
        .fillna(ml_df[TARGET_COL])
    )
    ml_df[TARGET_COL] = pd.to_numeric(
        ml_df[TARGET_COL], errors="coerce"
    ).fillna(0).astype(int)

    X = ml_df.drop(columns=[TARGET_COL])
    y = ml_df[TARGET_COL]

    num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
    X[num_cols] = X[num_cols].fillna(0)
    for c in cat_cols:
        X[c] = X[c].astype(str).replace("", "Unknown").fillna("Unknown")

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    logger.info("Loading model …")
    pipe       = joblib.load(MODEL_PATH)
    avg_monthly = float(ml_df["MonthlyCharges"].mean()) if "MonthlyCharges" in ml_df.columns else 65.0
    logger.info("Avg MonthlyCharges from data: %.2f", avg_monthly)
    y_prob     = pipe.predict_proba(X_test)[:, 1]
    y_true     = y_test.values

    df = threshold_table(y_true, y_prob, avg_monthly=avg_monthly)

    print("\n" + "=" * 95)
    print("  THRESHOLD OPERATING POINT TABLE")
    print("=" * 95)
    print(df.to_string(index=False))
    print("=" * 95)

    best_f1   = df.loc[df["F1"].idxmax()]
    best_prec = df.loc[df["Precision %"].idxmax()]
    net_vals  = df["Net Value ₹"].str.replace(r"[₹,]","",regex=True).astype(float)
    best_net  = df.loc[net_vals.idxmax()]

    print(f"\n  Best F1        → threshold {best_f1['Threshold']}  "
          f"Prec={best_f1['Precision %']}%  Rec={best_f1['Recall %']}%  F1={best_f1['F1']}")
    print(f"  Best Precision → threshold {best_prec['Threshold']}  "
          f"Prec={best_prec['Precision %']}%  Rec={best_prec['Recall %']}%  F1={best_prec['F1']}")
    print(f"  Best Net ROI   → threshold {best_net['Threshold']}  "
          f"Net={best_net['Net Value ₹']}  Prec={best_net['Precision %']}%")
    print("\n  → Set chosen threshold in src/config.py: THRESHOLD = x.xx\n")
