import logging
import pandas as pd
from src.config import TENURE_BINS, TENURE_LABELS, SERVICE_COLS

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering layer.

    Fixes vs original:
      - TENURE_LABELS from config (was "0-1yr" etc. — mismatched with preprocess.py)
      - 4 new high-signal telecom features added
    """
    df = df.copy()

    # ── tenure_group (shared labels with preprocess.py via config) ────────────
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=TENURE_BINS,
            labels=TENURE_LABELS,
            include_lowest=True,
        ).astype(str)

    # ── service_count: how many add-on services active ────────────────────────
    # Note: MultipleLines/InternetService use "No phone/internet service" — these
    # correctly evaluate to 0 since they don't equal "Yes".
    present = [c for c in SERVICE_COLS if c in df.columns]
    if present:
        df["service_count"] = (df[present] == "Yes").sum(axis=1).astype(float)

    # ── charge per active service ─────────────────────────────────────────────
    if "MonthlyCharges" in df.columns and "service_count" in df.columns:
        df["charge_per_service"] = df["MonthlyCharges"] / (df["service_count"] + 1)

    # ── average monthly spend derived from total charges ──────────────────────
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # ── new customer flag: tenure ≤ 3 months (highest churn-risk window) ──────
    if "tenure" in df.columns:
        df["is_new_customer"] = (df["tenure"] <= 3).astype(int)

    logger.debug("build_features: output shape %s", df.shape)
    return df
