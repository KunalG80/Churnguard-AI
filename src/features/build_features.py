import pandas as pd
from src.config import TENURE_BINS, TENURE_LABELS, SERVICE_COLS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering layer.
    BUG FIX: uses shared TENURE_LABELS constant (was mismatched vs preprocess.py).
    NEW: high-signal telecom features added.
    """
    df = df.copy()

    # ── tenure_group (consistent labels via shared constant) ─────────────────
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=TENURE_BINS,
            labels=TENURE_LABELS,
            include_lowest=True,
        ).astype(str)

    # ── NEW: service count ────────────────────────────────────────────────────
    present = [c for c in SERVICE_COLS if c in df.columns]
    if present:
        df["service_count"] = (df[present] == "Yes").sum(axis=1).astype(float)

    # ── NEW: charge per active service ────────────────────────────────────────
    if "MonthlyCharges" in df.columns and "service_count" in df.columns:
        df["charge_per_service"] = (
            df["MonthlyCharges"] / (df["service_count"] + 1)
        )

    # ── NEW: average monthly spend from total charges ─────────────────────────
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["avg_monthly_spend"] = (
            df["TotalCharges"] / (df["tenure"] + 1)
        )

    # ── NEW: is new customer (tenure ≤ 3 months — highest churn risk window) ──
    if "tenure" in df.columns:
        df["is_new_customer"] = (df["tenure"] <= 3).astype(int)

    return df
