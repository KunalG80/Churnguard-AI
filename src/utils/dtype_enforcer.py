import logging
import pandas as pd

logger = logging.getLogger(__name__)


def enforce_training_dtypes(uploaded: pd.DataFrame,
                             trained: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce uploaded DataFrame columns to match training dtypes.
    Fixed: regex now preserves negative sign (was [^\\d\\.], now [^\\d\\.\\-])
    so negative-valued numeric columns are not corrupted.
    """
    uploaded = uploaded.copy()

    for col in trained.columns:
        if col not in uploaded.columns:
            continue

        if trained[col].dtype in ["float64", "int64"]:
            uploaded[col] = (
                uploaded[col]
                .astype(str)
                .str.replace(r"[^\d.\-]", "", regex=True)   # ← fix: keep minus sign
            )
            uploaded[col] = pd.to_numeric(uploaded[col], errors="coerce")
        else:
            uploaded[col] = uploaded[col].astype(str)

    return uploaded
