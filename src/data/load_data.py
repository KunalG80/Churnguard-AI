import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

BINARY_MAP = {
    "yes": "Yes", "no": "No",
    "true": "Yes", "false": "No",
    "1": "Yes", "0": "No",
    "nan": "No", " ": "No",
}

BINARY_COLS = [
    "Partner", "Dependents", "PhoneService", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
]


def load_data(path: str | Path) -> pd.DataFrame:
    """
    Load CSV with encoding fallback.
    Normalises binary Yes/No columns immediately on load.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 failed, retrying with latin-1: %s", path)
        df = pd.read_csv(path, encoding="latin-1")

    logger.info("Loaded %d rows × %d cols from %s", *df.shape, path.name)

    # Normalise binary columns (absorbs categorical_cleaner.py logic)
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.strip().str.lower()
                .map(lambda v: BINARY_MAP.get(v, v))
            )

    return df
