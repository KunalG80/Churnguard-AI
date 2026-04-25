import logging
import joblib
import pandas as pd
from src.config import MODEL_PATH

logger = logging.getLogger(__name__)


def align_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align inference DataFrame to training feature schema.
    Missing columns added as 'Unknown'; extra columns dropped.
    """
    try:
        model = joblib.load(MODEL_PATH)
        expected = list(model.named_steps["preprocess"].feature_names_in_)
    except Exception as e:
        logger.error("Could not load model schema: %s", e)
        raise

    missing = [c for c in expected if c not in df.columns]
    extra   = [c for c in df.columns  if c not in expected]

    if missing:
        logger.warning("Adding %d missing columns as 'Unknown': %s", len(missing), missing)
    if extra:
        logger.info("Dropping %d extra columns: %s", len(extra), extra)

    for col in missing:
        df[col] = "Unknown"

    return df[expected]
