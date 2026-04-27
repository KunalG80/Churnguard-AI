import logging
from xgboost import XGBClassifier
from src.config import XGB_PARAMS

logger = logging.getLogger(__name__)


def train_xgboost(X_train, y_train, X_val=None, y_val=None) -> XGBClassifier:
    """
    Train XGBoost with class-imbalance correction via scale_pos_weight.
    Early stopping removed for broad version compatibility.
    """
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    logger.info("Class ratio neg/pos=%.2f → scale_pos_weight=%.2f",
                scale_pos_weight, scale_pos_weight)

    # Drop deprecated param before passing to constructor
    params = {k: v for k, v in XGB_PARAMS.items() if k != "use_label_encoder"}
    params["scale_pos_weight"] = scale_pos_weight

    model = XGBClassifier(**params)

    # Plain fit — works across all XGBoost versions
    model.fit(X_train, y_train)
    logger.info("Trained for %d estimators.", params["n_estimators"])

    return model
