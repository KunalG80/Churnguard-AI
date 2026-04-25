import numpy as np
from xgboost import XGBClassifier
from src.config import XGB_PARAMS


def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    """
    Train XGBoost with class-imbalance correction.
    Accepts optional validation set for early stopping.
    """
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}

    model = XGBClassifier(**params)

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    return model
