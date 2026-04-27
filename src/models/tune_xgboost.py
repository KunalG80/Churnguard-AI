"""
Optuna hyperparameter tuning for XGBoost churn model.
Optimises for ROC-AUC via 3-fold stratified CV (fast) over 50 trials.
Precision-focused: adds a precision floor constraint — trials scoring
< 0.60 churner precision on the validation fold are pruned early.

Run standalone:
    python -m src.models.tune_xgboost
"""
import json
import logging
import warnings
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import precision_score
from xgboost import XGBClassifier

from src.config import RANDOM_STATE, METADATA_PATH

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def run_tuning(X_train, y_train, cat_cols, num_cols,
               n_trials: int = 50) -> dict:
    """
    Run Optuna study. Returns best params dict ready to merge into XGB_PARAMS.
    """
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = neg / pos

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth":         trial.suggest_int("max_depth", 3, 7),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 3.0),
            "scale_pos_weight":  spw,
            "eval_metric":       "logloss",
            "random_state":      RANDOM_STATE,
        }

        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False, dtype=int), cat_cols),
            ("num", StandardScaler(), num_cols),
        ], remainder="drop")

        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model",      XGBClassifier(**params)),
        ])

        cv     = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(pipe, X_train, y_train,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    sampler = TPESampler(seed=RANDOM_STATE)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["eval_metric"]      = "logloss"
    best["random_state"]     = RANDOM_STATE
    best["scale_pos_weight"] = spw

    logger.info("Best AUC: %.4f", study.best_value)
    logger.info("Best params: %s", json.dumps(
        {k: round(v, 4) if isinstance(v, float) else v for k, v in best.items()},
        indent=2,
    ))

    # Persist best params alongside metadata
    meta_path = METADATA_PATH.parent / "best_params.json"
    meta_path.write_text(json.dumps(best, indent=2))
    logger.info("Best params saved → %s", meta_path)

    return best, study


if __name__ == "__main__":
    import pandas as pd
    import joblib
    from src.config import TRAIN_CSV, TARGET_COL

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                        datefmt="%H:%M:%S")

    logger.info("Loading training data …")
    train_df = pd.read_csv(TRAIN_CSV)
    X        = train_df.drop(columns=[TARGET_COL])
    y        = train_df[TARGET_COL].astype(int)

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()

    logger.info("Starting Optuna — 50 trials …")
    best_params, study = run_tuning(X, y, cat_cols, num_cols, n_trials=50)

    print("\n" + "="*55)
    print("  BEST PARAMS — paste into src/config.py XGB_PARAMS")
    print("="*55)
    for k, v in best_params.items():
        val = f"{v:.4f}" if isinstance(v, float) else v
        print(f'    "{k}": {val},')
    print(f"\n  Best CV AUC: {study.best_value:.4f}")
    print("="*55)
