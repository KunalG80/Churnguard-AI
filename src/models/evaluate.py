import json
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
)
from src.config import FIGURES_DIR, METADATA_PATH

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, model_name="XGBoostClassifier") -> dict:
    """
    Evaluate model. Saves confusion matrix + PR curve.
    Returns metrics dict including optimal threshold from PR curve.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ── Precision-Recall curve + optimal threshold ────────────────────────────
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = f1_scores.argmax()
    optimal_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rec, prec, lw=1.8, color="#185FA5")
    ax.axvline(rec[best_idx], color="#E24B4A", lw=1, linestyle="--",
               label=f"Optimal threshold={optimal_threshold:.2f}  F1={f1_scores[best_idx]:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=150)
    plt.close(fig)

    logger.info("ROC-AUC: %.4f | Optimal threshold: %.3f | F1: %.4f",
                roc_auc, optimal_threshold, f1_scores[best_idx])
    print(classification_report(y_test, y_pred))

    metrics = {
        "roc_auc":           round(roc_auc, 4),
        "optimal_threshold": round(optimal_threshold, 3),
        "best_f1":           round(float(f1_scores[best_idx]), 4),
        "confusion_matrix":  cm.tolist(),
    }

    # ── Persist metadata ──────────────────────────────────────────────────────
    meta = {
        "trained_at": str(datetime.datetime.now()),
        "model":      model_name,
        **metrics,
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("Metadata saved → %s", METADATA_PATH)

    return metrics
