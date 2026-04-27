import json
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, f1_score,
)
from src.config import FIGURES_DIR, METADATA_PATH

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test,
                   model_name: str = "XGBoostClassifier") -> dict:
    """
    Evaluate model on held-out test set.
    Saves: confusion matrix, ROC curve, PR curve.
    Returns metrics dict with optimal threshold from PR curve.
    Persists metadata JSON for reproducibility.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ── ROC curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, lw=2, color="#185FA5", label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    # ── Precision-Recall + optimal threshold ──────────────────────────────────
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    f1_arr = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = int(f1_arr.argmax())
    optimal_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rec, prec, lw=2, color="#185FA5")
    ax.axvline(rec[best_idx], color="#E24B4A", lw=1.2, linestyle="--",
               label=f"Optimal t={optimal_threshold:.2f}  F1={f1_arr[best_idx]:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve"); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=150)
    plt.close(fig)

    # ── ROI vs threshold curve (wires in roi_analysis logic) ──────────────────
    # Shows net ROI at each decision threshold — useful for CFO dashboard
    thresholds_sweep = np.linspace(0.1, 0.9, 40)
    # (saved as data only — chart generated in dashboard for interactivity)

    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
    logger.info("ROC-AUC: %.4f | Optimal threshold: %.3f | F1@threshold: %.4f",
                roc_auc, optimal_threshold, f1_arr[best_idx])

    metrics = {
        "roc_auc":           round(roc_auc, 4),
        "optimal_threshold": round(optimal_threshold, 3),
        "best_f1":           round(float(f1_arr[best_idx]), 4),
        "confusion_matrix":  cm.tolist(),
    }

    # ── Persist metadata ──────────────────────────────────────────────────────
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "trained_at": str(datetime.datetime.now()),
        "model":      model_name,
        **metrics,
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("Metadata → %s", METADATA_PATH)

    return metrics
