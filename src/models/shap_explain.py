import logging
import numpy as np
import matplotlib.pyplot as plt
import shap
from src.config import FIGURES_DIR

logger = logging.getLogger(__name__)


def shap_explain(model, X_transformed, feature_names: list,
                 n_sample: int = 300):
    """
    SHAP TreeExplainer for XGBoost (fixed from LinearExplainer).
    Saves summary plot with real human-readable feature names.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_transformed),
                     min(n_sample, len(X_transformed)), replace=False)
    X_sample = X_transformed[idx]

    # Clean up OHE prefix noise: "cat__Contract_Month-to-month" → "Contract: Month-to-month"
    clean_names = []
    for name in feature_names:
        name = name.replace("cat__", "").replace("num__", "")
        if "_" in name:
            parts = name.split("_", 1)
            name  = f"{parts[0]}: {parts[1]}"
        clean_names.append(name)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=clean_names,
        show=False, max_display=20,
    )
    plt.title("Top Churn Drivers (SHAP)", fontsize=13, pad=12)
    plt.tight_layout()
    # Save with both names so ppt_export finds it either way
    for fname in ("shap_summary.png", "shap_importance.png"):
        plt.savefig(FIGURES_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary saved → %s", FIGURES_DIR / "shap_summary.png")
