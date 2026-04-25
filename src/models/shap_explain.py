import shap
import logging
import matplotlib.pyplot as plt
from src.config import FIGURES_DIR

logger = logging.getLogger(__name__)


def shap_explain(model, X_transformed, feature_names: list, n_sample: int = 300):
    """
    BUG FIX: uses TreeExplainer (correct for XGBoost, was LinearExplainer).
    Saves SHAP summary plot with real feature names.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Sample for speed
    import numpy as np
    idx = np.random.choice(len(X_transformed), min(n_sample, len(X_transformed)), replace=False)
    X_sample = X_transformed[idx]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary saved → %s", FIGURES_DIR / "shap_summary.png")
