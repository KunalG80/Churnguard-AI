import shap
import matplotlib.pyplot as plt

def shap_xgboost(model, X):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    plt.tight_layout()
    plt.savefig("reports/figures/shap_xgb.png")
    plt.close()