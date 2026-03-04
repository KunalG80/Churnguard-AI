import shap
import matplotlib.pyplot as plt

def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

def shap_explain(model, preprocessor, X):

    sample = X.sample(min(200, len(X)))

    X_transformed = preprocessor.transform(sample)
    X_df = to_dense(X_transformed)

    explainer = shap.LinearExplainer(model, X_df)
    shap_values = explainer(X_df)

    shap.summary_plot(
        shap_values.values,
        X_df,
        show=False
    )

    plt.tight_layout()
    plt.savefig("reports/figures/shap_summary.png")
    plt.close()