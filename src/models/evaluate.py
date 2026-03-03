from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, X_test, y_test):
    """
    Evaluate classification model.
    Saves confusion matrix.
    """

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png")
    plt.close()

    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc)

    return {
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }