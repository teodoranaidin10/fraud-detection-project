# Afisare Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", cmap="PuRd"):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Frauda"]
    )
    disp.plot(cmap=cmap, values_format="d")
    plt.title(title)
    plt.grid(False)
    plt.show()


# Afisare Precision Recall Curve
def plot_pr_curve(y_val, y_val_proba, y_test, y_test_proba,
                  val_metrics, test_metrics,
                  model_name="Model"):
    precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_proba)
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_proba)

    plt.figure(figsize=(7, 5))

    plt.plot(
        recall_val, precision_val,
        label=f"Validation PR-AUC = {val_metrics['PR-AUC']:.4f}"
    )

    plt.plot(
        recall_test, precision_test,
        label=f"Test PR-AUC = {test_metrics['PR-AUC']:.4f}"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} - Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def plot_roc_curve(
    y_val, y_val_proba,
    y_test, y_test_proba,
    val_metrics,
    test_metrics,
    model_name="Model"
):
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr_val, tpr_val, label=f"Validation ROC-AUC = {val_metrics['ROC-AUC']:.4f}")
    plt.plot(fpr_test, tpr_test, label=f"Test ROC-AUC = {test_metrics['ROC-AUC']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


# COMPARATIE PR intre modele
def plot_pr_comparison(y_true, proba_dict, title="Precision-Recall Comparison"):
    """
    proba_dict = {
        "RF baseline": y_proba_rf,
        "RF + SMOTE": y_proba_rf_smote,
        "XGBoost": y_proba_xgb
    }
    """
    plt.figure(figsize=(7, 5))

    for model_name, y_proba in proba_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        plt.plot(recall, precision, label=model_name)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()