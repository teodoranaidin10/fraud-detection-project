import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def apply_threshold(y_proba, threshold=0.5):
    """
    Aplică un threshold pe probabilități și returnează etichete binare.
    """
    return (y_proba >= threshold).astype(int)


def evaluate_at_threshold(y_true, y_proba, threshold=0.5):
    """
    Calculează metricile pentru un threshold dat.
    """
    y_pred = apply_threshold(y_proba, threshold)

    metrics = {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics


def threshold_sweep(y_true, y_proba, thresholds=None):
    """
    Evaluează modelul pe o listă de threshold-uri.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    results = []
    for thr in thresholds:
        row = evaluate_at_threshold(y_true, y_proba, threshold=thr)
        results.append(row)

    return pd.DataFrame(results)


def get_best_threshold(results_df, metric="F1-score"):
    """
    Selectează threshold-ul optim în funcție de metrica aleasă.
    """
    best_idx = results_df[metric].idxmax()
    best_row = results_df.loc[best_idx]
    return best_row

def plot_threshold_metrics(results_df, model_name="Model", dataset_name="Validation"):
    """
    Plotează Precision / Recall / F1-score vs threshold.
    """
    plt.figure(figsize=(9, 6))
    plt.plot(results_df["Threshold"], results_df["Precision"], label="Precision")
    plt.plot(results_df["Threshold"], results_df["Recall"], label="Recall")
    plt.plot(results_df["Threshold"], results_df["F1-score"], label="F1-score")

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"{model_name} - Precision / Recall / F1 vs Threshold ({dataset_name})")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_confusion_matrix_threshold(
    y_true,
    y_proba,
    threshold=0.5,
    model_name="Model",
    dataset_name="Test",
    cmap="Blues"
):
    """
    Plotează confusion matrix pentru un threshold custom.
    """
    y_pred = apply_threshold(y_proba, threshold)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Fraud"]
    )
    disp.plot(cmap=cmap, values_format="d")
    plt.title(f"{model_name} - Confusion Matrix ({dataset_name}, thr={threshold:.2f})")
    plt.grid(False)
    plt.show()


def threshold_tuning_report(
    y_val,
    y_val_proba,
    y_test,
    y_test_proba,
    model_name="Model",
    optimize_for="F1-score"
):
    """
    - caută threshold optim pe validation
    - îl aplică pe test
    - returnează rezultatele
    """
    val_results = threshold_sweep(y_val, y_val_proba)

    best_row = get_best_threshold(val_results, metric=optimize_for)
    best_threshold = best_row["Threshold"]

    print(f"\n===== {model_name} | Threshold tuning =====")
    print(f"Metrică de optimizare: {optimize_for}")
    print(f"Best threshold pe Validation: {best_threshold:.2f}")

    print("\nMetrici pe Validation la threshold optim:")
    print(best_row)

    test_best_metrics = evaluate_at_threshold(
        y_test,
        y_test_proba,
        threshold=best_threshold
    )

    print("\nMetrici pe Test la threshold optim:")
    for k, v in test_best_metrics.items():
        if k == "Threshold":
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v:.6f}")

    return val_results, best_row, test_best_metrics