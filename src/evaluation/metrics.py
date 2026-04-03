import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


def compute_classification_metrics(y_true, y_pred, y_proba):
    """
    Calculează metricile principale pentru clasificare binară.
    Returnează un dicționar.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "PR-AUC": average_precision_score(y_true, y_proba)
    }
    return metrics


def print_metrics(metrics, dataset_name="Dataset", model_name="Model"):
    """
    Afișează frumos metricile în consolă.
    """
    print(f"\n===== {model_name} | {dataset_name} =====")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.6f}")


def evaluate_model(y_true, y_pred, y_proba, dataset_name="Dataset", model_name="Model"):
    """
    Calculează și afișează metricile.
    Returnează un dicționar cu rezultatele.
    """
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    print_metrics(metrics, dataset_name=dataset_name, model_name=model_name)
    return metrics


def metrics_to_dataframe(results_dict):
    """
    Transformă un dicționar de forma:
    {
        "RF Baseline - Validation": {...},
        "RF Baseline - Test": {...},
        ...
    }
    într-un DataFrame pandas.
    """
    df = pd.DataFrame(results_dict).T
    return df


def compare_models(results_dict, sort_by=None, ascending=False):
    """
    Creează un DataFrame comparativ pentru mai multe modele.

    Parameters:
    - results_dict: dict cu metrici
    - sort_by: numele metricii după care se sortează (opțional)
    - ascending: sensul sortării
    """
    df = metrics_to_dataframe(results_dict)

    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)

    return df