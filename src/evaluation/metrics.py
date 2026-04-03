# Functie generica de evaluare
def evaluate_model(y_true, y_pred, y_proba, dataset_name="Dataset", model_name="Model"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    print(f"\n===== {model_name} | Evaluare pentru {dataset_name} =====")
    print(f"Accuracy  : {acc:.6f}")
    print(f"Precision : {prec:.6f}")
    print(f"Recall    : {rec:.6f}")
    print(f"F1-score  : {f1:.6f}")
    print(f"ROC-AUC   : {roc_auc:.6f}")
    print(f"PR-AUC    : {pr_auc:.6f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=6, zero_division=0))

    return {
        "Model": model_name,
        "Dataset": dataset_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    }
