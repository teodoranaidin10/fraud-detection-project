# 6. XGBoost baseline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)


# XGBoost baseline
# Folosim datele originale + scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

xgb_baseline = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_baseline.fit(X_train, y_train)


# 5.3 Predictii XGBoost baseline
y_val_pred_xgb = xgb_baseline.predict(X_val)
y_val_proba_xgb = xgb_baseline.predict_proba(X_val)[:, 1]

y_test_pred_xgb = xgb_baseline.predict(X_test)
y_test_proba_xgb = xgb_baseline.predict_proba(X_test)[:, 1]


### Evaluare XGBoost baseline
val_metrics_xgb = evaluate_model(
    y_val, y_val_pred_xgb, y_val_proba_xgb,
    dataset_name="Validation",
    model_name="XGBoost Baseline"
)

test_metrics_xgb = evaluate_model(
    y_test, y_test_pred_xgb, y_test_proba_xgb,
    dataset_name="Test",
    model_name="XGBoost Baseline"
)

### Confusion Matrix - XGBoost baseline
plot_confusion_matrix(
    y_test,
    y_test_pred_xgb,
    title="XGBoost Baseline - Confusion Matrix (Test)"
)

# PLOT PR Curve for XGB Baseline
plot_pr_curve(
    y_val, y_val_proba_xgb,
    y_test, y_test_proba_xgb,
    val_metrics_xgb,
    test_metrics_xgb,
    model_name="XGBoost Baseline"
)


### Curbe ROC si PR - XGBoost baseline
fpr_val_xgb, tpr_val_xgb, _ = roc_curve(y_val, y_val_proba_xgb)
fpr_test_xgb, tpr_test_xgb, _ = roc_curve(y_test, y_test_proba_xgb)

plt.figure(figsize=(7, 5))
plt.plot(fpr_val_xgb, tpr_val_xgb, label=f"Validation ROC-AUC = {val_metrics_xgb['ROC-AUC']:.4f}")
plt.plot(fpr_test_xgb, tpr_test_xgb, label=f"Test ROC-AUC = {test_metrics_xgb['ROC-AUC']:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("XGBoost Baseline - ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

##### TRESHOLD TUNING

xgb_val_thresholds, xgb_best_row, xgb_test_best = threshold_tuning_report(
    y_val=y_val,
    y_val_proba=y_val_proba_xgb,
    y_test=y_test,
    y_test_proba=y_test_proba_xgb,
    model_name="XGBoost Baseline",
    optimize_for="F1-score"
)

display(xgb_val_thresholds.head())
plot_threshold_focus(xgb_val_thresholds, model_name="XGBoost Baseline", dataset_name="Validation")
plot_confusion_matrix_threshold(
    y_test,
    y_test_proba_xgb,
    threshold=xgb_best_row["Threshold"],
    model_name="XGBoost Baseline",
    dataset_name="Test",
    cmap="Purples"
)

# ------------------------------------------------
# 7. XGBoost + SMOTE
# IMPORTANT:
# X_train_smote este construit din X_train_scaled
# deci la predictie folosim X_val_scaled / X_test_scaled
# ------------------------------------------------
xgb_smote = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_smote.fit(X_train_smote, y_train_smote)

# 5.9 Predictii XGBoost + SMOTE
y_val_pred_xgb_smote = xgb_smote.predict(X_val_scaled)
y_val_proba_xgb_smote = xgb_smote.predict_proba(X_val_scaled)[:, 1]

y_test_pred_xgb_smote = xgb_smote.predict(X_test_scaled)
y_test_proba_xgb_smote = xgb_smote.predict_proba(X_test_scaled)[:, 1]

# 5.10 Evaluare XGBoost + SMOTE
val_metrics_xgb_smote = evaluate_model(
    y_val, y_val_pred_xgb_smote, y_val_proba_xgb_smote,
    dataset_name="Validation",
    model_name="XGBoost + SMOTE"
)

test_metrics_xgb_smote = evaluate_model(
    y_test, y_test_pred_xgb_smote, y_test_proba_xgb_smote,
    dataset_name="Test",
    model_name="XGBoost + SMOTE"
)

# 5.11 Confusion Matrix - XGBoost + SMOTE
plot_confusion_matrix(
    y_test,
    y_test_pred_xgb_smote,
    title="XGBoost + SMOTE - Confusion Matrix (Test)"
)

# PLOT PR Curve for XGB SMOTE
plot_pr_curve(
    y_val, y_val_proba_xgb_smote,
    y_test, y_test_proba_xgb_smote,
    val_metrics_xgb_smote,
    test_metrics_xgb_smote,
    model_name="XGBoost Baseline"
)
