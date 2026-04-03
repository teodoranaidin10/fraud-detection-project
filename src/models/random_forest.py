### Baseline Random Forest - Setul de date initial, fara SMOTE

from sklearn.ensemble import RandomForestClassifier
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

### Definirea modelului
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

### Antrenare
rf_model.fit(X_train, y_train)

### Predictii pe validation
y_val_pred = rf_model.predict(X_val)
y_val_proba = rf_model.predict_proba(X_val)[:, 1]

### Predictii pe test
y_test_pred = rf_model.predict(X_test)
y_test_proba = rf_model.predict_proba(X_test)[:, 1]

### Evaluare validation si test
val_metrics = evaluate_model(y_val, y_val_pred, y_val_proba, dataset_name="Validation", model_name="Random Forest Baseline")
test_metrics = evaluate_model(y_test, y_test_pred, y_test_proba, dataset_name="Test", model_name="Random Forest Baseline")

### Tabel sumar metrici
metrics_df = pd.DataFrame([val_metrics, test_metrics])
print("\nTabel metrici:")
display(metrics_df)

# PLOT Confusion Matrix - Test
plot_confusion_matrix( y_test, y_test_pred, title="Random Forest - Confusion Matrix (Test)")

# PLOT Precision-Recall Curve
plot_pr_curve(
    y_val, y_val_proba,
    y_test, y_test_proba,
    val_metrics,
    test_metrics,
    model_name="Random Forest Baseline"
)

# PLOT ROC Curve - Validation si Test
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

plt.figure(figsize=(7, 5))
plt.plot(fpr_val, tpr_val, label=f"Validation ROC-AUC = {val_metrics['ROC-AUC']:.4f}")
plt.plot(fpr_test, tpr_test, label=f"Test ROC-AUC = {test_metrics['ROC-AUC']:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest - ROC Curve")
plt.legend()
plt.grid(True)
plt.show()


##### TRESHOLD TUNING

rf_val_thresholds, rf_best_row, rf_test_best = threshold_tuning_report(
    y_val=y_val,
    y_val_proba=y_val_proba,
    y_test=y_test,
    y_test_proba=y_test_proba,
    model_name="Random Forest Baseline",
    optimize_for="F1-score"
)

display(rf_val_thresholds.head())
plot_threshold_focus(rf_val_thresholds, model_name="Random Forest Baseline", dataset_name="Validation")
plot_confusion_matrix_threshold(
    y_test,
    y_test_proba,
    threshold=rf_best_row["Threshold"],
    model_name="Random Forest Baseline",
    dataset_name="Test",
    cmap="Blues"
)


#### Model Random Forest antrenat pe date SMOTE
rf_smote_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

rf_smote_model.fit(X_train_smote, y_train_smote)

# 4.3 Predictii RF + SMOTE pe validation
y_val_pred_smote = rf_smote_model.predict(X_val_scaled)
y_val_proba_smote = rf_smote_model.predict_proba(X_val_scaled)[:, 1]

# 4.4 Predictii RF + SMOTE pe test
y_test_pred_smote = rf_smote_model.predict(X_test_scaled)
y_test_proba_smote = rf_smote_model.predict_proba(X_test_scaled)[:, 1]


# 4.5 Evaluare RF + SMOTE
val_metrics_smote = evaluate_model(
    y_val, y_val_pred_smote, y_val_proba_smote,
    dataset_name="Validation",
    model_name="RF + SMOTE"
)

test_metrics_smote = evaluate_model(
    y_test, y_test_pred_smote, y_test_proba_smote,
    dataset_name="Test",
    model_name="RF + SMOTE"
)

# 4.7 Confusion Matrix - Test
plot_confusion_matrix(
    y_test,
    y_test_pred_smote,
    title="Random Forest + SMOTE - Confusion Matrix (Test)"
)

# PLOT PR Curve pentru Random Forest cu SMOTE
plot_pr_curve(
    y_val, y_val_proba_smote,
    y_test, y_test_proba_smote,
    val_metrics_smote,
    test_metrics_smote,
    model_name="RF + SMOTE"
)
