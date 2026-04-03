from src.preprocessing.data_setup import prepare_data
from src.models.random_forest import (train_rf_baseline, train_rf_smote, predict_rf)
from src.models.xgboost import (train_xgb_baseline, train_xgb_smote, predict_xgb)
from src.evaluation.metrics import evaluate_model
from src.evaluation.plots import (plot_confusion_matrix, plot_pr_curve)
from src.models.threshold import (threshold_tuning_report, plot_threshold_metrics, plot_confusion_matrix_threshold)


def main():
    print("=== Fraud Detection Pipeline ===")

    # ------------------------------------------------
    # 1. DATA PREPARATION
    # ------------------------------------------------
    data = prepare_data(
        csv_path="data/raw/creditcard.csv",
        use_smote=True,
        return_numpy=True,
        verbose=True
    )

    # date originale
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    # date scalate
    X_train_scaled = data["X_train_scaled"]
    X_val_scaled = data["X_val_scaled"]
    X_test_scaled = data["X_test_scaled"]

    # date SMOTE
    X_train_smote = data["X_train_smote"]
    y_train_smote = data["y_train_smote"]

    # etichete
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    # ------------------------------------------------
    # 2. RANDOM FOREST BASELINE
    # ------------------------------------------------
    print("\n=== Random Forest Baseline ===")

    rf_baseline = train_rf_baseline(X_train, y_train)

    y_val_pred_rf, y_val_proba_rf = predict_rf(rf_baseline, X_val)
    y_test_pred_rf, y_test_proba_rf = predict_rf(rf_baseline, X_test)

    val_metrics_rf = evaluate_model(
        y_val, y_val_pred_rf, y_val_proba_rf,
        dataset_name="Validation",
        model_name="Random Forest Baseline"
    )

    test_metrics_rf = evaluate_model(
        y_test, y_test_pred_rf, y_test_proba_rf,
        dataset_name="Test",
        model_name="Random Forest Baseline"
    )

    plot_confusion_matrix(
        y_test,
        y_test_pred_rf,
        title="Random Forest Baseline - Confusion Matrix (Test)"
    )

    plot_pr_curve(
        y_val, y_val_proba_rf,
        y_test, y_test_proba_rf,
        val_metrics_rf,
        test_metrics_rf,
        model_name="Random Forest Baseline"
    )

    rf_val_thresholds, rf_best_row, rf_test_best = threshold_tuning_report(
        y_val=y_val,
        y_val_proba=y_val_proba_rf,
        y_test=y_test,
        y_test_proba=y_test_proba_rf,
        model_name="Random Forest Baseline",
        optimize_for="F1-score"
    )

    plot_threshold_metrics(
        rf_val_thresholds,
        model_name="Random Forest Baseline",
        dataset_name="Validation"
    )

    plot_confusion_matrix_threshold(
        y_test,
        y_test_proba_rf,
        threshold=rf_best_row["Threshold"],
        model_name="Random Forest Baseline",
        dataset_name="Test",
        cmap="Blues"
    )

    # ------------------------------------------------
    # 3. RANDOM FOREST + SMOTE
    # ------------------------------------------------
    print("\n=== Random Forest + SMOTE ===")

    rf_smote = train_rf_smote(X_train_smote, y_train_smote)

    y_val_pred_rf_smote, y_val_proba_rf_smote = predict_rf(rf_smote, X_val_scaled)
    y_test_pred_rf_smote, y_test_proba_rf_smote = predict_rf(rf_smote, X_test_scaled)

    val_metrics_rf_smote = evaluate_model(
        y_val, y_val_pred_rf_smote, y_val_proba_rf_smote,
        dataset_name="Validation",
        model_name="Random Forest + SMOTE"
    )

    test_metrics_rf_smote = evaluate_model(
        y_test, y_test_pred_rf_smote, y_test_proba_rf_smote,
        dataset_name="Test",
        model_name="Random Forest + SMOTE"
    )

    plot_confusion_matrix(
        y_test,
        y_test_pred_rf_smote,
        title="Random Forest + SMOTE - Confusion Matrix (Test)"
    )

    plot_pr_curve(
        y_val, y_val_proba_rf_smote,
        y_test, y_test_proba_rf_smote,
        val_metrics_rf_smote,
        test_metrics_rf_smote,
        model_name="Random Forest + SMOTE"
    )

    rf_smote_val_thresholds, rf_smote_best_row, rf_smote_test_best = threshold_tuning_report(
        y_val=y_val,
        y_val_proba=y_val_proba_rf_smote,
        y_test=y_test,
        y_test_proba=y_test_proba_rf_smote,
        model_name="Random Forest + SMOTE",
        optimize_for="F1-score"
    )

    plot_threshold_metrics(
        rf_smote_val_thresholds,
        model_name="Random Forest + SMOTE",
        dataset_name="Validation"
    )

    plot_confusion_matrix_threshold(
        y_test,
        y_test_proba_rf_smote,
        threshold=rf_smote_best_row["Threshold"],
        model_name="Random Forest + SMOTE",
        dataset_name="Test",
        cmap="Greens"
    )

    # ------------------------------------------------
    # 4. XGBOOST BASELINE
    # ------------------------------------------------
    print("\n=== XGBoost Baseline ===")

    xgb_baseline = train_xgb_baseline(X_train, y_train)

    y_val_pred_xgb, y_val_proba_xgb = predict_xgb(xgb_baseline, X_val)
    y_test_pred_xgb, y_test_proba_xgb = predict_xgb(xgb_baseline, X_test)

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

    plot_confusion_matrix(
        y_test,
        y_test_pred_xgb,
        title="XGBoost Baseline - Confusion Matrix (Test)"
    )

    plot_pr_curve(
        y_val, y_val_proba_xgb,
        y_test, y_test_proba_xgb,
        val_metrics_xgb,
        test_metrics_xgb,
        model_name="XGBoost Baseline"
    )

    xgb_val_thresholds, xgb_best_row, xgb_test_best = threshold_tuning_report(
        y_val=y_val,
        y_val_proba=y_val_proba_xgb,
        y_test=y_test,
        y_test_proba=y_test_proba_xgb,
        model_name="XGBoost Baseline",
        optimize_for="F1-score"
    )

    plot_threshold_metrics(
        xgb_val_thresholds,
        model_name="XGBoost Baseline",
        dataset_name="Validation"
    )

    plot_confusion_matrix_threshold(
        y_test,
        y_test_proba_xgb,
        threshold=xgb_best_row["Threshold"],
        model_name="XGBoost Baseline",
        dataset_name="Test",
        cmap="Purples"
    )

    # ------------------------------------------------
    # 5. XGBOOST + SMOTE
    # ------------------------------------------------
    print("\n=== XGBoost + SMOTE ===")

    xgb_smote = train_xgb_smote(X_train_smote, y_train_smote)

    y_val_pred_xgb_smote, y_val_proba_xgb_smote = predict_xgb(xgb_smote, X_val_scaled)
    y_test_pred_xgb_smote, y_test_proba_xgb_smote = predict_xgb(xgb_smote, X_test_scaled)

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

    plot_confusion_matrix(
        y_test,
        y_test_pred_xgb_smote,
        title="XGBoost + SMOTE - Confusion Matrix (Test)"
    )

    plot_pr_curve(
        y_val, y_val_proba_xgb_smote,
        y_test, y_test_proba_xgb_smote,
        val_metrics_xgb_smote,
        test_metrics_xgb_smote,
        model_name="XGBoost + SMOTE"
    )

    xgb_smote_val_thresholds, xgb_smote_best_row, xgb_smote_test_best = threshold_tuning_report(
        y_val=y_val,
        y_val_proba=y_val_proba_xgb_smote,
        y_test=y_test,
        y_test_proba=y_test_proba_xgb_smote,
        model_name="XGBoost + SMOTE",
        optimize_for="F1-score"
    )

    plot_threshold_metrics(
        xgb_smote_val_thresholds,
        model_name="XGBoost + SMOTE",
        dataset_name="Validation"
    )

    plot_confusion_matrix_threshold(
        y_test,
        y_test_proba_xgb_smote,
        threshold=xgb_smote_best_row["Threshold"],
        model_name="XGBoost + SMOTE",
        dataset_name="Test",
        cmap="Oranges"
    )

    print("\n=== Pipeline completed successfully ===")


if __name__ == "__main__":
    main()