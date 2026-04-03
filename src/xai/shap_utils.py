import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simple_shap_analysis(model, X_sample, feature_names, model_name="Model", local_index=0):
    """
    model: model antrenat (RandomForest / XGBoost)
    X_sample: DataFrame sau array
    feature_names: lista numelor de coloane
    model_name: numele modelului
    local_index: observația pentru explicație locală
    """

    # asigurăm DataFrame
    if not isinstance(X_sample, pd.DataFrame):
        X_sample = pd.DataFrame(X_sample, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # -----------------------------
    # Normalizare formă SHAP output
    # -----------------------------
    if isinstance(shap_values, list):
        # caz clasic: listă [class0, class1]
        shap_values_class1 = shap_values[1]
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            # posibil format: (n_samples, n_features, n_classes)
            shap_values_class1 = shap_values[:, :, 1]
        elif shap_values.ndim == 2:
            # format deja bun: (n_samples, n_features)
            shap_values_class1 = shap_values
        else:
            raise ValueError(f"Format SHAP neașteptat: {shap_values.shape}")
    else:
        raise ValueError(f"Tip SHAP neașteptat: {type(shap_values)}")

    # verificare finală
    print(f"{model_name} | shap_values_class1 shape: {shap_values_class1.shape}")

    # -----------------------------
    # GLOBAL
    # -----------------------------
    mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)

    shap.summary_plot(
    shap_values_class1,
    X_sample,
    feature_names=feature_names
     )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    print(f"\n===== {model_name} | SHAP Global =====")
    print(importance_df.head(10))

    plt.figure(figsize=(10, 6))
    top_global = importance_df.head(10).sort_values("mean_abs_shap")
    plt.barh(top_global["feature"], top_global["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"{model_name} - Top 10 SHAP Features")
    plt.grid(True)
    plt.show()

    # -----------------------------
    # LOCAL
    # -----------------------------
    print(f"\n===== {model_name} | SHAP Local (index={local_index}) =====")

    local_shap = shap_values_class1[local_index]
    local_features = X_sample.iloc[local_index]

    local_df = pd.DataFrame({
        "feature": feature_names,
        "value": local_features.values,
        "shap_value": local_shap,
        "abs_shap": np.abs(local_shap)
    }).sort_values("abs_shap", ascending=False)

    print(local_df.head(10)[["feature", "value", "shap_value"]])

    plt.figure(figsize=(10, 6))
    top_local = local_df.head(10).sort_values("shap_value")
    plt.barh(top_local["feature"], top_local["shap_value"])
    plt.xlabel("SHAP value")
    plt.title(f"{model_name} - Local explanation (sample {local_index})")
    plt.grid(True)
    plt.show()

    return importance_df, local_df, shap_values_class1

X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_train.columns)
X_explain = X_test_scaled_df.sample(300, random_state=42)

rf_shap_global, rf_shap_local, rf_shap_values = simple_shap_analysis(
    model=rf_model,
    X_sample=X_explain,
    feature_names=list(X_train.columns),
    model_name="Random Forest",
    local_index=0
)

xgb_shap_global, xgb_shap_local, xgb_shap_values = simple_shap_analysis(
    model=xgb_baseline,
    X_sample=X_explain,
    feature_names=list(X_train.columns),
    model_name="XGBoost",
    local_index=0
)