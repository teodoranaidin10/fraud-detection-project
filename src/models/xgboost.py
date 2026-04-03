from xgboost import XGBClassifier


def compute_scale_pos_weight(y_train):
    """
    Calculează scale_pos_weight pentru clase dezechilibrate.
    """
    negative_count = (y_train == 0).sum()
    positive_count = max((y_train == 1).sum(), 1)
    return negative_count / positive_count


def train_xgb_baseline(
    X_train,
    y_train,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
):
    """
    Antrenează modelul XGBoost baseline folosind datele originale
    și scale_pos_weight pentru dezechilibrul claselor.
    """
    scale_pos_weight = compute_scale_pos_weight(y_train)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=n_jobs
    )

    model.fit(X_train, y_train)
    return model


def train_xgb_smote(
    X_train_smote,
    y_train_smote,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
):
    """
    Antrenează modelul XGBoost pe date echilibrate cu SMOTE.
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=n_jobs
    )

    model.fit(X_train_smote, y_train_smote)
    return model


def predict_xgb(model, X):
    """
    Returnează predicțiile binare și probabilitățile pentru clasa pozitivă.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return y_pred, y_proba