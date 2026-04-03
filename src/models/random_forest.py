from sklearn.ensemble import RandomForestClassifier

def train_rf_baseline(
    X_train,
    y_train,
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
):
    """
    Antrenează modelul Random Forest baseline pe datele originale.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs
    )

    model.fit(X_train, y_train)
    return model


def train_rf_smote(
    X_train_smote,
    y_train_smote,
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
):
    """
    Antrenează modelul Random Forest pe date echilibrate cu SMOTE.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs
    )

    model.fit(X_train_smote, y_train_smote)
    return model


def predict_rf(model, X):
    """
    Returnează predicțiile binare și probabilitățile pentru clasa pozitivă.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return y_pred, y_proba