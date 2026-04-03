import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Încarcă dataset-ul din fișier CSV.
    """
    df = pd.read_csv(csv_path)
    return df


def split_features_target(df: pd.DataFrame, target_col: str = "Class"):
    """
    Separă feature-urile și target-ul.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
):
    """
    Împarte datele în train / validation / test, cu stratificare.
    """
    if round(train_size + val_size + test_size, 5) != 1.0:
        raise ValueError("train_size + val_size + test_size trebuie să fie 1.0")

    # Mai întâi: train și temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1 - train_size),
        stratify=y,
        random_state=random_state
    )

    # Apoi: temp -> validation + test
    relative_test_size = test_size / (val_size + test_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_datasets(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Aplică StandardScaler:
    - fit pe train
    - transform pe validation și test
    """
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def apply_smote(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 1.0,
    random_state: int = 42,
    k_neighbors: int = 5,
):
    """
    Aplică SMOTE doar pe train.
    """
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors
    )

    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    return X_train_smote, y_train_smote


def convert_to_numpy(
    X_train_scaled: pd.DataFrame,
    X_val_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    X_train_smote,
    y_train_smote,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
):
    """
    Convertește datele în numpy arrays pentru modele DL.
    """
    arrays = {
        "X_train_scaled_np": X_train_scaled.to_numpy(dtype=np.float32),
        "X_val_scaled_np": X_val_scaled.to_numpy(dtype=np.float32),
        "X_test_scaled_np": X_test_scaled.to_numpy(dtype=np.float32),
        "X_train_smote_np": np.asarray(X_train_smote, dtype=np.float32),
        "y_train_smote_np": np.asarray(y_train_smote, dtype=np.float32),
        "y_train_np": y_train.to_numpy(dtype=np.float32),
        "y_val_np": y_val.to_numpy(dtype=np.float32),
        "y_test_np": y_test.to_numpy(dtype=np.float32),
    }
    return arrays


def print_data_summary(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    X_train_smote=None,
    y_train_smote=None,
):
    """
    Afișează informații utile pentru debugging și verificare.
    """
    print("Train:", X_train.shape, y_train.mean())
    print("Val:", X_val.shape, y_val.mean())
    print("Test:", X_test.shape, y_test.mean())

    print("\nValori lipsă în X_train:")
    print(X_train.isnull().sum().sort_values(ascending=False).head(10))

    print("\nValori lipsă în X_val:")
    print(X_val.isnull().sum().sort_values(ascending=False).head(10))

    print("\nValori lipsă în X_test:")
    print(X_test.isnull().sum().sort_values(ascending=False).head(10))

    print("\nDistribuția clasei înainte de SMOTE:")
    print(y_train.value_counts())
    print(y_train.value_counts(normalize=True))

    if X_train_smote is not None and y_train_smote is not None:
        print("\nDistribuția clasei după SMOTE:")
        print(pd.Series(y_train_smote).value_counts())
        print(pd.Series(y_train_smote).value_counts(normalize=True))

        print("\nShape după SMOTE:")
        print("X_train_smote:", X_train_smote.shape)
        print("y_train_smote:", y_train_smote.shape)


def prepare_data(
    csv_path: str,
    target_col: str = "Class",
    random_state: int = 42,
    use_smote: bool = True,
    return_numpy: bool = True,
    verbose: bool = True,
):
    """
    Pipeline complet:
    - load dataset
    - split X/y
    - split train/val/test
    - scaling
    - SMOTE pe train
    - conversie opțională la numpy

    Returnează un dicționar cu toate obiectele utile.
    """
    df = load_dataset(csv_path)
    X, y = split_features_target(df, target_col=target_col)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y, random_state=random_state
    )

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_datasets(
        X_train, X_val, X_test
    )

    X_train_smote, y_train_smote = None, None
    if use_smote:
        X_train_smote, y_train_smote = apply_smote(
            X_train_scaled, y_train, random_state=random_state
        )

    if verbose:
        print_data_summary(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            X_train_smote=X_train_smote,
            y_train_smote=y_train_smote
        )

        print("\nShape după scaling:")
        print("X_train_scaled:", X_train_scaled.shape)
        print("X_val_scaled:", X_val_scaled.shape)
        print("X_test_scaled:", X_test_scaled.shape)

    data = {
        "df": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_smote": X_train_smote,
        "y_train_smote": y_train_smote,
        "scaler": scaler,
    }

    if return_numpy:
        np_data = convert_to_numpy(
            X_train_scaled=X_train_scaled,
            X_val_scaled=X_val_scaled,
            X_test_scaled=X_test_scaled,
            X_train_smote=X_train_smote,
            y_train_smote=y_train_smote,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test
        )
        data.update(np_data)

        if verbose:
            print("\nConversie la numpy finalizată.")
            print("X_train_scaled_np:", data["X_train_scaled_np"].shape)
            if use_smote and data["X_train_smote_np"] is not None:
                print("X_train_smote_np:", data["X_train_smote_np"].shape)

    return data