### 1. Incarcarea setului de date si split pentru train, validation si test
### Train 70%. val 15%, test 15%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv("/kaggle/input/datasets/organizations/mlg-ulb/creditcardfraud/creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y, # Asigură că proporția de fraudă rămâne aceeași în toate subseturile.
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
) # Se imparte temp in 15% validare si 15% testare

# Print dimensiune dataset - (număr_linii, număr_features) si .mean() pe y, adica procentul de frauda, 17% pe fiecare set
print("Train:", X_train.shape, y_train.mean())
print("Val:", X_val.shape, y_val.mean())
print("Test:", X_test.shape, y_test.mean())


### 2. Preprocesare + Scaling + SMOTE -> DUPLICATE????

# 2.1 Verificare valori lipsa
print("Valori lipsa in X_train:\n", X_train.isnull().sum().sort_values(ascending=False).head(10))
print("Valori lipsa in X_val:\n", X_val.isnull().sum().sort_values(ascending=False).head(10))
print("Valori lipsa in X_test:\n", X_test.isnull().sum().sort_values(ascending=False).head(10))

# 2.2 Scaling
# StandardScaler este foarte important pentru MLP si Autoencoder.
# Il folosim pe toate coloanele numerice.
# ------------------------------------------------
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

print("\nShape dupa scaling:")
print("X_train_scaled:", X_train_scaled.shape)
print("X_val_scaled:", X_val_scaled.shape)
print("X_test_scaled:", X_test_scaled.shape)

# 2.4 Verificare distributie clasa inainte de SMOTE
print("\nDistributia clasei inainte de SMOTE:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True))


# 2.5 Aplicare SMOTE doar pe TRAIN
# IMPORTANT:
# - nu aplicam SMOTE pe validation
# - nu aplicam SMOTE pe test
smote = SMOTE(
    sampling_strategy=1.0,   # echilibrare completa: minoritara -> egala cu majoritara
    random_state=42,
    k_neighbors=5
)

# SMOTE genereaza date sintetice pentru clasa minoritara -> 199020 - 344 = 198676 fraude sintetice
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# 2.6 Verificare distributie clasa dupa SMOTE
print("\nDistributia clasei dupa SMOTE:")
print(pd.Series(y_train_smote).value_counts())
print(pd.Series(y_train_smote).value_counts(normalize=True))

print("\nShape dupa SMOTE:")
print("X_train_smote:", X_train_smote.shape)
print("y_train_smote:", y_train_smote.shape)

# Conversie in numpy pentru modele DL
X_train_scaled_np = X_train_scaled.to_numpy(dtype=np.float32)
X_val_scaled_np = X_val_scaled.to_numpy(dtype=np.float32)
X_test_scaled_np = X_test_scaled.to_numpy(dtype=np.float32)

X_train_smote_np = np.asarray(X_train_smote, dtype=np.float32)
y_train_smote_np = np.asarray(y_train_smote, dtype=np.float32)

y_train_np = y_train.to_numpy(dtype=np.float32)
y_val_np = y_val.to_numpy(dtype=np.float32)
y_test_np = y_test.to_numpy(dtype=np.float32)

print("\nConversie la numpy finalizata.")
print("X_train_scaled_np:", X_train_scaled_np.shape)
print("X_train_smote_np:", X_train_smote_np.shape)