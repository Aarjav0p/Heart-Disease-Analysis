import os
import glob
import json
import joblib
import kagglehub
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

DATASET_ID = "sid321axn/heart-statlog-cleveland-hungary-final"

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("Downloading dataset...")
path = kagglehub.dataset_download(DATASET_ID)
csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
if not csv_files:
    raise FileNotFoundError("No CSV file found in the downloaded dataset.")
csv_files.sort(key=lambda p: ("heart" not in os.path.basename(p).lower(), len(p)))
data = pd.read_csv(csv_files[0])
data.columns = data.columns.str.strip()

data = data.drop_duplicates().reset_index(drop=True)

if "target" not in data.columns:
    raise ValueError(f"Expected 'target' column. Found: {data.columns.tolist()}")

if data["target"].dtype == object:
    mapping = {
        "0": 0, "1": 1,
        "No": 0, "Yes": 1,
        "no": 0, "yes": 1,
        "Absence": 0, "Presence": 1,
    }
    data["target"] = data["target"].map(mapping)

data["target"] = pd.to_numeric(data["target"], errors="coerce")
if data["target"].isna().any():
    raise ValueError("Target contains values that could not be converted to 0/1.")
data["target"] = data["target"].astype(int)

data.to_csv("data/heart_disease.csv", index=False)

X = data.drop(columns=["target"])
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

categorical_cols = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()
numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

known_categorical = [
    "sex",
    "chest pain type",
    "fasting blood sugar",
    "resting ecg",
    "exercise angina",
    "ST slope",
]
for col in known_categorical:
    if col in X_train.columns and col not in categorical_cols:
        categorical_cols.append(col)
for col in categorical_cols:
    if col in numerical_cols:
        numerical_cols.remove(col)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols),
])

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )),
])

random_forest = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42
    )),
])

print("Training Logistic Regression...")
logistic_model.fit(X_train, y_train)

print("Training Random Forest...")
random_forest.fit(X_train, y_train)

def evaluate(model):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "balancedAccuracy": round(float(balanced_accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "rocAuc": round(float(roc_auc_score(y_test, prob)), 4),
    }

metrics = {
    "Logistic Regression": evaluate(logistic_model),
    "Random Forest": evaluate(random_forest),
}

joblib.dump(logistic_model, "models/logistic_regression.joblib")
joblib.dump(random_forest, "models/random_forest.joblib")

metadata = {
    "dataset": DATASET_ID,
    "rows": int(len(data)),
    "features": list(X.columns),
    "categorical_columns": categorical_cols,
    "numerical_columns": numerical_cols,
    "target": "target",
    "random_state": 42,
    "test_size": 0.20,
    "metrics": metrics,
}

with open("models/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nTraining complete.")
print("Created models/logistic_regression.joblib")
print("Created models/random_forest.joblib")
print("Created models/metadata.json")
print("Created data/heart_disease.csv")
print("\nTest-set metrics:")
for name, vals in metrics.items():
    print(name, vals)
