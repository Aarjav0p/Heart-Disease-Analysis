import os
import glob
from functools import lru_cache
from typing import Literal

import kagglehub
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

APP_TITLE = "CardioLens Model API"
DATASET_ID = "sid321axn/heart-statlog-cleveland-hungary-final"

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientInput(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: Literal["Male", "Female"]
    chestPain: Literal["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"]
    bp: float = Field(..., ge=50, le=300)
    cholesterol: float = Field(..., ge=50, le=700)
    sugar: Literal["Normal", "Elevated"]
    ecg: Literal["Normal", "ST-T abnormality", "LV hypertrophy"]
    maxRate: float = Field(..., ge=40, le=250)
    angina: Literal["No", "Yes"]
    oldpeak: float = Field(..., ge=-5, le=10)
    slope: Literal["Upsloping", "Flat", "Downsloping"]

def _find_csv(download_path: str) -> str:
    csv_files = glob.glob(os.path.join(download_path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the downloaded Kaggle dataset.")
    # Prefer a heart-disease-looking filename where possible.
    csv_files.sort(key=lambda p: ("heart" not in os.path.basename(p).lower(), len(p)))
    return csv_files[0]

@lru_cache(maxsize=1)
def build_models():
    download_path = kagglehub.dataset_download(DATASET_ID)
    csv_path = _find_csv(download_path)
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()
    data = data.drop_duplicates().reset_index(drop=True)

    if "target" not in data.columns:
        raise RuntimeError(f"Expected 'target' column. Found: {data.columns.tolist()}")

    # Ensure target is numeric 0/1.
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
        raise RuntimeError("Target contains values that could not be converted to 0/1.")
    data["target"] = data["target"].astype(int)

    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # The public combined dataset stores several categorical clinical variables
    # as integer codes. Treat those fields as categorical for model training.
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

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    lr = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )
    rf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42
            )),
        ]
    )

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    def metrics(model, name):
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        return {
            "model": name,
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "balancedAccuracy": round(float(balanced_accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "rocAuc": round(float(roc_auc_score(y_test, prob)), 4),
        }

    test_metrics = [metrics(lr, "Logistic Regression"), metrics(rf, "Random Forest")]

    return {
        "lr": lr,
        "rf": rf,
        "metrics": test_metrics,
        "rows": int(len(data)),
        "features": int(X.shape[1]),
        "targetBalance": {
            "0": int((y == 0).sum()),
            "1": int((y == 1).sum()),
        },
    }

def _patient_frame(p: PatientInput) -> pd.DataFrame:
    return pd.DataFrame([{
        "age": p.age,
        "sex": 1 if p.sex == "Male" else 0,
        "chest pain type": {
            "Typical angina": 1,
            "Atypical angina": 2,
            "Non-anginal pain": 3,
            "Asymptomatic": 4,
        }[p.chestPain],
        "resting bp s": p.bp,
        "cholesterol": p.cholesterol,
        "fasting blood sugar": 1 if p.sugar == "Elevated" else 0,
        "resting ecg": {
            "Normal": 0,
            "ST-T abnormality": 1,
            "LV hypertrophy": 2,
        }[p.ecg],
        "max heart rate": p.maxRate,
        "exercise angina": 1 if p.angina == "Yes" else 0,
        "oldpeak": p.oldpeak,
        "ST slope": {
            "Upsloping": 1,
            "Flat": 2,
            "Downsloping": 3,
        }[p.slope],
    }])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-summary")
def model_summary():
    bundle = build_models()
    return {
        "datasetRows": bundle["rows"],
        "featureCount": bundle["features"],
        "metrics": bundle["metrics"],
        "targetBalance": bundle["targetBalance"],
    }

@app.post("/predict")
def predict(patient: PatientInput):
    try:
        bundle = build_models()
        frame = _patient_frame(patient)

        lr_prob = float(bundle["lr"].predict_proba(frame)[0, 1])
        rf_prob = float(bundle["rf"].predict_proba(frame)[0, 1])
        lr_pred = int(bundle["lr"].predict(frame)[0])
        rf_pred = int(bundle["rf"].predict(frame)[0])

        # Display a neutral "consensus probability" as the simple mean
        # of the two model probabilities; it is not presented as a calibrated diagnosis.
        consensus = (lr_prob + rf_prob) / 2.0

        return {
            "logisticRegression": {
                "prediction": lr_pred,
                "label": "Positive" if lr_pred == 1 else "Negative",
                "probability": round(lr_prob * 100, 1),
            },
            "randomForest": {
                "prediction": rf_pred,
                "label": "Positive" if rf_pred == 1 else "Negative",
                "probability": round(rf_prob * 100, 1),
            },
            "consensusProbability": round(consensus * 100, 1),
            "metrics": bundle["metrics"],
            "disclaimer": "Educational model output; not a clinical diagnosis.",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
