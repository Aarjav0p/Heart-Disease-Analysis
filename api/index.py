"""
CardioLens — Vercel Serverless Function (Python)

Loads pre-trained scikit-learn pipelines from ../models/ and exposes
/api/health, /api/model-summary, and /api/predict endpoints.
"""

import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="CardioLens Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading — happens once per cold start (< 1 s)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


def _load_artifact(path: Path):
    """Load a joblib artifact or return None if missing."""
    if not path.exists():
        return None
    return joblib.load(path)


logistic_model = _load_artifact(MODEL_DIR / "logistic_regression.joblib")
random_forest = _load_artifact(MODEL_DIR / "random_forest.joblib")

metadata: dict = {}
meta_path = MODEL_DIR / "metadata.json"
if meta_path.exists():
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class PatientInput(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: Literal["Male", "Female"]
    chestPain: Literal[
        "Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"
    ]
    bp: float = Field(..., ge=50, le=300)
    cholesterol: float = Field(..., ge=50, le=700)
    sugar: Literal["Normal", "Elevated"]
    ecg: Literal["Normal", "ST-T abnormality", "LV hypertrophy"]
    maxRate: float = Field(..., ge=40, le=250)
    angina: Literal["No", "Yes"]
    oldpeak: float = Field(..., ge=-5, le=10)
    slope: Literal["Upsloping", "Flat", "Downsloping"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patient_frame(p: PatientInput) -> pd.DataFrame:
    """Convert validated patient input into the DataFrame the pipelines expect."""
    return pd.DataFrame(
        [
            {
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
            }
        ]
    )


def _models_ready() -> bool:
    return logistic_model is not None and random_forest is not None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health():
    return {"status": "ok", "modelsLoaded": _models_ready()}


@app.get("/api/model-summary")
def model_summary():
    return {
        "datasetRows": metadata.get("rows"),
        "featureCount": len(metadata.get("features", [])),
        "metrics": [
            {"model": name, **values}
            for name, values in metadata.get("metrics", {}).items()
        ],
        "modelsLoaded": _models_ready(),
    }


@app.post("/api/predict")
def predict(patient: PatientInput):
    if not _models_ready():
        raise HTTPException(
            status_code=500,
            detail="Model artifacts are missing. Run `python train_models.py` locally and redeploy.",
        )

    try:
        frame = _patient_frame(patient)

        lr_prob = float(logistic_model.predict_proba(frame)[0, 1])
        rf_prob = float(random_forest.predict_proba(frame)[0, 1])
        lr_pred = int(logistic_model.predict(frame)[0])
        rf_pred = int(random_forest.predict(frame)[0])

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
            "metrics": [
                {"model": name, **values}
                for name, values in metadata.get("metrics", {}).items()
            ],
            "disclaimer": "Educational model output; not a clinical diagnosis.",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
