"""FastAPI service exposing the trained diagnosis model.

Every prediction request is appended to logs/predictions.jsonl so that
`python -m src.drift` can later compare production inputs against the
training distribution.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"

_log_lock = threading.Lock()


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description="The 30 numeric features of the breast-cancer dataset, in canonical order.",
        min_length=30,
        max_length=30,
    )


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability_malignant: float
    probability_benign: float


def _load_artifacts():
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    metrics = json.loads((ARTIFACTS_DIR / "metrics.json").read_text())
    feature_names = json.loads((ARTIFACTS_DIR / "feature_names.json").read_text())
    return model, metrics, feature_names


app = FastAPI(
    title="Medical Classifier API",
    description="Breast-cancer diagnosis model served with full MLOps tooling "
    "(CI/CD, containerization, request logging, drift monitoring).",
    version="1.0.0",
)

model, training_metrics, feature_names = _load_artifacts()


def _log_request(features: list[float], prediction: int) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "prediction": prediction,
    }
    with _log_lock, PREDICTIONS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model/info")
def model_info() -> dict:
    return {"feature_names": feature_names, "metrics": training_metrics}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        proba = model.predict_proba([request.features])[0]
    except Exception as exc:  # malformed values that pass schema validation
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # In this dataset target=0 is malignant, target=1 is benign.
    prediction = int(proba[1] >= 0.5)
    _log_request(request.features, prediction)

    return PredictionResponse(
        prediction=prediction,
        label="benign" if prediction == 1 else "malignant",
        probability_malignant=round(float(proba[0]), 4),
        probability_benign=round(float(proba[1]), 4),
    )
