"""FastAPI wrapper for the trained Iris classifier."""
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "iris_classifier.joblib"
app = FastAPI(title="Iris Classifier API", version="1.0.0")


class PredictionRequest(BaseModel):
    sepal_length_cm: float = Field(gt=0, le=10)
    sepal_width_cm: float = Field(gt=0, le=10)
    petal_length_cm: float = Field(gt=0, le=10)
    petal_width_cm: float = Field(gt=0, le=10)


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]


@app.on_event("startup")
def load_model() -> None:
    if not MODEL_PATH.is_file():
        raise RuntimeError("Model artifact is missing. Run `python -m src.classifier train` first.")
    # joblib/pickle artifacts must only be loaded from trusted sources.
    app.state.bundle = joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if hasattr(app.state, "bundle") else "starting"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    bundle = getattr(app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    values = np.asarray([[
        request.sepal_length_cm,
        request.sepal_width_cm,
        request.petal_length_cm,
        request.petal_width_cm,
    ]], dtype=float)
    model = bundle["model"]
    predicted_index = int(model.predict(values)[0])
    probabilities = model.predict_proba(values)[0]
    return PredictionResponse(
        prediction=bundle["class_names"][predicted_index],
        probabilities={name: round(float(probabilities[i]), 4)
                       for i, name in enumerate(bundle["class_names"])},
    )
