
from pathlib import Path
from time import perf_counter
from threading import Lock

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "iris_model.joblib"

app = FastAPI(
    title="Iris Classification API",
    description="Predict Iris flower species using a trained Random Forest model.",
    version="1.1.0",
)

model = joblib.load(MODEL_PATH)

metrics = {
    "total_requests": 0,
    "prediction_requests": 0,
    "prediction_errors": 0,
    "total_response_time_ms": 0.0,
}
metrics_lock = Lock()


class IrisFeatures(BaseModel):
    sepal_length: float = Field(gt=0, le=10)
    sepal_width: float = Field(gt=0, le=10)
    petal_length: float = Field(gt=0, le=10)
    petal_width: float = Field(gt=0, le=10)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start = perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed_ms = (perf_counter() - start) * 1000
        with metrics_lock:
            metrics["total_requests"] += 1
            metrics["total_response_time_ms"] += elapsed_ms
            if request.url.path == "/predict":
                metrics["prediction_requests"] += 1
                if "response" in locals() and response.status_code >= 400:
                    metrics["prediction_errors"] += 1


@app.get("/")
def home():
    return {"message": "Iris Classification API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/metrics")
def get_metrics():
    with metrics_lock:
        result = dict(metrics)

    count = result["total_requests"]
    result["average_response_time_ms"] = round(
        result["total_response_time_ms"] / count, 2
    ) if count else 0.0
    result["total_response_time_ms"] = round(
        result["total_response_time_ms"], 2
    )
    return result


@app.post("/predict")
def predict(features: IrisFeatures):
    values = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])

    try:
        prediction = int(model.predict(values)[0])
        probabilities = model.predict_proba(values)[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Model prediction failed",
        ) from exc

    species = ["setosa", "versicolor", "virginica"][prediction]

    return {
        "prediction": prediction,
        "species": species,
        "probabilities": {
            name: round(float(probability), 4)
            for name, probability in zip(
                ["setosa", "versicolor", "virginica"],
                probabilities,
            )
        },
    }
