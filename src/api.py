
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "iris_model.joblib"

app = FastAPI(
    title="Iris Classification API",
    description="Predict Iris flower species using a trained Random Forest model.",
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(gt=0, le=10)
    sepal_width: float = Field(gt=0, le=10)
    petal_length: float = Field(gt=0, le=10)
    petal_width: float = Field(gt=0, le=10)


@app.get("/")
def home():
    return {"message": "Iris Classification API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(features: IrisFeatures):
    values = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])

    prediction = int(model.predict(values)[0])
    species = ["setosa", "versicolor", "virginica"][prediction]

    probabilities = model.predict_proba(values)[0]

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
