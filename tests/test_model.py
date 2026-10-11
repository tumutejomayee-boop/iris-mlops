
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "iris_model.joblib"


def load_trained_model():
    assert MODEL_PATH.exists(), (
        "Trained model not found. Run python src/train.py first."
    )
    return joblib.load(MODEL_PATH)


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_model_predicts_valid_class():
    model = load_trained_model()
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(sample)

    assert prediction.shape == (1,)
    assert int(prediction[0]) in [0, 1, 2]


def test_model_predicts_all_iris_classes():
    model = load_trained_model()
    iris = load_iris()
    predictions = model.predict(iris.data)

    assert set(predictions).issubset({0, 1, 2})
    assert len(predictions) == len(iris.data)
