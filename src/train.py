
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATA_PATH = DATA_DIR / "iris.csv"
MODEL_PATH = MODEL_DIR / "iris_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def train_model():
    # Load the Iris dataset
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    # Save a reproducible dataset snapshot
    df.to_csv(DATA_PATH, index=False)

    X = df[iris.feature_names]
    y = df["target"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    # Track the experiment with MLflow
    mlflow.set_tracking_uri((ROOT_DIR / "mlruns").as_uri())
    mlflow.set_experiment("iris-classification")

    with mlflow.start_run():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("accuracy", accuracy)

        # Save the trained model
        joblib.dump(model, MODEL_PATH)

        # Save evaluation metrics
        metrics = {
            "accuracy": float(accuracy),
            "classification_report": classification_report(
                y_test,
                predictions,
                target_names=iris.target_names,
                output_dict=True,
            ),
            "features": iris.feature_names,
            "classes": iris.target_names.tolist(),
        }

        METRICS_PATH.write_text(json.dumps(metrics, indent=4))

        # Log the model as an MLflow artifact
        mlflow.sklearn.log_model(model, artifact_path="iris_model")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Model saved to: {MODEL_PATH}")
        print(f"Dataset saved to: {DATA_PATH}")
        print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    train_model()
