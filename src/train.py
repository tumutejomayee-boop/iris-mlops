

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.models
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


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

    # Save the dataset
    df.to_csv(DATA_PATH, index=False)

    X = df[iris.feature_names]
    y = df["target"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Create the model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    # Configure MLflow
    mlflow.set_tracking_uri((ROOT_DIR / "mlruns").as_uri())
    mlflow.set_experiment("iris-classification")

    with mlflow.start_run():
        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log parameters and metrics
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

        # Create MLflow model signature and input example
        input_example = X_test.iloc[:5]
        signature = mlflow.models.infer_signature(
            input_example,
            model.predict(input_example),
        )

        # Log the model to MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="iris_model",
            signature=signature,
            input_example=input_example,
        )

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Dataset saved to: {DATA_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    train_model()
