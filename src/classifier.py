"""Train and use a reproducible Iris flower classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "iris_classifier.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"
RANDOM_STATE = 42


def load_data():
    """Load the real, canonical Iris dataset bundled with scikit-learn."""
    dataset = load_iris(as_frame=True)
    return dataset.data, dataset.target, list(dataset.target_names), list(dataset.feature_names)


def build_model():
    """Return a leakage-safe preprocessing and classification pipeline."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )


def train(model_path: Path = MODEL_PATH, metrics_path: Path = METRICS_PATH) -> dict:
    X, y, class_names, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(
        y_test, predictions, target_names=class_names, output_dict=True, zero_division=0
    )
    metrics = {
        "dataset": "Iris (scikit-learn bundled dataset)",
        "rows": int(len(X)),
        "features": feature_names,
        "classes": class_names,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": report,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names, "class_names": class_names}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def predict(features: list[float], model_path: Path = MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run `python -m src.classifier train` first.")
    bundle = joblib.load(model_path)
    if len(features) != len(bundle["feature_names"]):
        raise ValueError(f"Expected {len(bundle['feature_names'])} measurements, got {len(features)}.")
    if not np.isfinite(features).all():
        raise ValueError("All measurements must be finite numbers.")
    values = np.asarray(features, dtype=float).reshape(1, -1)
    model = bundle["model"]
    predicted_index = int(model.predict(values)[0])
    probabilities = model.predict_proba(values)[0]
    return {
        "prediction": bundle["class_names"][predicted_index],
        "probabilities": {
            name: round(float(probabilities[i]), 4)
            for i, name in enumerate(bundle["class_names"])
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or query the Iris flower classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="Train, evaluate, and save the model")
    predict_parser = subparsers.add_parser("predict", help="Predict a flower species")
    predict_parser.add_argument(
        "measurements", nargs=4, type=float,
        metavar=("SEPAL_LENGTH_CM", "SEPAL_WIDTH_CM", "PETAL_LENGTH_CM", "PETAL_WIDTH_CM"),
    )
    args = parser.parse_args()
    if args.command == "train":
        metrics = train()
        print(f"Test accuracy: {metrics['accuracy']:.3f}")
        print(f"Saved model: {MODEL_PATH}")
        print(f"Saved metrics: {METRICS_PATH}")
    else:
        print(json.dumps(predict(args.measurements), indent=2))


if __name__ == "__main__":
    main()
