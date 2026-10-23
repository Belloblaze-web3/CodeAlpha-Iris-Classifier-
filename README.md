# Iris Flower Classifier — AI Project 2

A compact, reproducible supervised-learning project implementing the workflow requested in the attached DecodeLabs Project 2 brief: load and understand a dataset, create train/test sets, train a classifier, and evaluate it on held-out data.

## Problem and dataset

The model predicts one of three Iris species from four measurements: sepal length, sepal width, petal length, and petal width (all in cm). It uses scikit-learn's canonical Iris dataset, a real, small, labeled dataset bundled with scikit-learn (150 observations, 4 numeric features, 3 classes). No external download or fabricated records are required.

## Engineering choices

- Stratified 80/20 train/test split keeps class proportions representative; `random_state=42` makes the split reproducible.
- A scikit-learn `Pipeline` fits `StandardScaler` on training data only, then trains multinomial logistic regression. Keeping preprocessing inside the pipeline avoids test-set leakage.
- Evaluation reports accuracy, per-class precision/recall/F1, and a confusion matrix on held-out observations.
- Training persists the fitted pipeline and JSON metrics under `artifacts/`.
- CLI validates input dimensions and finite numeric values and returns class probabilities.

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python -m src.classifier train
python -m src.classifier predict 5.1 3.5 1.4 0.2
python -m unittest discover -s tests -v
```

The first run prints held-out test accuracy and creates `artifacts/iris_classifier.joblib` and `artifacts/metrics.json`. The supplied example measurements are for demonstration only; enter the same four features in the documented order.

## Project layout

```text
src/classifier.py  Data loading, pipeline, training/evaluation, prediction CLI
artifacts/         Generated model and evaluation metrics
 tests/            Unit tests for data, training, and prediction
requirements.txt   Runtime and test dependencies
```

## Reproducibility and limitations

Metrics are for the fixed 20% held-out split and should not be interpreted as a guarantee of real-world performance. Iris is an educational dataset, not a production domain dataset. For production, validate on representative data, monitor drift, manage model versions, and review privacy and safety requirements. Only load the generated joblib artifact from a trusted source; pickle-based formats can execute code when deserialized.


## Deployment options

Install both deployment wrappers' dependencies with `python -m pip install -r requirements-deploy.txt`, then train the model if `artifacts/iris_classifier.joblib` is absent.

### FastAPI (JSON API)

Best when another service, mobile app, or website will call the model.

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for interactive Swagger documentation. Health check: `GET /health`. Prediction: `POST /predict` with JSON:

```json
{
  "sepal_length_cm": 5.1,
  "sepal_width_cm": 3.5,
  "petal_length_cm": 1.4,
  "petal_width_cm": 0.2
}
```

Example command:

```bash
curl -X POST http://localhost:8000/predict \\
  -H 'Content-Type: application/json' \\
  -d '{"sepal_length_cm":5.1,"sepal_width_cm":3.5,"petal_length_cm":1.4,"petal_width_cm":0.2}'
```

For a public deployment, use a managed container/web service, bind to its provided port, and set up HTTPS, authentication/rate limits as appropriate, health checks, logging, and resource limits. Do not expose an unauthenticated prediction API to sensitive or costly workloads.

### Streamlit (interactive demo)

Best for a quick human-facing demo rather than a general-purpose API.

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

The demo collects the four measurements and displays a prediction and class-probability chart. It can be hosted on Streamlit Community Cloud or a container host; include `requirements-deploy.txt` and the model artifact, or configure the build to train the model before launch.

### Packaging notes

The model is loaded from the project `artifacts/` folder, so keep that artifact in the deployment image or create it during build. For repeatable releases, pin exact dependency versions after validating a deployment environment and record the training-data/model version. Treat `.joblib` files as trusted executable artifacts only. These examples are deployment-ready starting points, not a public hosted service; they do not provision hosting, a domain, or authentication.
