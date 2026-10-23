"""Small interactive demo for the trained Iris classifier."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "iris_classifier.joblib"

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")
st.title("Iris Flower Classifier")
st.write("Enter flower measurements in centimetres to estimate its species.")

if not MODEL_PATH.is_file():
    st.error("Model not found. From the project root, run `python -m src.classifier train` first.")
    st.stop()

@st.cache_resource
def load_bundle():
    # Load only a trusted joblib model artifact.
    return joblib.load(MODEL_PATH)

bundle = load_bundle()
model = bundle["model"]
classes = bundle["class_names"]

with st.form("measurements"):
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Sepal length (cm)", min_value=0.1, max_value=10.0, value=5.1, step=0.1)
        petal_length = st.number_input("Petal length (cm)", min_value=0.1, max_value=10.0, value=1.4, step=0.1)
    with col2:
        sepal_width = st.number_input("Sepal width (cm)", min_value=0.1, max_value=10.0, value=3.5, step=0.1)
        petal_width = st.number_input("Petal width (cm)", min_value=0.1, max_value=10.0, value=0.2, step=0.1)
    submitted = st.form_submit_button("Predict species", type="primary")

if submitted:
    values = [[sepal_length, sepal_width, petal_length, petal_width]]
    index = int(model.predict(values)[0])
    probabilities = model.predict_proba(values)[0]
    st.success(f"Predicted species: **{classes[index]}**")
    chart = pd.DataFrame({"Estimated probability": probabilities}, index=classes)
    st.bar_chart(chart)
    st.caption("Educational demonstration only; validate on representative data before operational use.")
