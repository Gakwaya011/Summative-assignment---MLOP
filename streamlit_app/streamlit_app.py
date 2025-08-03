import streamlit as st
import requests
import base64
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt

# --- App Configuration ---
st.set_page_config(
    page_title="Vehicle Classifier",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Constants ---
API_URL = "http://localhost:5000"
PREDICT_ENDPOINT = f"{API_URL}/predict"
RETRAIN_ENDPOINT = f"{API_URL}/retrain"

st.title("🚗 MLOps Vehicle Classifier")
st.markdown("Upload a photo of a car or a motorcycle for a prediction, or upload a zip file of new data to retrain the model.")

# --- Prediction Function ---
def get_prediction(image_bytes):
    try:
        base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        payload = {"image": base64_encoded_image}
        response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result.get("predicted_class"), result.get("confidence")
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# --- Retrain Function ---
def trigger_retraining(zip_file):
    try:
        files = {'zip_file': (zip_file.name, zip_file, 'application/zip')}
        response = requests.post(RETRAIN_ENDPOINT, files=files, timeout=600)  # Increased timeout for training
        response.raise_for_status()
        result = response.json()
        return result
    except Exception as e:
        st.error(f"Retraining error: {e}")
        return None

# --- UI: Prediction ---
st.header("Predict a Vehicle")
uploaded_file_predict = st.file_uploader("Choose a vehicle image...", type=["jpg", "jpeg", "png"])

if uploaded_file_predict:
    st.image(uploaded_file_predict, caption='Uploaded Image', use_column_width=True)
    image_bytes = uploaded_file_predict.read()
    if st.button("Predict"):
        with st.spinner("Making prediction..."):
            predicted_class, confidence = get_prediction(image_bytes)
            if predicted_class and confidence is not None:
                st.success("Prediction complete!")
                st.write(f"### Prediction: {predicted_class}")
                st.write(f"### Confidence: **{confidence:.2%}**")

st.divider()

# --- UI: Retraining ---
st.header("Retrain Model with New Data")
st.markdown(
    "Upload a zip file containing new images to retrain the model. "
    "The zip file should be structured with folders for each class (e.g., `Cars`, `Motorcycles`)."
)
uploaded_file_retrain = st.file_uploader("Choose a zip file with new data...", type=["zip"])

if uploaded_file_retrain:
    st.info(f"File '{uploaded_file_retrain.name}' uploaded. Click the button to start retraining.")

    if st.button("Start Retraining"):
        with st.spinner("Starting retraining process... This may take a while..."):
            retrain_result = trigger_retraining(uploaded_file_retrain)

            if retrain_result:
                st.success(retrain_result.get('message', 'Retraining done!'))

                metrics = retrain_result.get('metrics', {})
                if metrics:
                    st.subheader("Retraining Metrics")
                    # Show metrics values
                    for metric_name, metric_value in metrics.items():
                        st.write(f"**{metric_name.capitalize()}**: {metric_value:.4f}")

                    # Plot metrics bar chart
                    metric_names = list(metrics.keys())
                    metric_values = [metrics[k] for k in metric_names]

                    fig, ax = plt.subplots()
                    ax.bar(metric_names, metric_values, color='skyblue')
                    ax.set_ylim(0, 1)
                    ax.set_title('Retraining Metrics')
                    for i, v in enumerate(metric_values):
                        ax.text(i, v + 0.02, f"{v:.4f}", ha='center', fontsize=10)
                    st.pyplot(fig)
                else:
                    st.info("No retraining metrics available.")

            else:
                st.error("Retraining failed or returned no data.")
