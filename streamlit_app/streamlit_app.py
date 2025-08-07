# vehicle_classifier_app.py

import streamlit as st
import requests
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd

# --- App Configuration ---
st.set_page_config(
    page_title="Vehicle Classifier",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Force matplotlib to use non-GUI backend
import matplotlib
matplotlib.use('Agg')

# --- Constants ---
API_URL = "https://vehicle-classifier-api.onrender.com"
PREDICT_ENDPOINT = f"{API_URL}/predict"
RETRAIN_ENDPOINT = f"{API_URL}/retrain"

st.title("🚗 MLOps Vehicle Classifier")
st.markdown("Upload a photo of a car or a motorcycle for a prediction, or upload a zip file of new data to retrain the model.")

# --- Prediction Function ---
def get_prediction(image_bytes):
    try:
        from PIL import Image
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        jpeg_bytes = buffered.getvalue()

        base64_encoded_image = base64.b64encode(jpeg_bytes).decode('utf-8')
        payload = {"image": base64_encoded_image}

        response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        return result.get("predicted_class"), result.get("confidence")

    except requests.exceptions.RequestException as e:
        st.error("🔴 API request failed")
        st.exception(e)
        return None, None
    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)
        return None, None

# --- Retraining Function ---
def trigger_retraining(zip_file):
    try:
        zip_file.seek(0)
        files = {'zip_file': (zip_file.name, zip_file, 'application/zip')}
        response = requests.post(RETRAIN_ENDPOINT, files=files, timeout=900)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("⏳ Retraining timed out. Try again later.")
        return None
    except requests.exceptions.RequestException as e:
        st.error("🔴 Retraining request failed")
        st.exception(e)
        return None
    except Exception as e:
        st.error("❌ Retraining failed")
        st.exception(e)
        return None

# --- API Health Check ---
def check_api_health():
    try:
        response = requests.get(API_URL, timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.warning("⚠️ API unreachable")
        return False

# --- Sidebar: API Status ---
with st.sidebar:
    st.header("API Status")
    if check_api_health():
        st.success("🟢 API is online")
    else:
        st.error("🔴 API is offline")

    st.markdown("---")
    st.markdown("**API Endpoints:**")
    st.code(f"Predict: {PREDICT_ENDPOINT}")
    st.code(f"Retrain: {RETRAIN_ENDPOINT}")

# --- UI: Predict ---
st.header("🔍 Predict a Vehicle")
uploaded_file_predict = st.file_uploader(
    "Upload a vehicle image...",
    type=["jpg", "jpeg", "png"],
    key="predict_uploader"
)

if uploaded_file_predict:
    st.image(uploaded_file_predict, caption='Uploaded Image', use_column_width=True)

    if st.button("🎯 Make Prediction", type="primary"):
        with st.spinner("Processing prediction..."):
            uploaded_file_predict.seek(0)
            image_bytes = uploaded_file_predict.read()

            predicted_class, confidence = get_prediction(image_bytes)

            if predicted_class is not None and confidence is not None:
                st.success("✅ Prediction successful")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", predicted_class)
                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")

                st.progress(confidence)

                if confidence > 0.8:
                    st.success("High confidence prediction 🎯")
                elif confidence > 0.6:
                    st.info("Medium confidence prediction")
                else:
                    st.warning("Low confidence - consider using a clearer image")
            else:
                st.error("❌ Could not get a prediction from the API")

st.divider()

# --- UI: Retrain ---
st.header("🔄 Retrain Model")
st.markdown(
    """
    Upload a `.zip` file with your new training images.

    **Folder structure:**
    ```
    your_data.zip
    ├── Cars/
    │   ├── car1.jpg
    │   └── ...
    └── Motorcycles/
        ├── bike1.jpg
        └── ...
    ```
    """
)

uploaded_file_retrain = st.file_uploader(
    "Upload ZIP for retraining...",
    type=["zip"],
    key="retrain_uploader"
)

if uploaded_file_retrain:
    st.info(f"📁 File uploaded: `{uploaded_file_retrain.name}` ({uploaded_file_retrain.size} bytes)")

    if uploaded_file_retrain.size > 50 * 1024 * 1024:
        st.warning("⚠️ File is large — retraining may take longer or fail on limited servers.")

    if st.button("🚀 Start Retraining", type="primary"):
        with st.spinner("Retraining in progress..."):
            retrain_result = trigger_retraining(uploaded_file_retrain)

            if retrain_result:
                st.success(retrain_result.get('message', '✅ Retraining complete'))

                metrics = retrain_result.get('metrics', {})
                if metrics:
                    st.subheader("📊 Retraining Metrics")
                    cols = st.columns(len(metrics))
                    for i, (k, v) in enumerate(metrics.items()):
                        with cols[i]:
                            st.metric(k.replace("_", " ").title(), f"{v:.4f}" if isinstance(v, float) else str(v))

                    # Chart
                    numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                    if numeric_metrics:
                        st.subheader("📈 Metric Chart")
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.bar(numeric_metrics.keys(), numeric_metrics.values())
                        ax.set_ylabel("Score")
                        ax.set_ylim(0, 1)
                        ax.set_title("Model Performance")
                        plt.xticks(rotation=45, ha="right")
                        st.pyplot(fig)
                else:
                    st.info("ℹ️ No metrics returned.")
            else:
                st.error("❌ Retraining failed")

# --- Footer ---
st.markdown("---")
st.markdown("**Tips:**")
st.markdown("- Use sharp, well-lit images")
st.markdown("- Make sure folder names are correct (`Cars/`, `Motorcycles/`)")
st.markdown("- Retraining may take a few minutes depending on data size")
