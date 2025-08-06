import streamlit as st
import requests
import base64
import os
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

# Force matplotlib to use non-GUI backend for server deployment
import matplotlib
matplotlib.use('Agg')

# --- Constants ---
API_URL = "http://127.0.0.1:5000"
PREDICT_ENDPOINT = f"{API_URL}/predict"
RETRAIN_ENDPOINT = f"{API_URL}/retrain"

st.title("🚗 MLOps Vehicle Classifier")
st.markdown("Upload a photo of a car or a motorcycle for a prediction, or upload a zip file of new data to retrain the model.")

# --- Prediction Function ---
def get_prediction(image_bytes):
    try:
        # Force convert to JPEG bytes using PIL (ensures clean decoding on backend)
        from PIL import Image
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        jpeg_bytes = buffered.getvalue()

        base64_encoded_image = base64.b64encode(jpeg_bytes).decode('utf-8')
        payload = {"image": base64_encoded_image}
        response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result.get("predicted_class"), result.get("confidence")
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None, None
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# --- Retrain Function ---
def trigger_retraining(zip_file):
    try:
        # Reset file pointer to beginning
        zip_file.seek(0)
        files = {'zip_file': (zip_file.name, zip_file, 'application/zip')}
        
        response = requests.post(RETRAIN_ENDPOINT, files=files, timeout=600)
        response.raise_for_status()
        result = response.json()
        return result
    except requests.exceptions.Timeout:
        st.error("Retraining request timed out. The process might still be running.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None
    except Exception as e:
        st.error(f"Retraining error: {e}")
        return None

# --- Health Check ---
def check_api_health():
    try:
        response = requests.get(API_URL, timeout=10)
        return response.status_code == 200
    except:
        return False

# --- API Status ---
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

# --- UI: Prediction ---
st.header("🔍 Predict a Vehicle")
uploaded_file_predict = st.file_uploader(
    "Choose a vehicle image...", 
    type=["jpg", "jpeg", "png"],
    key="predict_uploader"
)

if uploaded_file_predict:
    st.image(uploaded_file_predict, caption='Uploaded Image', use_column_width=True)
    
    if st.button("🎯 Make Prediction", type="primary"):
        with st.spinner("Making prediction..."):
            # Reset file pointer and read bytes
            uploaded_file_predict.seek(0)
            image_bytes = uploaded_file_predict.read()
            
            predicted_class, confidence = get_prediction(image_bytes)
            
            if predicted_class and confidence is not None:
                st.success("✅ Prediction complete!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", predicted_class)
                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Visual confidence bar
                st.progress(confidence)
                
                if confidence > 0.8:
                    st.success("High confidence prediction! 🎉")
                elif confidence > 0.6:
                    st.info("Medium confidence prediction")
                else:
                    st.warning("Low confidence - consider using a clearer image")
            else:
                st.error("❌ Prediction failed. Please try again.")

st.divider()

# --- UI: Retraining ---
st.header("🔄 Retrain Model with New Data")
st.markdown(
    """
    Upload a zip file containing new images to retrain the model. 
    
    **Required structure:**
    ```
    your_data.zip
    ├── Cars/
    │   ├── car1.jpg
    │   ├── car2.jpg
    │   └── ...
    └── Motorcycles/
        ├── bike1.jpg
        ├── bike2.jpg
        └── ...
    ```
    """
)

uploaded_file_retrain = st.file_uploader(
    "Choose a zip file with new data...", 
    type=["zip"],
    key="retrain_uploader"
)

if uploaded_file_retrain:
    st.info(f"📁 File '{uploaded_file_retrain.name}' uploaded ({uploaded_file_retrain.size} bytes)")
    
    # Show file size warning
    if uploaded_file_retrain.size > 50 * 1024 * 1024:  # 50MB
        st.warning("⚠️ Large file detected. Training might take longer or fail due to memory limits.")
    
    if st.button("🚀 Start Retraining", type="primary"):
        with st.spinner("Starting retraining process... This may take several minutes..."):
            retrain_result = trigger_retraining(uploaded_file_retrain)

            if retrain_result:
                st.success(retrain_result.get('message', '✅ Retraining completed!'))

                metrics = retrain_result.get('metrics', {})
                if metrics:
                    st.subheader("📊 Retraining Metrics")
                    
                    # Display metrics in columns
                    cols = st.columns(len(metrics))
                    for i, (metric_name, metric_value) in enumerate(metrics.items()):
                        with cols[i]:
                            st.metric(
                                label=metric_name.replace('_', ' ').title(),
                                value=f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value)
                            )
                    
                    # Plot metrics bar chart
                    if len(metrics) > 1:
                        st.subheader("📈 Metrics Visualization")
                        
                        # Filter numeric metrics for plotting
                        numeric_metrics = {k: v for k, v in metrics.items() 
                                         if isinstance(v, (int, float)) and k != 'loss'}
                        
                        if numeric_metrics:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            metric_names = list(numeric_metrics.keys())
                            metric_values = list(numeric_metrics.values())
                            
                            bars = ax.bar(metric_names, metric_values, 
                                        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
                            ax.set_ylim(0, 1)
                            ax.set_title('Retraining Performance Metrics', fontsize=16, fontweight='bold')
                            ax.set_ylabel('Score', fontsize=12)
                            
                            # Add value labels on bars
                            for bar, value in zip(bars, metric_values):
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                            
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                else:
                    st.info("ℹ️ No detailed metrics available from this training session.")
            else:
                st.error("❌ Retraining failed. Please check your data format and try again.")

# --- Footer ---
st.markdown("---")
st.markdown("**💡 Tips:**")
st.markdown("- Use clear, well-lit images for better predictions")
st.markdown("- Ensure zip files contain proper folder structure for retraining")
st.markdown("- Training time depends on data size and server load")