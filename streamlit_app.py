import streamlit as st
import requests
import base64
from io import BytesIO

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
    """
    Sends an image to the Flask API and returns the prediction.
    """
    try:
        # Encode image to base64
        base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Create the payload
        payload = {"image": base64_encoded_image}
        
        # Send the POST request to the API
        response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=30)
        response.raise_for_status()
        
        # Parse the JSON response
        result = response.json()
        return result.get("predicted_class"), result.get("confidence")

    except requests.exceptions.ConnectionError as e:
        st.error(f"Failed to connect to the Flask API at {PREDICT_ENDPOINT}. Please ensure the API is running.")
        st.error(f"Error: {e}")
        return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred during the API request: {e}")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None, None

def trigger_retraining(zip_file):
    """
    Sends a zip file of new data to the Flask API to trigger retraining.
    """
    try:
        files = {'zip_file': zip_file.getvalue()}
        response = requests.post(RETRAIN_ENDPOINT, files=files, timeout=60)
        response.raise_for_status()

        result = response.json()
        st.success(f"Retraining triggered successfully: {result['message']}")
    except requests.exceptions.ConnectionError as e:
        st.error(f"Failed to connect to the Flask API at {RETRAIN_ENDPOINT}. Please ensure the API is running.")
        st.error(f"Error: {e}")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred during the API request: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# --- UI Layout ---
st.header("Predict a Vehicle")
uploaded_file_predict = st.file_uploader("Choose a vehicle image...", type=["jpg", "jpeg", "png"])

if uploaded_file_predict is not None:
    st.image(uploaded_file_predict, caption='Uploaded Image', use_column_width=True)
    image_bytes = uploaded_file_predict.read()
    
    predict_button = st.button("Predict")
    if predict_button:
        with st.spinner("Making prediction..."):
            predicted_class, confidence = get_prediction(image_bytes)
            
            if predicted_class and confidence is not None:
                st.success("Prediction complete!")
                st.write(f"### Prediction: {predicted_class}")
                st.write(f"### Confidence: **{confidence:.2%}**")

st.divider()

st.header("Retrain Model with New Data")
st.markdown("Upload a zip file containing new images to retrain the model. The zip file should be structured with folders for each class (e.g., `Cars`, `Motorcycles`).")
uploaded_file_retrain = st.file_uploader("Choose a zip file with new data...", type=["zip"])

if uploaded_file_retrain is not None:
    st.info(f"File '{uploaded_file_retrain.name}' uploaded. Click the button to start retraining.")
    
    retrain_button = st.button("Start Retraining")
    if retrain_button:
        with st.spinner("Starting retraining process..."):
            trigger_retraining(uploaded_file_retrain)
