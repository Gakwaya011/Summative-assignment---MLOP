import streamlit as st
import os
import sys
from PIL import Image
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from predictor import load_model, predict_image, CLASS_NAMES

# Define the model path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'vehicle_classifier_model.keras')

# Load the model once at the beginning
@st.cache_resource
def load_cached_model():
    """Loads the model and caches it to prevent reloading on every interaction."""
    try:
        return load_model(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {MODEL_PATH}. Please run src/train.py first.")
        return None

model = load_cached_model()

st.title("Vehicle Classifier")
st.write("Upload an image of a car or a motorcycle to see the prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    if model:
        # Streamlit needs a file path, so we save the upload temporarily
        temp_file_path = os.path.join(PROJECT_ROOT, "temp_image.jpg")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Make the prediction
        predicted_class, confidence = predict_image(model, temp_file_path)

        # Clean up the temporary file
        os.remove(temp_file_path)

        # Display the result
        st.success(f"Prediction: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}**")