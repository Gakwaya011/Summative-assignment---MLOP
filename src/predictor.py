import os
import numpy as np
import logging
from PIL import Image
import io
import traceback

# Configure logging
logger = logging.getLogger(__name__)

# Try to import TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    logger.info("TensorFlow imported successfully.")
except ImportError as e:
    logger.error(f"TensorFlow import failed: {e}", exc_info=True)
    TF_AVAILABLE = False

# Try to import PIL with error handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
    logger.info("PIL imported successfully.")
except ImportError as e:
    logger.error(f"PIL import failed: {e}", exc_info=True)
    PIL_AVAILABLE = False

# Constants
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'vehicle_classifier_model.keras')
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Global variables
model = None
class_names = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'unknown']

def create_dummy_model():
    """Create a simple dummy model for testing."""
    if not TF_AVAILABLE:
        return None
    
    try:
        dummy_model = keras.Sequential([
            keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(len(class_names) - 1, activation='softmax')
        ])
        dummy_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        logger.info("Created dummy model.")
        return dummy_model
    except Exception as e:
        logger.error(f"Error creating dummy model: {e}", exc_info=True)
        return None

def load_model():
    """Load the trained model with comprehensive error handling."""
    global model
    
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available, cannot load model.")
        return False
    
    try:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            logger.info("Model loaded successfully for prediction.")
            return True
        else:
            logger.warning(f"Model not found at {MODEL_PATH}, creating dummy model.")
            model = create_dummy_model()
            return model is not None
    except Exception as e:
        logger.error(f"Error loading model from {MODEL_PATH}: {e}", exc_info=True)
        model = create_dummy_model()
        return model is not None

def simple_preprocess_image(image_bytes):
    """
    Robust image preprocessing function.
    
    Args:
        image_bytes (bytes): Raw image bytes.
        
    Returns:
        np.ndarray or None: Preprocessed image array.
    """
    if not PIL_AVAILABLE:
        logger.error("PIL not available for image processing.")
        return None
    
    try:
        if not image_bytes or len(image_bytes) < 50:
            raise ValueError("Invalid or empty image data.")
        
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
        
        image_array = np.array(image)
        
        if image_array.shape != (IMG_HEIGHT, IMG_WIDTH, 3):
            raise ValueError(f"Unexpected image shape: {image_array.shape}")
        
        image_array = image_array.astype(np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        logger.info(f"Image preprocessed successfully: shape {image_array.shape}.")
        return image_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}", exc_info=True)
        return None

def make_prediction(image_bytes):
    """
    Make a prediction with extensive error handling.
    
    Args:
        image_bytes (bytes): Raw image bytes.
        
    Returns:
        tuple: (predicted_class, confidence)
    """
    global model

    try:
        if model is None:
            if not load_model():
                logger.error("Could not load model.")
                return "error_no_model", 0.0

        if not isinstance(image_bytes, bytes) or not image_bytes:
            logger.error(f"Invalid or empty image_bytes provided.")
            return "error_no_data", 0.0

        processed_image = simple_preprocess_image(image_bytes)
        if processed_image is None:
            logger.error("Image preprocessing failed.")
            return "error_preprocessing", 0.0

        predictions = model.predict(processed_image, verbose=0)
        logger.info(f"Prediction output shape: {predictions.shape}")

        if len(predictions.shape) == 2 and predictions.shape[1] == len(class_names) - 1:
            predicted_index = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_index])
            predicted_class = class_names[predicted_index]
        else:
            logger.error(f"Unexpected prediction shape or output: {predictions.shape}")
            return "error_model_type_mismatch", 0.0

        confidence = round(max(0.0, min(1.0, confidence)), 4)
        logger.info(f"Predicted class: {predicted_class}, Confidence: {confidence}")
        return predicted_class, confidence

    except Exception as e:
        logger.error(f"Unexpected error in make_prediction: {e}", exc_info=True)
        return "error_unexpected", 0.0

try:
    load_model()
except Exception as e:
    logger.error(f"Error during module initialization: {e}", exc_info=True)
