import os
import numpy as np
import logging
import base64
import binascii
import io
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

# Try to import TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    logger.info("TensorFlow imported successfully")
except ImportError as e:
    logger.error(f"TensorFlow import failed: {e}")
    TF_AVAILABLE = False

# Try to import PIL with error handling
try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
    logger.info("PIL imported successfully")
except ImportError as e:
    logger.error(f"PIL import failed: {e}")
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
            keras.layers.Dense(1, activation='sigmoid')
        ])
        dummy_model.compile(optimizer='adam', loss='binary_crossentropy')
        logger.info("Created dummy model")
        return dummy_model
    except Exception as e:
        logger.error(f"Error creating dummy model: {e}")
        return None

def load_model():
    """Load the trained model with comprehensive error handling."""
    global model
    
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available, cannot load model")
        return False
    
    try:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            logger.info("Model loaded successfully for prediction.")
            return True
        else:
            logger.warning(f"Model not found at {MODEL_PATH}, creating dummy model")
            model = create_dummy_model()
            return model is not None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = create_dummy_model()
        return model is not None


def simple_preprocess_image(image_input):
    """
    Robust image preprocessing:
    - Accepts bytes or base64 string input
    - Validates and fixes base64 padding if needed
    - Loads and resizes image
    - Converts to normalized numpy array with batch dimension
    """
    if not PIL_AVAILABLE:
        logger.error("PIL not available for image processing")
        return None

    try:
        # Decode base64 if input is string
        if isinstance(image_input, str):
            missing_padding = len(image_input) % 4
            if missing_padding != 0:
                image_input += "=" * (4 - missing_padding)
            try:
                image_bytes = base64.b64decode(image_input, validate=True)
            except (binascii.Error, ValueError) as decode_err:
                logger.error(f"Base64 decode error: {decode_err}")
                return None
        elif isinstance(image_input, bytes):
            image_bytes = image_input
        else:
            logger.error(f"Invalid input type: {type(image_input)}")
            return None

        if not image_bytes or len(image_bytes) < 50:
            logger.error("Image data too small or empty")
            return None

        clean_bytes = image_bytes.replace(b'\x00', b'')

        image_stream = io.BytesIO(clean_bytes)
        image = Image.open(image_stream)

        # Try to verify the image
        try:
            image.verify()
            image_stream.seek(0)
            image = Image.open(image_stream)
        except Exception:
            image = Image.open(io.BytesIO(clean_bytes))  # fallback

        if image.format not in ["JPEG", "PNG", "GIF"]:
            logger.warning(f"Image format '{image.format}' not standard, attempting to continue")

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
        image_array = np.array(image).astype(np.float32) / 255.0

        if image_array.shape != (IMG_HEIGHT, IMG_WIDTH, 3):
            logger.error(f"Unexpected image shape: {image_array.shape}")
            return None

        image_array = np.expand_dims(image_array, axis=0)
        logger.info(f"Image preprocessed successfully: shape {image_array.shape}")
        return image_array

    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        try:
            debug_path = os.path.join(PROJECT_ROOT, "failed_image_debug.jpg")
            with open(debug_path, "wb") as f:
                f.write(image_bytes if 'image_bytes' in locals() else b'')
            logger.info(f"Saved invalid image bytes to {debug_path} for inspection.")
        except Exception as dump_error:
            logger.error(f"Failed to save debug image: {dump_error}")
        return None


def make_prediction(image_input):
    """
    Make a prediction with extensive error handling.
    """
    global model

    try:
        if model is None:
            if not load_model():
                logger.error("Could not load model")
                return "error_no_model", 0.0

        if isinstance(image_input, str):
            try:
                image_bytes = base64.b64decode(image_input)
                logger.info(f"Base64 image decoded successfully ({len(image_bytes)} bytes)")
            except Exception as e:
                logger.error(f"Failed to decode base64 image: {e}")
                return "error_base64_decode", 0.0
        elif isinstance(image_input, bytes):
            image_bytes = image_input
        else:
            logger.error(f"Invalid image_input type: {type(image_input)}")
            return "error_invalid_type", 0.0

        logger.info(f"Processing image with {len(image_bytes)} bytes")
        processed_image = simple_preprocess_image(image_bytes)
        if processed_image is None:
            logger.error("Image preprocessing failed")
            return "error_preprocessing", 0.0

        if not TF_AVAILABLE or model is None:
            logger.warning("Model not available, returning random prediction")
            return "unknown", 0.5

        try:
            predictions = model.predict(processed_image, verbose=0)
            logger.info(f"Model prediction successful, shape: {predictions.shape}")

            if len(predictions.shape) == 2:
                if predictions.shape[1] > 1:
                    predicted_index = np.argmax(predictions[0])
                    confidence = float(predictions[0][predicted_index])
                    predicted_class = class_names[min(predicted_index, len(class_names) - 1)]
                else:
                    confidence = float(predictions[0][0])
                    predicted_class = class_names[0] if confidence > 0.5 else "background"
            else:
                confidence = float(predictions[0])
                predicted_class = class_names[0] if confidence > 0.5 else "background"

            confidence = max(0.0, min(1.0, confidence))
            logger.info(f"Prediction: {predicted_class}, Confidence: {confidence:.4f}")
            return str(predicted_class), float(confidence)

        except Exception as pred_error:
            logger.error(f"Model prediction error: {pred_error}")
            return "error_prediction", 0.0

    except Exception as e:
        logger.error(f"Unexpected error in make_prediction: {e}")
        return "error_unexpected", 0.0

# Initialize model on module import
try:
    load_model()
except Exception as e:
    logger.error(f"Error during module initialization: {e}")
