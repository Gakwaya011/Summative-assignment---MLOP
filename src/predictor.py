import os
import numpy as np
import logging

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

def simple_preprocess_image(image_bytes):
    """
    Simple image preprocessing that's less likely to fail.
    
    Args:
        image_bytes (bytes): Raw image bytes
        
    Returns:
        np.ndarray or None: Preprocessed image array
    """
    if not PIL_AVAILABLE:
        logger.error("PIL not available for image processing")
        return None
    
    try:
        # Basic validation
        if not image_bytes or len(image_bytes) < 50:
            raise ValueError("Invalid image data")
        
        # Remove any null bytes that might cause issues
        clean_bytes = image_bytes.replace(b'\x00', b'')
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(clean_bytes))
        
        # Optional format check (JPEG/PNG)
        if image.format not in ["JPEG", "PNG"]:
            raise ValueError(f"Unsupported image format: {image.format}")
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Validate array shape
        if image_array.shape != (IMG_HEIGHT, IMG_WIDTH, 3):
            raise ValueError(f"Unexpected image shape: {image_array.shape}")
        
        # Normalize pixel values
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        logger.info(f"Image preprocessed successfully: shape {image_array.shape}")
        return image_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        
        # Dump image bytes for debugging
        try:
            debug_path = os.path.join(PROJECT_ROOT, "failed_image_debug.jpg")
            with open(debug_path, "wb") as f:
                f.write(image_bytes)
            logger.info(f"Saved invalid image bytes to {debug_path} for inspection.")
        except Exception as dump_error:
            logger.error(f"Failed to save debug image: {dump_error}")
        
        return None

def make_prediction(image_bytes):
    """
    Make a prediction with extensive error handling.
    
    Args:
        image_bytes (bytes): Raw image bytes
        
    Returns:
        tuple: (predicted_class, confidence)
    """
    global model
    
    try:
        # Load model if not already loaded
        if model is None:
            if not load_model():
                logger.error("Could not load model")
                return "error_no_model", 0.0
        
        # Basic input validation
        if not image_bytes:
            logger.error("No image bytes provided")
            return "error_no_data", 0.0
        
        if not isinstance(image_bytes, bytes):
            logger.error(f"Invalid image_bytes type: {type(image_bytes)}")
            return "error_invalid_type", 0.0
        
        logger.info(f"Processing image with {len(image_bytes)} bytes")
        
        # Preprocess image
        processed_image = simple_preprocess_image(image_bytes)
        if processed_image is None:
            logger.error("Image preprocessing failed")
            return "error_preprocessing", 0.0
        
        # Make prediction
        if not TF_AVAILABLE or model is None:
            logger.warning("Model not available, returning random prediction")
            return "unknown", 0.5
        
        try:
            predictions = model.predict(processed_image, verbose=0)
            logger.info(f"Model prediction successful, shape: {predictions.shape}")
            
            # Handle different prediction formats
            if len(predictions.shape) == 2:
                if predictions.shape[1] > 1:
                    # Multi-class classification
                    predicted_index = np.argmax(predictions[0])
                    confidence = float(predictions[0][predicted_index])
                    predicted_class = class_names[min(predicted_index, len(class_names) - 1)]
                else:
                    # Binary classification
                    confidence = float(predictions[0][0])
                    predicted_class = class_names[0] if confidence > 0.5 else "background"
            else:
                # Single prediction
                confidence = float(predictions[0])
                predicted_class = class_names[0] if confidence > 0.5 else "background"
            
            # Ensure valid confidence range
            confidence = max(0.0, min(1.0, confidence))
            
            logger.info(f"Prediction: {predicted_class}, Confidence: {confidence:.4f}")
            return str(predicted_class), float(confidence)
            
        except Exception as pred_error:
            logger.error(f"Model prediction error: {pred_error}")
            return "error_prediction", 0.0
        
    except Exception as e:
        logger.error(f"Unexpected error in make_prediction: {e}")
        return "error_unexpected", 0.0

# Initialize model when module is imported
try:
    load_model()
except Exception as e:
    logger.error(f"Error during module initialization: {e}")
