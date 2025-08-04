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
        
        logger.info(f"Original image bytes length: {len(image_bytes)}")
        logger.info(f"First 20 bytes (hex): {image_bytes[:20].hex()}")
        
        # Check if bytes start with data URL prefix and strip it
        if isinstance(image_bytes, str):
            # If somehow we got a string instead of bytes
            if image_bytes.startswith('data:image'):
                image_bytes = image_bytes.split(',')[1]
            # Convert base64 string to bytes
            import base64
            image_bytes = base64.b64decode(image_bytes)
        
        # Validate image format by checking magic bytes
        if not (image_bytes.startswith(b'\xff\xd8\xff') or  # JPEG
                image_bytes.startswith(b'\x89PNG\r\n\x1a\n') or  # PNG
                image_bytes.startswith(b'GIF8')):  # GIF
            logger.warning(f"Unknown image format. First 10 bytes: {image_bytes[:10].hex()}")
            # Don't fail immediately, let PIL try to handle it
        
        # Try to create PIL Image with better error handling
        try:
            image_stream = io.BytesIO(image_bytes)
            image = Image.open(image_stream)
            # Force load the image to catch any format issues early
            image.load()
            logger.info(f"Successfully opened image: {image.format}, {image.mode}, {image.size}")
        except Exception as pil_error:
            logger.error(f"PIL failed to open image: {pil_error}")
            # Try alternative approach - save to temp file and reload
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_file.write(image_bytes)
                    temp_file.flush()
                    image = Image.open(temp_file.name)
                    image.load()
                    os.unlink(temp_file.name)  # Clean up temp file
                    logger.info("Successfully opened image using temp file method")
            except Exception as temp_error:
                logger.error(f"Temp file method also failed: {temp_error}")
                raise pil_error  # Re-raise original error
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info(f"Converted image to RGB mode")
        
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
        
        # Enhanced debugging - save image bytes for inspection
        try:
            debug_path = os.path.join(PROJECT_ROOT, "failed_image_debug.bin")
            with open(debug_path, "wb") as f:
                f.write(image_bytes)
            logger.info(f"Saved invalid image bytes to {debug_path} for inspection.")
            
            # Also try to save as different formats for debugging
            for ext in ['.jpg', '.png', '.gif']:
                try:
                    debug_path_ext = os.path.join(PROJECT_ROOT, f"failed_image_debug{ext}")
                    with open(debug_path_ext, "wb") as f:
                        f.write(image_bytes)
                except:
                    pass
                    
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

        if not isinstance(image_bytes, (bytes, str)):
            logger.error(f"Invalid image_bytes type: {type(image_bytes)}")
            return "error_invalid_type", 0.0

        logger.info(f"Processing image with {len(image_bytes)} bytes")

        # Preprocess image
        processed_image = simple_preprocess_image(image_bytes)
        if processed_image is None:
            logger.error("Image preprocessing failed")
            return "error_preprocessing", 0.0

        # Model check
        if not TF_AVAILABLE or model is None:
            logger.warning("Model not available, returning unknown")
            return "unknown", 0.5

        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        logger.info(f"Prediction output: {predictions}")

        # Interpret prediction based on output shape
        if len(predictions.shape) == 2 and predictions.shape[1] > 1:
            # Multi-class classification (softmax)
            predicted_index = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_index])
            predicted_class = (
                class_names[predicted_index] if predicted_index < len(class_names)
                else "unknown"
            )

        elif len(predictions.shape) == 2 and predictions.shape[1] == 1:
            # Binary classification (sigmoid)
            confidence = float(predictions[0][0])
            predicted_class = class_names[0] if confidence > 0.5 else class_names[-1]

        elif len(predictions.shape) == 1:
            # Single-output fallback (shouldn't happen ideally)
            confidence = float(predictions[0])
            predicted_class = class_names[0] if confidence > 0.5 else class_names[-1]

        else:
            logger.error(f"Unexpected prediction shape: {predictions.shape}")
            return "error_model_type_mismatch", 0.0

        confidence = round(max(0.0, min(1.0, confidence)), 4)
        logger.info(f"Predicted class: {predicted_class}, Confidence: {confidence}")
        return predicted_class, confidence

    except Exception as e:
        logger.error(f"Unexpected error in make_prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return "error_unexpected", 0.0


# Initialize model when module is imported
try:
    load_model()
except Exception as e:
    logger.error(f"Error during module initialization: {e}")