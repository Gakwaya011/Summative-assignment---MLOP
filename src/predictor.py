import os
import numpy as np
import logging
from PIL import Image
import io
import traceback
import time
from functools import lru_cache
# Add this at the VERY TOP of predictor.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None  # Critical fallback

# ========== CONFIGURATION ==========
# Disable GPU (Critical for Render's free tier)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configure logging - reduced to WARNING for production
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'vehicle_classifier_model.keras')
IMG_HEIGHT, IMG_WIDTH = 128, 128
MAX_PREDICTION_TIME = 10  # seconds

# ========== MODEL MANAGEMENT ==========
class ModelManager:
    _instance = None
    class_names = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'unknown']
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
            cls._instance.last_load_time = 0
        return cls._instance
    
    def load_model(self):
        """Load model with memory optimization"""
        try:
            # Clear previous session to free memory
            tf.keras.backend.clear_session()
            
            # Load with custom options for efficiency
            self.model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False  # Faster loading if not retraining
            )
            logger.warning("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Model load failed: {str(e)}")
            self.model = self._create_dummy_model()
            return self.model is not None
    
    def _create_dummy_model(self):
        """Lightweight fallback model"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(len(self.class_names) - 1, activation='softmax')
            ])
            logger.warning("Created dummy model")
            return model
        except Exception as e:
            logger.error(f"Dummy model failed: {str(e)}")
            return None

# ========== IMAGE PROCESSING ==========
@lru_cache(maxsize=1)
def get_image_processor():
    """Cache image processor to save memory"""
    return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

def preprocess_image(image_bytes):
    """Optimized image preprocessing with timeout"""
    start_time = time.time()
    
    try:
        # Fast validation
        if len(image_bytes) < 1024:  # Minimum reasonable image size
            raise ValueError("Image too small")
        
        # Convert bytes to PIL with size check
        img = Image.open(io.BytesIO(image_bytes))
        if img.size[0] * img.size[1] > 3000*3000:  # Prevent giant images
            img = img.resize((3000, 3000))
        
        # Convert and resize
        img = img.convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img) / 255.0
        
        # Timeout check
        if (time.time() - start_time) > MAX_PREDICTION_TIME/2:
            raise TimeoutError("Preprocessing too slow")
            
        return np.expand_dims(img_array, axis=0)
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        return None

# ========== PREDICTION LOGIC ==========
def predict_with_timeout(image_bytes):
    """Main prediction with strict timeout"""
    model_mgr = ModelManager()
    
    # Validate input first
    if not isinstance(image_bytes, (bytes, bytearray)):
        return "invalid_input", 0.0
    
    # Get model (load if needed)
    if model_mgr.model is None and not model_mgr.load_model():
        return "model_unavailable", 0.0
    
    # Preprocess with timeout
    start_time = time.time()
    processed_img = preprocess_image(image_bytes)
    if processed_img is None:
        return "preprocessing_failed", 0.0
    
    # Prediction with timeout check
    try:
        if (time.time() - start_time) > MAX_PREDICTION_TIME * 0.8:
            raise TimeoutError("Prediction would exceed timeout")
            
        predictions = model_mgr.model.predict(
            processed_img,
            batch_size=1,  # Smallest possible batch
            verbose=0       # Disable progress prints
        )
        
        # Process results
        pred_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_idx])
        return model_mgr.class_names[pred_idx], round(confidence, 4)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return "prediction_error", 0.0

# Initialize on import
model_manager = ModelManager()
model_manager.load_model()