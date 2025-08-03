import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io

# Constants
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'vehicle_classifier_model.keras')
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Load model once at module level
model = None
class_names = None

def load_model():
    """Load the trained model."""
    global model, class_names
    try:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully for prediction.")
            
            # Try to infer class names from model or use defaults
            try:
                # If you have class names saved elsewhere, load them here
                class_names = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']  # Default classes
            except:
                class_names = ['unknown']
                
        else:
            print(f"Model not found at {MODEL_PATH}")
            # Create a dummy model for fallback
            model = create_dummy_model()
            class_names = ['unknown']
    except Exception as e:
        print(f"Error loading model: {e}")
        model = create_dummy_model()
        class_names = ['unknown']

def create_dummy_model():
    """Create a dummy model for fallback."""
    dummy_model = keras.Sequential([
        keras.layers.Conv2D(1, (1, 1), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    dummy_model.compile(optimizer='adam', loss='binary_crossentropy')
    return dummy_model

def preprocess_image(image_bytes):
    """
    Preprocess image bytes for prediction.
    
    Args:
        image_bytes (bytes): Raw image bytes
        
    Returns:
        np.ndarray: Preprocessed image array
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize pixel values to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise ValueError(f"Could not process image: {str(e)}")

def make_prediction(image_bytes):
    """
    Make a prediction on the given image bytes.
    
    Args:
        image_bytes (bytes): Raw image bytes
        
    Returns:
        tuple: (predicted_class, confidence)
    """
    global model, class_names
    
    # Load model if not already loaded
    if model is None:
        load_model()
    
    try:
        # Validate input
        if not image_bytes:
            raise ValueError("Empty image bytes provided")
        
        if len(image_bytes) < 100:  # Minimum reasonable file size
            raise ValueError("Image bytes too small, likely corrupted")
        
        # Check for null bytes in the beginning of the file (could indicate corruption)
        if b'\x00' in image_bytes[:50]:
            print("Warning: Null bytes detected in image header")
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Handle different model outputs
        if len(predictions.shape) == 2 and predictions.shape[1] > 1:
            # Multi-class classification
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_index])
            predicted_class = class_names[predicted_class_index] if predicted_class_index < len(class_names) else 'unknown'
        else:
            # Binary classification
            confidence = float(predictions[0][0])
            predicted_class = class_names[0] if confidence > 0.5 else 'background'
        
        # Ensure confidence is a valid float
        confidence = max(0.0, min(1.0, confidence))
        
        print(f"Prediction: {predicted_class}, Confidence: {confidence:.4f}")
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Prediction error: {e}")
        # Return a safe fallback
        return "error", 0.0

# Load model when module is imported
load_model()