import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Define common parameters to ensure consistency
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ['Cars', 'Motorcycles'] # This should match your data classes

def load_model(model_path: str):
    """
    Loads a saved Keras model.
    """
    print(f"Loading model from: {model_path}")
    
    # Custom objects are required if the model uses custom metrics, layers, or regularizers
    custom_objects = {
        'Precision': tf.keras.metrics.Precision,
        'Recall': tf.keras.metrics.Recall,
        'AUC': tf.keras.metrics.AUC,
        'l2': tf.keras.regularizers.l2
    }
    
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    print("Model loaded successfully!")
    return model

def preprocess_image(image_path: str):
    """
    Loads and preprocesses a single image for prediction.
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0  # Rescale to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image(model: keras.Model, image_path: str):
    """
    Makes a prediction on a single image.
    """
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    
    # The output is a single value from the sigmoid layer
    prediction_value = prediction[0][0]
    
    # Map the prediction value to a class label and confidence
    if prediction_value >= 0.5:
        predicted_class_index = 1 # Motorcycles
        confidence = prediction_value
    else:
        predicted_class_index = 0 # Cars
        confidence = 1 - prediction_value
        
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    
    return predicted_class_name, confidence

if __name__ == "__main__":
    # Example usage for testing the predictor
    print("Running predictor.py directly for a test...")
    
    # Define paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    model_path = os.path.join(project_root, 'models', 'vehicle_classifier_model.keras')
    
    # --- IMPORTANT ---
    # You need to provide a path to a sample image for this test.
    # Replace 'path/to/your/sample_image.jpg' with a real image path.
    # For example: os.path.join(project_root, 'data', 'test', 'Cars', 'car_001.jpg')
    sample_image_path = os.path.join(project_root, 'data', 'test', 'Cars', 'Car (1).jpg')
    
    # Load the model and make a prediction
    try:
        loaded_model = load_model(model_path)
        predicted_class, confidence = predict_image(loaded_model, sample_image_path)
        
        print("\n--- Prediction Results ---")
        print(f"Sample Image: {sample_image_path}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}")
    except FileNotFoundError:
        print(f"\nError: Could not find model at {model_path} or image at {sample_image_path}.")
        print("Please ensure the paths are correct and the files exist.")