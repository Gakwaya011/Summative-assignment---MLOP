import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Add the project root to the system path
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))
sys.path.append(project_root)

# Correct pathing to the model
MODEL_PATH = os.path.join(project_root, 'models', 'vehicle_classifier_model.keras')
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ['Car', 'Motorcycle']

# Load the model once globally to avoid reloading on every request
try:
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully for prediction.")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    model = None

def make_prediction(image_file):
    """
    Makes a prediction on a single image.
    Args:
        image_file (str or file-like object): The image to predict.
    Returns:
        tuple: (predicted_class, confidence)
    """
    if model is None:
        return "Error", 0.0

    try:
        # Load and preprocess the image
        img = load_img(image_file, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        # Make prediction
        # The model's final layer is sigmoid, so predictions is a probability score
        predictions = model.predict(img_array)
        score = float(predictions[0][0])
        
        # Determine the predicted class and confidence
        if score > 0.5:
            predicted_class = CLASS_NAMES[1]  # Motorcycle
            confidence = score
        else:
            predicted_class = CLASS_NAMES[0]  # Car
            confidence = 1.0 - score

        return predicted_class, confidence

    except Exception as e:
        print(f"Prediction failed: {e}")
        return "Prediction Error", 0.0

if __name__ == '__main__':
    # Example usage for local testing
    example_image_path = os.path.join(project_root, 'data', 'test', 'Cars', '2016-honda-accord-coupe-ex-l-v-6-review-0.jpg')
    if os.path.exists(example_image_path):
        predicted_class, confidence = make_prediction(example_image_path)
        print(f"Prediction for {example_image_path}: {predicted_class} with confidence {confidence:.2f}")
    else:
        print(f"Test image not found at {example_image_path}")

