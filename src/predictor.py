import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Constants
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ['Car', 'Motorcycle']

# Set your model path here — adjust as needed
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))
MODEL_PATH = os.path.join(project_root, 'models', 'vehicle_classifier_model.keras')

# Load model once globally
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully for prediction.")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    model = None

def make_prediction(image_file):
    if model is None:
        return "Error", 0.0

    try:
        # Load and preprocess the image
        img = load_img(image_file, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize pixel values like training
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

        # Predict
        predictions = model.predict(img_array)
        score = predictions[0][0]  # sigmoid output scalar between 0 and 1

        print(f"Prediction raw score (sigmoid output): {score:.4f}")

        # Threshold at 0.5 for binary classification
        if score > 0.5:
            predicted_class = CLASS_NAMES[1]  # Motorcycle
            confidence = score
        else:
            predicted_class = CLASS_NAMES[0]  # Car
            confidence = 1 - score

        return predicted_class, float(confidence)

    except Exception as e:
        print(f"Prediction failed: {e}")
        return "Prediction Error", 0.0
from src.train import retrain_model  # Make sure retrain_model is imported

def retrain_and_get_metrics(data_dir: str):
    try:
        model, history = retrain_model(data_dir)
        model.save("model.h5")

        metrics = {
            "accuracy": float(history.history.get("accuracy", [0])[-1]),
            "val_accuracy": float(history.history.get("val_accuracy", [0])[-1]),
            "loss": float(history.history.get("loss", [0])[-1]),
            "val_loss": float(history.history.get("val_loss", [0])[-1]),
        }

        return metrics
    except Exception as e:
        print(f"[ERROR] Retraining failed: {e}")
        return {"error": str(e)}



if __name__ == '__main__':
    # Local test example
    example_image_path = os.path.join(project_root, 'data', 'test', 'Cars', '2016-honda-accord-coupe-ex-l-v-6-review-0.jpg')
    if os.path.exists(example_image_path):
        pred_class, conf = make_prediction(example_image_path)
        print(f"Prediction for {example_image_path}: {pred_class} with confidence {conf:.2f}")
    else:
        print(f"Test image not found at {example_image_path}")
