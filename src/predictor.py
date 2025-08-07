import os
import requests
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Constants
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ['Car', 'Motorcycle']

# Model file setup
MODEL_URL = "https://huggingface.co/Brillant011/vehicle-classifier-model/resolve/main/vehicle_classifier_model.tflite"
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))
MODEL_PATH = os.path.join(project_root, 'models', 'vehicle_classifier_model.tflite')

# Download model from Hugging Face if not present
def download_model_if_missing():
    if not os.path.exists(MODEL_PATH):
        print("Downloading TFLite model from Hugging Face...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        try:
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("TFLite model downloaded successfully.")
        except Exception as e:
            print(f"Failed to download model: {e}")

# Load TFLite model once globally
try:
    download_model_if_missing()
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite model loaded successfully.")
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    interpreter = None


def preprocess_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def make_prediction(image_file):
    if interpreter is None:
        return "Error", 0.0

    try:
        img_array = preprocess_image(image_file)
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        score = predictions[0][0]

        print(f"Prediction raw score (sigmoid output): {score:.4f}")

        if score > 0.5:
            predicted_class = CLASS_NAMES[1]
            confidence = score
        else:
            predicted_class = CLASS_NAMES[0]
            confidence = 1 - score

        return predicted_class, float(confidence)

    except Exception as e:
        print(f"Prediction failed: {e}")
        return "Prediction Error", 0.0


# Optional: Keep this if you're retraining locally
from src.train import retrain_model

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


# Local test
if __name__ == '__main__':
    example_image_path = os.path.join(project_root, 'data', 'test', 'Cars', '2016-honda-accord-coupe-ex-l-v-6-review-0.jpg')
    if os.path.exists(example_image_path):
        pred_class, conf = make_prediction(example_image_path)
        print(f"Prediction for {example_image_path}: {pred_class} with confidence {conf:.2f}")
    else:
        print(f"Test image not found at {example_image_path}")
