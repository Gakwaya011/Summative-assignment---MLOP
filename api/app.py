# --- File: api/app.py ---
import os
import sys
import base64
import shutil
import zipfile
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# --- Path Setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))
sys.path.append(project_root)

# Imports after path fix
from src.predictor import make_prediction
from src.train import retrain_model

# --- Flask Setup ---
app = Flask(__name__)
CORS(app)

# Directories
UPLOAD_FOLDER = os.path.join(project_root, 'uploads')
RETRAINING_DATA_DIR = os.path.join(project_root, 'retraining_data')
ORIGINAL_TRAIN_DATA_DIR = os.path.join(project_root, 'data', 'train')
COMBINED_DATA_DIR = os.path.join(project_root, 'combined_data_for_retraining')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RETRAINING_DATA_DIR, exist_ok=True)

@app.route('/', methods=['GET'])
def home():
    return "MLOPs API is running!", 200

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_data = base64.b64decode(data['image'])
        image_file = BytesIO(image_data)

        predicted_class, confidence = make_prediction(image_file)

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": float(confidence)
        }), 200

    except Exception as e:
        print(f"Prediction endpoint error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    if 'zip_file' not in request.files:
        return jsonify({'error': 'No zip file part in the request'}), 400

    zip_file = request.files['zip_file']
    if zip_file.filename == '':
        return jsonify({'error': 'No selected zip file'}), 400

    temp_zip_path = os.path.join(UPLOAD_FOLDER, secure_filename(zip_file.filename))
    zip_file.save(temp_zip_path)

    try:
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(RETRAINING_DATA_DIR)

        # Reset combined training directory
        if os.path.exists(COMBINED_DATA_DIR):
            shutil.rmtree(COMBINED_DATA_DIR)
        shutil.copytree(ORIGINAL_TRAIN_DATA_DIR, COMBINED_DATA_DIR)

        # Merge new data with original
        for class_name in ['Car', 'Motorcycle']:
            src_dir = os.path.join(RETRAINING_DATA_DIR, class_name)
            tgt_dir = os.path.join(COMBINED_DATA_DIR, class_name)
            if os.path.exists(src_dir):
                shutil.copytree(src_dir, tgt_dir, dirs_exist_ok=True)

        # Retrain model and get training history
        history = retrain_model(COMBINED_DATA_DIR)

        # Collect metrics
        metrics = {
            'final_accuracy': float(history.history.get('accuracy', [0])[-1]),
            'final_val_accuracy': float(history.history.get('val_accuracy', [0])[-1]),
            'final_loss': float(history.history.get('loss', [0])[-1]),
            'final_val_loss': float(history.history.get('val_loss', [0])[-1])
        }

        return jsonify({
            'message': 'Retraining completed successfully.',
            'metrics': metrics
        }), 200

    except Exception as e:
        print(f"Retraining error: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up temporary files
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        if os.path.exists(RETRAINING_DATA_DIR):
            shutil.rmtree(RETRAINING_DATA_DIR)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
