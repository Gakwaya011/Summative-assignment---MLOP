import os
import sys
import base64
import shutil
import zipfile
import subprocess
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from src.predictor import make_prediction

# --- CRITICAL FIX FOR PATHING ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_script_dir)
src_parent_dir = os.path.join(project_root)

if src_parent_dir not in sys.path:
    sys.path.append(src_parent_dir)  # ✅ Fix: Add path to sys.path

# --- Configuration ---
app = Flask(__name__)
CORS(app)  # Enable CORS for the Streamlit front-end

UPLOAD_FOLDER = os.path.join(project_root, 'uploads')
RETAINING_DATA_DIR = os.path.join(project_root, 'retraining_data')
ORIGINAL_TRAIN_DATA_DIR = os.path.join(project_root, 'data', 'train')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RETAINING_DATA_DIR, exist_ok=True)

@app.route('/', methods=['GET'])
def home():
    return "MLOPs API is running!", 200

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.get_json()  # ✅ Fix: Proper indentation
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_data = data['image']
        image_bytes = base64.b64decode(image_data)
        image_file = BytesIO(image_bytes)

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

    if zip_file:
        temp_zip_path = os.path.join(UPLOAD_FOLDER, secure_filename(zip_file.filename))
        zip_file.save(temp_zip_path)

        try:
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(RETAINING_DATA_DIR)

            combined_data_dir = os.path.join(project_root, 'combined_data_for_retraining')
            if os.path.exists(combined_data_dir):
                shutil.rmtree(combined_data_dir)
            shutil.copytree(ORIGINAL_TRAIN_DATA_DIR, combined_data_dir)

            for class_name in ['Cars', 'Motorcycles']:
                source_dir = os.path.join(RETAINING_DATA_DIR, class_name)
                target_dir = os.path.join(combined_data_dir, class_name)
                if os.path.exists(source_dir):
                    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

            print(f"Starting retraining process with data from {combined_data_dir}")
            subprocess.Popen(
                ['python', 'src/train.py'],
                cwd=project_root
            )

            return jsonify({'message': 'Retraining process started successfully. This may take some time.'}), 202

        except Exception as e:
            return jsonify({'error': str(e)}), 500

        finally:
            if os.path.exists(temp_zip_path):
                os.remove(temp_zip_path)
            if os.path.exists(RETAINING_DATA_DIR):
                shutil.rmtree(RETAINING_DATA_DIR)

if __name__ == '__main__':
    pass
