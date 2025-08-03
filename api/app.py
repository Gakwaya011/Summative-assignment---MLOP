import os
import sys
import base64
import shutil
import zipfile
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Setup Import Path ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
src_dir = os.path.join(project_root, 'src')

if src_dir not in sys.path:
    sys.path.append(src_dir)

from train import train_model_from_dir
from predictor import make_prediction

# --- Flask Setup ---
app = Flask(__name__)
CORS(app)

# --- Directory Setup ---
UPLOAD_FOLDER = os.path.join(project_root, 'uploads')
RETRAINING_DATA_DIR = os.path.join(project_root, 'retraining_data')
ORIGINAL_TRAIN_DATA_DIR = os.path.join(project_root, 'data', 'train')
COMBINED_DATA_DIR = os.path.join(project_root, 'combined_data_for_retraining')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RETRAINING_DATA_DIR, exist_ok=True)

# --- Health Check ---
@app.route('/', methods=['GET'])
def home():
    return "✅ MLOps API is running!", 200

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_bytes = base64.b64decode(data['image'])
        image_file = BytesIO(image_bytes)

        predicted_class, confidence = make_prediction(image_file)

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": float(confidence)
        }), 200

    except Exception as e:
        print(f"[❌ Prediction error] {e}")
        return jsonify({"error": str(e)}), 500

# --- Retraining Endpoint ---
@app.route('/retrain', methods=['POST'])
def retrain():
    if 'zip_file' not in request.files:
        return jsonify({'error': 'No zip file uploaded'}), 400

    zip_file = request.files['zip_file']
    if zip_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    temp_zip_path = os.path.join(UPLOAD_FOLDER, secure_filename(zip_file.filename))
    zip_file.save(temp_zip_path)

    try:
        # Unzip new data
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(RETRAINING_DATA_DIR)

        # Clean and merge data
        if os.path.exists(COMBINED_DATA_DIR):
            shutil.rmtree(COMBINED_DATA_DIR)
        shutil.copytree(ORIGINAL_TRAIN_DATA_DIR, COMBINED_DATA_DIR)

        # Dynamically merge 'Cars' and 'Motorcycles' from retraining data
        for class_name in ['Cars', 'Motorcycles']:
            found = False
            for root, dirs, _ in os.walk(RETRAINING_DATA_DIR):
                if class_name in dirs:
                    source_dir = os.path.join(root, class_name)
                    target_dir = os.path.join(COMBINED_DATA_DIR, class_name)
                    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
                    found = True
                    break
            if not found:
                print(f"⚠️ Folder '{class_name}' not found in ZIP")

        print(f"✅ Starting retraining from: {COMBINED_DATA_DIR}")
        print(f"Classes in combined data: {os.listdir(COMBINED_DATA_DIR)}")

        history = train_model_from_dir(COMBINED_DATA_DIR)

        metrics = {
            'loss': history.history['loss'][-1],
            'accuracy': history.history['accuracy'][-1],
            'precision': history.history.get('precision', [None])[-1],
            'recall': history.history.get('recall', [None])[-1],
            'roc_auc': history.history.get('roc_auc', [None])[-1]
        }

        # Clean up
        os.remove(temp_zip_path)
        shutil.rmtree(RETRAINING_DATA_DIR)

        return jsonify({
            'message': '✅ Retraining completed successfully.',
            'metrics': {k: v for k, v in metrics.items() if v is not None}
        }), 200

    except Exception as e:
        print(f"[❌ Retraining error] {e}")
        return jsonify({'error': str(e)}), 500

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
