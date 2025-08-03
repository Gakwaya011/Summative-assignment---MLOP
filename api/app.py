import os
import base64
import shutil
import zipfile
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import sys

# Add the root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

# Try different import approaches
try:
    from src.predictor import make_prediction
    from src.train import retrain_model
except ImportError as e:
    print(f"Import error with src.module: {e}")
    try:
        # Alternative import if running from different directory
        import sys
        sys.path.append(os.path.join(project_root, 'src'))
        from predictor import make_prediction
        from train import retrain_model
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        # Last resort - direct path import
        import importlib.util
        
        # Load predictor
        predictor_path = os.path.join(project_root, 'src', 'predictor.py')
        spec = importlib.util.spec_from_file_location("predictor", predictor_path)
        predictor_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predictor_module)
        make_prediction = predictor_module.make_prediction
        
        # Load train
        train_path = os.path.join(project_root, 'src', 'train.py')
        spec = importlib.util.spec_from_file_location("train", train_path)
        train_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_module)
        retrain_model = train_module.retrain_model

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({'message': 'Vehicle Classifier API is running.'})

@app.route('/predict', methods=['POST'])
def predict():
    image_bytes = None
    if 'file' in request.files:
        file = request.files['file']
        image_bytes = file.read()
    elif request.is_json and 'image' in request.json:
        try:
            image_b64 = request.json['image']
            image_bytes = base64.b64decode(image_b64)
        except Exception as e:
            return jsonify({'error': f'Error decoding base64 image: {str(e)}'}), 400
    if image_bytes is None:
        return jsonify({'error': "No image data provided. Expecting 'file' in form-data or 'image' in JSON."}), 400
    try:
        predicted_class, confidence = make_prediction(image_bytes)
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    if 'zip_file' not in request.files:
        return jsonify({'error': 'No zip_file in request.files'}), 400
    zip_file = request.files['zip_file']
    filename = secure_filename(zip_file.filename)
    if not filename.endswith('.zip'):
        return jsonify({'error': 'Uploaded file must be a .zip file'}), 400
    
    # We'll use a temp dir relative to the current working directory, which should be the project root
    temp_dir = "temp_data"
    os.makedirs(temp_dir, exist_ok=True)
    try:
        zip_file_path = os.path.join(temp_dir, filename)
        zip_file.save(zip_file_path)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        history = retrain_model(temp_dir)
        
        metrics = {
            'accuracy': history.history['accuracy'][-1],
            'val_accuracy': history.history['val_accuracy'][-1],
            'loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1],
            'precision': history.history.get('precision')[-1] if 'precision' in history.history else None,
            'recall': history.history.get('recall')[-1] if 'recall' in history.history else None,
            'roc_auc': history.history.get('roc_auc')[-1] if 'roc_auc' in history.history else None
        }
        return jsonify({'message': 'Retraining completed successfully.', 'metrics': metrics})
    except Exception as e:
        return jsonify({'error': f'Retraining failed: {str(e)}'}), 500
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)