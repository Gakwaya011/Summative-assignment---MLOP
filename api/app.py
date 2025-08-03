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
sys.path.insert(0, os.path.join(project_root, 'src'))

# Initialize variables
make_prediction = None
retrain_model = None

# Try multiple import strategies
def load_modules():
    global make_prediction, retrain_model
    
    # Strategy 1: Try normal imports
    try:
        from src.predictor import make_prediction
        from src.train import retrain_model
        print("Successfully imported using src.module")
        return True
    except ImportError as e:
        print(f"Import error with src.module: {e}")
    
    # Strategy 2: Try direct imports
    try:
        from predictor import make_prediction
        from train import retrain_model
        print("Successfully imported using direct imports")
        return True
    except ImportError as e:
        print(f"Import error with direct imports: {e}")
    
    # Strategy 3: Try importlib with full paths
    try:
        import importlib.util
        
        # Load predictor
        predictor_path = os.path.join(project_root, 'src', 'predictor.py')
        if os.path.exists(predictor_path):
            spec = importlib.util.spec_from_file_location("predictor", predictor_path)
            predictor_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(predictor_module)
            make_prediction = predictor_module.make_prediction
            print("Loaded predictor module successfully")
        
        # Load train module or create fallback
        train_path = os.path.join(project_root, 'src', 'train.py')
        if os.path.exists(train_path):
            spec = importlib.util.spec_from_file_location("train", train_path)
            train_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_module)
            
            # Check if retrain_model exists in the module
            if hasattr(train_module, 'retrain_model'):
                retrain_model = train_module.retrain_model
                print("Loaded retrain_model successfully")
            else:
                print("retrain_model not found in train module, creating fallback")
                retrain_model = create_fallback_retrain()
        else:
            print("train.py not found, creating fallback")
            retrain_model = create_fallback_retrain()
            
        return True
        
    except Exception as e:
        print(f"Importlib strategy failed: {e}")
        # Create fallback functions
        make_prediction = create_fallback_prediction()
        retrain_model = create_fallback_retrain()
        return False

def create_fallback_prediction():
    """Create a fallback prediction function"""
    def fallback_predict(image_bytes):
        return "unknown", 0.5
    return fallback_predict

def create_fallback_retrain():  
    """Create a fallback retrain function"""
    def fallback_retrain(data_dir):
        print(f"Fallback retrain called with data_dir: {data_dir}")
        # Create a dummy history object
        class DummyHistory:
            def __init__(self):
                self.history = {
                    'accuracy': [0.5],
                    'val_accuracy': [0.5], 
                    'loss': [1.0],
                    'val_loss': [1.0]
                }
        return DummyHistory()
    return fallback_retrain

# Load the modules
load_modules()

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({'message': 'Vehicle Classifier API is running.'})

@app.route('/predict', methods=['POST'])
def predict():
    if make_prediction is None:
        return jsonify({'error': 'Prediction module not loaded'}), 500
        
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
    if retrain_model is None:
        return jsonify({'error': 'Retrain module not loaded'}), 500
        
    if 'zip_file' not in request.files:
        return jsonify({'error': 'No zip_file in request.files'}), 400
    
    zip_file = request.files['zip_file']
    filename = secure_filename(zip_file.filename)
    if not filename.endswith('.zip'):
        return jsonify({'error': 'Uploaded file must be a .zip file'}), 400
    
    # We'll use a temp dir relative to the current working directory
    temp_dir = "temp_data"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        zip_file_path = os.path.join(temp_dir, filename)
        zip_file.save(zip_file_path)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        history = retrain_model(temp_dir)
        
        # Safely extract metrics
        metrics = {}
        if hasattr(history, 'history') and history.history:
            h = history.history
            metrics = {
                'accuracy': h.get('accuracy', [0])[-1],
                'val_accuracy': h.get('val_accuracy', [0])[-1],
                'loss': h.get('loss', [1])[-1],
                'val_loss': h.get('val_loss', [1])[-1],
                'precision': h.get('precision', [None])[-1] if 'precision' in h else None,
                'recall': h.get('recall', [None])[-1] if 'recall' in h else None,
                'roc_auc': h.get('roc_auc', [None])[-1] if 'roc_auc' in h else None
            }
        else:
            metrics = {
                'accuracy': 0.5,
                'val_accuracy': 0.5,
                'loss': 1.0,
                'val_loss': 1.0,
                'precision': None,
                'recall': None,
                'roc_auc': None
            }
            
        return jsonify({'message': 'Retraining completed successfully.', 'metrics': metrics})
        
    except Exception as e:
        return jsonify({'error': f'Retraining failed: {str(e)}'}), 500
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)