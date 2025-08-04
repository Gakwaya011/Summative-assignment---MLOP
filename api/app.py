import os
import base64
import shutil
import zipfile
import logging
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import sys
import threading
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF logs

# ========== CONFIGURATION ==========
# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')  # Persistent logs
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_PREDICTION_SIZE = 5 * 1024 * 1024  # 5MB
MAX_RETRAIN_SIZE = 50 * 1024 * 1024    # 50MB
TEMP_DIR = "/tmp/vehicle_classifier"    # More reliable than cwd

# ========== INITIALIZATION ==========
app = Flask(__name__)
CORS(app)

# Initialize modules with retry logic
def initialize_modules(max_retries=3, delay=2):
    """Robust module initialization with retries"""
    global make_prediction, retrain_model
    
    for attempt in range(max_retries):
        try:
            from src.predictor import make_prediction
            from src.train import retrain_model
            logger.info(f"Modules loaded successfully (attempt {attempt+1})")
            return True
        except ImportError as e:
            logger.warning(f"Import attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    
    logger.error("All import attempts failed. Using fallbacks.")
    make_prediction = create_fallback_predictor()
    retrain_model = create_fallback_trainer()
    return False

def create_fallback_predictor():
    """Lightweight prediction fallback"""
    def predict(image_bytes):
        logger.warning("Using fallback predictor")
        return "fallback", 0.5
    return predict

def create_fallback_trainer():
    """Lightweight training fallback"""
    def train(data_dir):
        logger.warning("Using fallback trainer")
        return {"status": "fallback"}
    return train

# Initialize with retries
initialize_modules()

# ========== HELPER FUNCTIONS ==========
def validate_image_size(image_bytes):
    """Prevent memory bombs"""
    if len(image_bytes) > MAX_PREDICTION_SIZE:
        raise ValueError(f"Image exceeds {MAX_PREDICTION_SIZE//1024//1024}MB limit")

def clean_temp_dir():
    """Ensure clean temporary directory"""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

# ========== ROUTES ==========
@app.route('/')
def health_check():
    """Lightweight health endpoint"""
    return jsonify({
        'status': 'ok',
        'features': {
            'prediction': make_prediction.__name__ != 'predict',
            'retraining': retrain_model.__name__ != 'train'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Optimized prediction endpoint"""
    start_time = time.time()
    
    try:
        # Fast input validation
        if not request.content_length or request.content_length > MAX_PREDICTION_SIZE:
            return jsonify({'error': 'Invalid content size'}), 413
            
        # Handle both file upload and base64
        if 'file' in request.files:
            image_bytes = request.files['file'].read()
        elif request.is_json:
            image_bytes = base64.b64decode(request.json.get('image', ''))
        else:
            return jsonify({'error': 'Unsupported content type'}), 400
            
        # Size check
        validate_image_size(image_bytes)
        
        # Make prediction with timeout awareness
        if (time.time() - start_time) > 5:  # Already took too long
            return jsonify({'error': 'Request processing too slow'}), 408
            
        class_name, confidence = make_prediction(image_bytes)
        
        return jsonify({
            'class': class_name,
            'confidence': confidence,
            'processing_time': round(time.time() - start_time, 2)
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def background_retrain(data_dir):
    """Safe retraining in background"""
    try:
        logger.info(f"Starting retrain with {len(os.listdir(data_dir))} files")
        retrain_model(data_dir)
        logger.info("Retrain completed")
    except Exception as e:
        logger.error(f"Retrain failed: {str(e)}")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)

@app.route('/retrain', methods=['POST'])
def retrain():
    """Async retraining endpoint"""
    clean_temp_dir()  # Start fresh
    
    try:
        # Validate input
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if not file.filename.endswith('.zip'):
            return jsonify({'error': 'Only ZIP files accepted'}), 400
            
        # Save and extract
        zip_path = os.path.join(TEMP_DIR, secure_filename(file.filename))
        file.save(zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TEMP_DIR)
        
        # Start background job
        thread = threading.Thread(target=background_retrain, args=(TEMP_DIR,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'started',
            'message': 'Retraining in progress'
        }), 202
        
    except Exception as e:
        clean_temp_dir()
        return jsonify({'error': str(e)}), 500

# ========== STARTUP ==========
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting optimized server on port {port}")
    app.run(host='0.0.0.0', port=port, threaded=True)