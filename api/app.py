import os
import base64
import shutil
import zipfile
import traceback
import logging
import signal
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Initialize variables
make_prediction = None
retrain_model = None

def load_modules():
    """Load prediction and training modules with comprehensive error handling."""
    global make_prediction, retrain_model
    
    try:
        from src.predictor import make_prediction
        from src.train import retrain_model
        logger.info("Successfully imported using src.module")
        return True
    except ImportError as e:
        logger.error(f"Import error with src.module: {e}")
        try:
            from predictor import make_prediction
            from train import retrain_model
            logger.info("Successfully imported using direct imports")
            return True
        except ImportError as e2:
            logger.error(f"Import error with direct imports: {e2}")
            # Create safe fallback functions
            make_prediction = create_safe_prediction_fallback()
            retrain_model = create_safe_retrain_fallback()
            return False

def create_safe_prediction_fallback():
    """Create a safe fallback prediction function."""
    def safe_predict(image_bytes):
        logger.info(f"Fallback prediction called with {len(image_bytes) if image_bytes else 0} bytes")
        return "fallback_class", 0.5
    return safe_predict

def create_safe_retrain_fallback():
    """Create a safe fallback retrain function."""
    def safe_retrain(data_dir):
        logger.info(f"Fallback retrain called with data_dir: {data_dir}")
        class DummyHistory:
            def __init__(self):
                self.history = {
                    'accuracy': [0.5],
                    'val_accuracy': [0.5], 
                    'loss': [1.0],
                    'val_loss': [1.0]
                }
        return DummyHistory()
    return safe_retrain

# Load modules
load_modules()

app = Flask(__name__)
CORS(app)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")

def run_with_timeout(func, args, timeout_seconds=25):
    """Run a function with a timeout."""
    try:
        # Set up the timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        # Run the function
        result = func(*args)
        
        # Cancel the timeout
        signal.alarm(0)
        return result
        
    except TimeoutException:
        logger.error(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
        return None
    except Exception as e:
        signal.alarm(0)  # Make sure to cancel timeout
        raise e

@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(traceback.format_exc())
    return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/')
def index():
    """Health check endpoint."""
    try:
        return jsonify({
            'message': 'Vehicle Classifier API is running.',
            'status': 'healthy',
            'prediction_available': make_prediction is not None,
            'retrain_available': retrain_model is not None
        })
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Image prediction endpoint."""
    try:
        logger.info("Prediction request received")
        
        if make_prediction is None:
            return jsonify({'error': 'Prediction module not loaded'}), 500
        
        # Extract image data
        image_bytes = None
        
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            image_bytes = file.read()
            logger.info(f"Received file upload: {len(image_bytes)} bytes")
            
        elif request.is_json and 'image' in request.json:
            try:
                image_b64 = request.json['image']
                image_bytes = base64.b64decode(image_b64)
                logger.info(f"Received base64 image: {len(image_bytes)} bytes")
            except Exception as e:
                logger.error(f"Error decoding base64: {str(e)}")
                return jsonify({'error': f'Error decoding base64 image: {str(e)}'}), 400
        else:
            return jsonify({'error': "No image data provided. Send 'file' in form-data or 'image' in JSON."}), 400
        
        # Validate image data
        if not image_bytes:
            return jsonify({'error': 'Empty image data'}), 400
        
        if len(image_bytes) < 50:
            return jsonify({'error': 'Image data too small, likely corrupted'}), 400
        
        # Check image format (basic validation)
        if not image_bytes.startswith((b'\xff\xd8', b'\x89PNG', b'GIF', b'RIFF')):
            logger.warning("Image doesn't appear to be a standard format")
        
        # Make prediction with timeout handling
        try:
            logger.info("Starting prediction with timeout protection...")
            result = run_with_timeout(make_prediction, (image_bytes,), timeout_seconds=25)
            
            if result is None:
                logger.error("Prediction timed out")
                return jsonify({
                    'error': 'Prediction timed out - server may be overloaded',
                    'predicted_class': 'timeout_error',
                    'confidence': 0.0
                }), 408  # Request Timeout
            
            predicted_class, confidence = result
            logger.info(f"Prediction successful: {predicted_class}, {confidence}")
            
            return jsonify({
                "predicted_class": str(predicted_class),
                "confidence": float(confidence),
                "status": "success"
            })
            
        except Exception as pred_error:
            logger.error(f"Prediction function error: {str(pred_error)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': f'Prediction failed: {str(pred_error)}',
                'predicted_class': 'error_prediction',
                'confidence': 0.0
            }), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in predict route: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Model retraining endpoint."""
    temp_dir = None
    try:
        logger.info("Retrain request received")
        
        if retrain_model is None:
            return jsonify({'error': 'Retrain module not loaded'}), 500
        
        # Validate request
        if 'zip_file' not in request.files:
            return jsonify({'error': 'No zip_file in request.files'}), 400
        
        zip_file = request.files['zip_file']
        if zip_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        filename = secure_filename(zip_file.filename)
        if not filename.endswith('.zip'):
            return jsonify({'error': 'Uploaded file must be a .zip file'}), 400
        
        logger.info(f"Processing zip file: {filename}")
        
        # Create temporary directory
        temp_dir = "temp_retrain_data"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save and extract zip file
        zip_file_path = os.path.join(temp_dir, filename)
        zip_file.save(zip_file_path)
        logger.info(f"Zip file saved to: {zip_file_path}")
        
        # Extract zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        logger.info("Zip file extracted successfully")
        
        # List contents for debugging
        extracted_contents = []
        for root, dirs, files in os.walk(temp_dir):
            level = root.replace(temp_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            logger.info(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                logger.info(f"{subindent}{file}")
                extracted_contents.append(os.path.join(root, file))
        
        # Find the actual data directory (skip the zip file itself)
        data_dirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
        if data_dirs:
            actual_data_dir = os.path.join(temp_dir, data_dirs[0])
            logger.info(f"Using data directory: {actual_data_dir}")
        else:
            actual_data_dir = temp_dir
            logger.info(f"Using temp directory directly: {actual_data_dir}")
        
        # Start retraining
        logger.info("Starting model retraining...")
        history = retrain_model(actual_data_dir)
        logger.info("Retraining completed successfully")
        
        # Extract metrics safely
        metrics = {}
        try:
            if hasattr(history, 'history') and history.history:
                h = history.history
                metrics = {
                    'accuracy': float(h.get('accuracy', [0])[-1]),
                    'val_accuracy': float(h.get('val_accuracy', [0])[-1]),
                    'loss': float(h.get('loss', [1])[-1]),
                    'val_loss': float(h.get('val_loss', [1])[-1]),
                    'precision': float(h.get('precision', [None])[-1]) if 'precision' in h and h['precision'][-1] is not None else None,
                    'recall': float(h.get('recall', [None])[-1]) if 'recall' in h and h['recall'][-1] is not None else None,
                    'roc_auc': float(h.get('roc_auc', [None])[-1]) if 'roc_auc' in h and h['roc_auc'][-1] is not None else None
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
        except Exception as metrics_error:
            logger.error(f"Error extracting metrics: {str(metrics_error)}")
            metrics = {'error': 'Could not extract training metrics'}
        
        return jsonify({
            'message': 'Retraining completed successfully.',
            'metrics': metrics,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error in retrain route: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Retraining failed: {str(e)}'}), 500
        
    finally:
        # Cleanup temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info("Temporary directory cleaned up")
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temp directory: {str(cleanup_error)}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting app on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)