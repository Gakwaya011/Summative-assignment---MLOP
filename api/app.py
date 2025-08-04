import os
import base64
import shutil
import zipfile
import traceback
import logging
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import sys
import threading

# Configure logging to be more descriptive
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        logger.info("Successfully imported modules from src.")
        return True
    except ImportError as e:
        logger.error(f"Import error with src.module: {e}. Attempting direct imports...")
        # Fallback to direct imports
        try:
            from predictor import make_prediction
            from train import retrain_model
            logger.info("Successfully imported using direct imports.")
            return True
        except ImportError as e2:
            logger.critical(f"Final import failed for all modules: {e2}. Application will run with fallbacks.")
            make_prediction = create_safe_prediction_fallback()
            retrain_model = create_safe_retrain_fallback()
            return False

def create_safe_prediction_fallback():
    """Create a safe fallback prediction function."""
    def safe_predict(image_bytes):
        logger.warning(f"Fallback prediction called. Cannot make real predictions.")
        return "fallback_class", 0.5
    return safe_predict

def create_safe_retrain_fallback():
    """Create a safe fallback retrain function."""
    def safe_retrain(data_dir):
        logger.warning(f"Fallback retrain called. Will not retrain.")
        class DummyHistory:
            def __init__(self):
                self.history = {
                    'accuracy': [0.5], 'val_accuracy': [0.5],
                    'loss': [1.0], 'val_loss': [1.0]
                }
        return DummyHistory()
    return safe_retrain

# Load modules
load_modules()

app = Flask(__name__)
CORS(app)

@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({'error': f'Internal server error: {e}'}), 500

@app.route('/')
def index():
    """Health check endpoint."""
    try:
        return jsonify({
            'message': 'Vehicle Classifier API is running.',
            'status': 'healthy',
            'prediction_available': make_prediction.__name__ != 'safe_predict',
            'retrain_available': retrain_model.__name__ != 'safe_retrain'
        })
    except Exception as e:
        logger.error(f"Error in index route: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Image prediction endpoint."""
    try:
        logger.info("Prediction request received")
        
        if make_prediction.__name__ == 'safe_predict':
            return jsonify({'error': 'Prediction module not loaded'}), 500
        
        image_bytes = None
        
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            image_bytes = file.read()
            logger.info(f"Received file upload of {len(image_bytes)} bytes.")
        elif request.is_json and 'image' in request.json:
            try:
                image_b64 = request.json['image']
                image_bytes = base64.b64decode(image_b64)
                logger.info(f"Received base64 image of {len(image_bytes)} bytes.")
            except Exception as e:
                logger.error(f"Error decoding base64: {e}", exc_info=True)
                return jsonify({'error': f'Error decoding base64 image: {e}'}), 400
        else:
            return jsonify({'error': "No image data provided. Send 'file' in form-data or 'image' in JSON."}), 400
        
        if not image_bytes:
            return jsonify({'error': 'Empty image data'}), 400
        
        predicted_class, confidence = make_prediction(image_bytes)
        
        if "error" in predicted_class:
            return jsonify({'error': f"Prediction failed: {predicted_class}"}), 500
        
        logger.info(f"Prediction successful: {predicted_class}, Confidence: {confidence}")
        
        return jsonify({
            "predicted_class": str(predicted_class),
            "confidence": float(confidence),
            "status": "success"
        })
            
    except Exception as e:
        logger.error(f"Unexpected error in predict route: {e}", exc_info=True)
        return jsonify({'error': f'Unexpected error: {e}'}), 500

def retrain_async(actual_data_dir):
    """
    A function to run the retraining process in a separate thread.
    This prevents the web server from blocking and timing out.
    """
    try:
        logger.info(f"Starting retraining process in background thread.")
        history = retrain_model(actual_data_dir)
        logger.info("Retraining completed successfully in background.")
        
        # You could save the metrics to a file or database here
        # so the main API can retrieve the results later.
        
    except Exception as e:
        logger.error(f"Background retraining process failed: {e}", exc_info=True)
    finally:
        # Cleanup temporary directory
        if actual_data_dir and os.path.exists(actual_data_dir):
            try:
                shutil.rmtree(actual_data_dir, ignore_errors=True)
                logger.info("Temporary retraining data cleaned up.")
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temp directory: {cleanup_error}")

@app.route('/retrain', methods=['POST'])
def retrain():
    """Model retraining endpoint."""
    temp_dir = None
    try:
        logger.info("Retrain request received")
        
        if retrain_model.__name__ == 'safe_retrain':
            return jsonify({'error': 'Retrain module not loaded'}), 500
        
        if 'zip_file' not in request.files:
            return jsonify({'error': 'No zip_file in request.files'}), 400
        
        zip_file = request.files['zip_file']
        filename = secure_filename(zip_file.filename)
        
        if not filename.endswith('.zip'):
            return jsonify({'error': 'Uploaded file must be a .zip file'}), 400
        
        logger.info(f"Processing zip file: {filename}")
        
        temp_dir = os.path.join(os.getcwd(), "temp_retrain_data")
        os.makedirs(temp_dir, exist_ok=True)
        
        zip_file_path = os.path.join(temp_dir, filename)
        zip_file.save(zip_file_path)
        logger.info(f"Zip file saved to: {zip_file_path}")
        
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        logger.info("Zip file extracted successfully.")
        
        extracted_dir_name = os.path.splitext(filename)[0]
        actual_data_dir = os.path.join(temp_dir, extracted_dir_name)
        
        if not os.path.isdir(actual_data_dir):
            logger.error(f"Expected directory {actual_data_dir} not found after extraction.")
            return jsonify({'error': 'Failed to find extracted data directory.'}), 500

        logger.info(f"Using data directory: {actual_data_dir}")
        
        # Start the retraining process in a background thread
        thread = threading.Thread(target=retrain_async, args=(actual_data_dir,))
        thread.daemon = True
        thread.start()
        
        # Immediately return a response to the client
        return jsonify({
            'message': 'Retraining started in the background. The process may take several minutes.',
            'status': 'accepted'
        }), 202
        
    except Exception as e:
        logger.error(f"Error in retrain route: {e}", exc_info=True)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify({'error': f'Retraining failed to start: {e}'}), 500

if __name__ == '__main__':
    # Use a longer timeout for gunicorn, which is configured in render.yaml
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting app on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)
