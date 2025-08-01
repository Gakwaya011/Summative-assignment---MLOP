import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import sys

# Add the src directory to the Python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from predictor import load_model, predict_image, CLASS_NAMES

# Define the path to the saved model and ensure the upload folder exists
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'vehicle_classifier_model.keras')
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask app and load the model once
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please run src/train.py first.")
    model = None

@app.route('/')
def home():
    """Renders the HTML form for image upload."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for making predictions on an uploaded image.
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            predicted_class, confidence = predict_image(model, filepath)
            os.remove(filepath) # Clean up the uploaded file
            
            return jsonify({
                "class_name": predicted_class,
                "confidence": float(confidence)
            })
        except Exception as e:
            os.remove(filepath)
            return jsonify({"error": str(e)}), 500
            
if __name__ == '__main__':
    # Flask will automatically find the app object if run with 'flask run'
    # We add this for direct execution compatibility
    app.run(debug=True, port=5000)