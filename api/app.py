import os
import zipfile
import shutil
import base64
import numpy as np
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import tensorflow as tf
import gc

app = Flask(__name__)
CORS(app)

MODEL_PATH = "vehicle_classifier_model.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Cars', 'Motorcycles']
model = None  # Lazy-loaded global


def create_initial_model():
    """Create a basic model if no trained model exists"""
    print("Creating initial model...")
    initial_model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255, input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    initial_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    # Create dummy data to initialize the model
    dummy_data = tf.random.normal((1, 224, 224, 3))
    initial_model.predict(dummy_data)
    
    initial_model.save(MODEL_PATH)
    print("Initial model created and saved.")
    return initial_model


def load_current_model():
    global model
    if model is None:
        print("Loading model into memory...")
        try:
            if os.path.exists(MODEL_PATH):
                model = load_model(MODEL_PATH)
                print("Model loaded successfully for prediction.")
            else:
                print("No existing model found, creating initial model...")
                model = create_initial_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new model...")
            model = create_initial_model()
    return model


def clean_dataset_directory(directory):
    """Remove non-image files and empty folders from dataset directory"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    
    for root, dirs, files in os.walk(directory, topdown=False):
        # Remove non-image files
        for file in files:
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file.lower())
            if ext not in valid_extensions:
                print(f"Removing non-image file: {file}")
                try:
                    os.remove(file_path)
                except:
                    pass
        
        # Remove empty directories
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):
                    print(f"Removing empty directory: {dir_name}")
                    os.rmdir(dir_path)
            except:
                pass


def validate_dataset_structure(directory):
    """Validate dataset structure and return detailed info"""
    try:
        class_dirs = []
        total_images = 0
        
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                # Count images in this directory
                image_count = 0
                valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
                
                for file in os.listdir(item_path):
                    _, ext = os.path.splitext(file.lower())
                    if ext in valid_extensions:
                        image_count += 1
                
                if image_count > 0:
                    class_dirs.append({
                        'name': item,
                        'count': image_count
                    })
                    total_images += image_count
        
        return {
            'valid': len(class_dirs) >= 2 and total_images >= 4,
            'class_dirs': class_dirs,
            'total_images': total_images,
            'error': None
        }
    
    except Exception as e:
        return {
            'valid': False,
            'class_dirs': [],
            'total_images': 0,
            'error': str(e)
        }


@app.route("/")
def home():
    return jsonify({
        "status": "healthy",
        "service": "Vehicle Classifier API",
        "version": "1.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "retrain": "/retrain (POST)",
            "health": "/health (GET)"
        }
    }), 200


@app.route("/health")
def health_check():
    """Additional health check endpoint"""
    try:
        # Test if model can be loaded
        current_model = load_current_model()
        model_status = "loaded" if current_model else "error"
        
        return jsonify({
            "status": "healthy",
            "model_status": model_status,
            "model_exists": os.path.exists(MODEL_PATH)
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Handle JSON payload with base64 image (from Streamlit)
        if request.is_json:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({"error": "No image data provided"}), 400
            
            # Decode base64 image
            try:
                image_bytes = base64.b64decode(data['image'])
                image_file = BytesIO(image_bytes)
            except Exception as e:
                return jsonify({"error": "Invalid base64 image data"}), 400
        
        # Handle file upload (alternative method)
        elif "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No selected file"}), 400
            
            os.makedirs("temp", exist_ok=True)
            img_path = os.path.join("temp", secure_filename(file.filename))
            file.save(img_path)
            image_file = img_path
        
        else:
            return jsonify({"error": "No image data provided"}), 400

        # Process image
        if isinstance(image_file, str):
            # File path
            img = image.load_img(image_file, target_size=IMG_SIZE)
            img_path = image_file
        else:
            # BytesIO object
            img = image.load_img(image_file, target_size=IMG_SIZE)
            img_path = None

        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) / 255.0

        # Make prediction
        current_model = load_current_model()
        predictions = current_model.predict(img_array)
        score = float(predictions[0][0])
        
        # Determine class and confidence
        if score > 0.5:
            predicted_class = CLASS_NAMES[1]  # Motorcycles
            confidence = score
        else:
            predicted_class = CLASS_NAMES[0]  # Cars
            confidence = 1 - score

        # Clean up memory and temp files
        tf.keras.backend.clear_session()
        gc.collect()
        
        if img_path and os.path.exists(img_path):
            os.remove(img_path)

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "raw_score": float(score)
        }), 200

    except Exception as e:
        print(f"⚠️ Prediction failed: {e}")
        # Clean up on error
        tf.keras.backend.clear_session()
        gc.collect()
        return jsonify({"error": str(e)}), 500


@app.route("/retrain", methods=["POST"])
def retrain():
    # Check for zip file (handle both 'file' and 'zip_file' field names)
    zip_file = None
    if "zip_file" in request.files:
        zip_file = request.files["zip_file"]
    elif "file" in request.files:
        zip_file = request.files["file"]
    
    if not zip_file or zip_file.filename == "":
        return jsonify({"error": "No zip file provided"}), 400

    # Save uploaded zip file
    os.makedirs("temp", exist_ok=True)
    zip_path = os.path.join("temp", secure_filename(zip_file.filename))
    zip_file.save(zip_path)

    extract_path = "temp/extracted_data"
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    os.makedirs(extract_path)

    try:
        # Extract zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"✅ Extracted zip to: {extract_path}")
        
        # Clean the dataset - remove non-image files
        clean_dataset_directory(extract_path)
        
        # Validate dataset structure
        validation_result = validate_dataset_structure(extract_path)
        
        if not validation_result['valid']:
            error_msg = f"Invalid dataset structure. Found {len(validation_result['class_dirs'])} classes with {validation_result['total_images']} total images. Need at least 2 classes with minimum 2 images each."
            if validation_result['error']:
                error_msg += f" Error: {validation_result['error']}"
            
            print(f"Dataset validation failed: {error_msg}")
            print(f"Classes found: {validation_result['class_dirs']}")
            
            return jsonify({
                "error": error_msg,
                "details": validation_result
            }), 400
        
        print(f"✅ Dataset validation passed: {validation_result}")
        
    except zipfile.BadZipFile:
        print("⚠️ Invalid zip file")
        return jsonify({"error": "Invalid zip file"}), 400
    except Exception as e:
        print(f"⚠️ Error extracting zip: {e}")
        return jsonify({"error": f"Error extracting zip file: {str(e)}"}), 500
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

    try:
        # Create datasets with error handling
        batch_size = 16  # Reduced for memory
        
        try:
            train_ds = image_dataset_from_directory(
                extract_path, 
                validation_split=0.2, 
                subset="training",
                seed=123, 
                image_size=IMG_SIZE, 
                batch_size=batch_size,
                label_mode='binary'  # For binary classification
            )
            val_ds = image_dataset_from_directory(
                extract_path, 
                validation_split=0.2, 
                subset="validation",
                seed=123, 
                image_size=IMG_SIZE, 
                batch_size=batch_size,
                label_mode='binary'  # For binary classification
            )
            
            print(f"✅ Datasets created successfully")
            print(f"Training batches: {len(train_ds)}")
            print(f"Validation batches: {len(val_ds)}")
            
        except Exception as dataset_error:
            print(f"Dataset creation error: {dataset_error}")
            return jsonify({"error": f"Invalid dataset format: {str(dataset_error)}"}), 400

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

        # Create lighter model for free tier
        new_model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1. / 255, input_shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(16, 3, activation='relu'),  # Reduced filters
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),  # Reduced filters
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),  # Reduced filters
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),  # Reduced neurons
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        new_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, verbose=1)
        ]

        print("🚀 Starting training...")
        # Train with fewer epochs for memory
        history = new_model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=5,  # Reduced epochs
            callbacks=callbacks,
            verbose=1
        )

        print("📊 Evaluating model...")
        # Evaluation
        y_true, y_pred = [], []
        for images, labels in val_ds:
            preds = new_model.predict(images)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.flatten())

        y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_pred)

        # Save and reload the model
        new_model.save(MODEL_PATH)
        print("✅ Model saved successfully")

        # Update global model
        global model
        model = new_model

        # Clean up
        shutil.rmtree(extract_path)
        tf.keras.backend.clear_session()
        gc.collect()

        return jsonify({
            "message": "✅ Retraining completed successfully.",
            "dataset_info": validation_result,
            "metrics": {
                "accuracy": float(history.history["val_accuracy"][-1]),
                "loss": float(history.history["val_loss"][-1]),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "roc_auc": round(roc_auc, 4),
            }
        }), 200

    except Exception as e:
        print(f"⚠️ Retraining failed: {e}")
        # Clean up on error
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        tf.keras.backend.clear_session()
        gc.collect()
        return jsonify({"error": f"Training failed: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)