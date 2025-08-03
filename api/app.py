import os
import zipfile
import shutil
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import tensorflow as tf

app = Flask(__name__)
MODEL_PATH = "vehicle_classifier_model.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Cars', 'Motorcycles']
model = None  # global model variable


def load_current_model():
    global model
    if model is None:
        model = load_model(MODEL_PATH)
    return model


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    img_path = os.path.join("temp", secure_filename(file.filename))
    os.makedirs("temp", exist_ok=True)
    file.save(img_path)

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    model = load_current_model()
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[int(predictions[0][0] > 0.5)]
    confidence = float(predictions[0][0]) if predicted_class == "Motorcycles" else float(1 - predictions[0][0])

    os.remove(img_path)

    return jsonify({"class": predicted_class, "confidence": round(confidence, 4)})


@app.route("/retrain", methods=["POST"])
def retrain():
    if "file" not in request.files:
        return jsonify({"error": "No zip file provided"}), 400

    zip_file = request.files["file"]
    zip_path = os.path.join("temp", secure_filename(zip_file.filename))
    os.makedirs("temp", exist_ok=True)
    zip_file.save(zip_path)

    extract_path = "temp/extracted_data"
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    os.makedirs(extract_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    os.remove(zip_path)

    batch_size = 32
    train_ds = image_dataset_from_directory(
        extract_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=batch_size,
    )
    val_ds = image_dataset_from_directory(
        extract_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=batch_size,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    new_model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255, input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    new_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, verbose=1)
    ]

    history = new_model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

    # Evaluation
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        preds = new_model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.flatten())

    y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    roc_auc = roc_auc_score(y_true, y_pred)

    # Save and reload the model
    new_model.save(MODEL_PATH)

    # Reload into memory
    global model
    model = load_model(MODEL_PATH)

    shutil.rmtree(extract_path)

    return jsonify({
        "message": "Retraining completed successfully.",
        "accuracy": float(history.history["val_accuracy"][-1]),
        "loss": float(history.history["val_loss"][-1]),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "roc_auc": round(roc_auc, 4),
    })
