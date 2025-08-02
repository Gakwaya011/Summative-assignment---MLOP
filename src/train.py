import os
import sys
import shutil
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image_dataset_from_directory
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Dynamically add the project root to the system path
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))
sys.path.append(project_root)

# --- Configuration ---
DATA_DIR = os.path.join(project_root, 'combined_data_for_retraining')
MODEL_SAVE_PATH = os.path.join(project_root, 'models', 'vehicle_classifier_model.keras')
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32
EPOCHS = 50

def load_data(data_dir):
    """
    Loads training and validation datasets from a directory.
    """
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return None, None, None

    print(f"Loading data from: {data_dir} and {data_dir}")
    train_dataset = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    val_dataset = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_dataset.class_names
    print(f"Found class names: {class_names}")
    print(f"Number of batches in training set: {tf.data.experimental.cardinality(train_dataset)}")
    print(f"Number of batches in test set: {tf.data.experimental.cardinality(val_dataset)}")

    return train_dataset, val_dataset, class_names

def create_model():
    """
    Creates a new TensorFlow Keras model for vehicle classification.
    """
    print("Creating a new model...")
    model = keras.Sequential([
        keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(128, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid') # Binary classification
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(name='roc_auc')
        ]
    )
    model.summary()
    return model

def train_model(model, train_ds, val_ds):
    """
    Retrains an existing model or trains a new one.
    """
    # Create callbacks for better training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(filepath=MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    ]

    print(f"Training model with data from: {DATA_DIR} for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    return history

def evaluate_model(model, val_ds):
    """
    Evaluates the final model performance.
    """
    print("--- Evaluating Final Model ---")
    loss, accuracy, precision, recall, roc_auc = model.evaluate(val_ds)
    print(f"Final Validation Loss: {loss:.4f}")
    print(f"Final Validation Accuracy: {accuracy:.4f}")
    print(f"Final Validation Precision: {precision:.4f}")
    print(f"Final Validation Recall: {recall:.4f}")
    print(f"Final Validation ROC AUC: {roc_auc:.4f}")

def main():
    """Main function to handle model loading, training, and evaluation."""
    train_ds, val_ds, class_names = load_data(DATA_DIR)
    
    if train_ds is None:
        return

    # Check if a model exists, otherwise create a new one
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            model = keras.models.load_model(MODEL_SAVE_PATH)
            print(f"Loading existing model from {MODEL_SAVE_PATH} for retraining...")
            model.summary()
        except Exception as e:
            print(f"Could not load existing model. Creating a new one. Error: {e}")
            model = create_model()
    else:
        model = create_model()

    train_model(model, train_ds, val_ds)
    evaluate_model(model, val_ds)
    print("--- Model Training Completed ---")
    
if __name__ == '__main__':
    main()
