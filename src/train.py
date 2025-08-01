import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("Attempting to configure system path...")
# Add the src directory to the system path to allow importing modules
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))
sys.path.append(project_root)
print(f"System path configured. Project root added: {project_root}")

try:
    # Import modules from src/
    print("Attempting to import modules from src...")
    from src.data_loader import get_datasets, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
    from src.model_builder import build_cnn_model
    print("Modules imported successfully.")
except ImportError as e:
    print("An ImportError occurred! This is likely a pathing issue.")
    print(f"Error details: {e}")
    sys.exit(1) # Exit the script with an error code

# Define paths (relative to the project root, for consistency)
TRAIN_DATA_DIR = os.path.join(project_root, 'data', 'train')
TEST_DATA_DIR = os.path.join(project_root, 'data', 'test')
MODEL_SAVE_PATH = os.path.join(project_root, 'models', 'vehicle_classifier_model.keras') # Use the .keras extension!

def train_model(epochs: int = 5, initial_model_path: str = None):
    """
    Orchestrates the training of the CNN model.

    Args:
        epochs (int): Number of training epochs.
        initial_model_path (str, optional): Path to a pre-existing model to load and continue training.
                                            If None, a new model is built.
    Returns:
        tf.keras.Model: The trained model.
    """
    print("\n--- Starting Model Training Process ---")

    # 1. Load Data
    train_ds, test_ds, class_names = get_datasets(TRAIN_DATA_DIR, TEST_DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

    # 2. Build or Load Model
    if initial_model_path and os.path.exists(initial_model_path):
        print(f"Loading existing model from {initial_model_path} for retraining...")
        # Custom objects are important when loading models with custom metrics/layers/regularizers
        model = keras.models.load_model(
            initial_model_path,
            custom_objects={
                'Precision': tf.keras.metrics.Precision,
                'Recall': tf.keras.metrics.Recall,
                'AUC': tf.keras.metrics.AUC,
                'l2': tf.keras.regularizers.l2 # Important for models using L2 regularization
            }
        )
    else:
        print("Building a new model...")
        model = build_cnn_model(IMG_HEIGHT, IMG_WIDTH)

    model.summary()

    # 3. Define Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)
    ]

    # 4. Train Model
    print(f"\nTraining model for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    print("\n--- Model Training Completed ---")

    # 5. Evaluate Model on Test Set (optional, but good for feedback)
    print("\n--- Evaluating Model on Test Set ---")
    loss, accuracy, precision, recall, roc_auc = model.evaluate(test_ds)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test ROC AUC: {roc_auc:.4f}")

    # 6. Save the trained model
    print(f"\nSaving trained model to: {MODEL_SAVE_PATH}")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print("Model saved successfully!")

    return model

if __name__ == "__main__":
    print("Executing train.py directly...")
    trained_model = train_model(epochs=50) # Reduced epochs to 5 for quicker testing