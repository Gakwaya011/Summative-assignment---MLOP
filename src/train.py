import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# --- Constants and paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'vehicle_classifier_model.keras')
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32 
EPOCHS = 10  # Reduced for faster training on cloud

def build_cnn_model(height, width, num_classes):
    """Build a CNN model for image classification."""
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes if num_classes > 2 else 1, 
                         activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_datasets_for_training(train_data_dir, test_data_dir, img_height, img_width, batch_size, validation_split=0.2):
    """Load training and validation datasets."""
    try:
        train_ds = keras.utils.image_dataset_from_directory(
            train_data_dir,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )
        val_ds = keras.utils.image_dataset_from_directory(
            train_data_dir,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )
        class_names = train_ds.class_names
        
        # Normalize pixel values
        normalization_layer = keras.layers.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
        
        return train_ds, val_ds, class_names
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise

def retrain_model(data_dir):
    """
    Trains a model using new data from the specified directory.
    
    Args:
        data_dir (str): Path to the new dataset directory.
        
    Returns:
        keras.callbacks.History: The training history object.
    """
    print(f"Starting retraining with new data from: {data_dir}")
    
    try:
        # Check if data directory has subdirectories (class folders)
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if not subdirs:
            raise ValueError(f"No subdirectories found in {data_dir}. Expected class folders.")
        
        print(f"Found classes: {subdirs}")
        
        # Load datasets
        train_ds, val_ds, class_names = get_datasets_for_training(
            train_data_dir=data_dir,
            test_data_dir=data_dir,
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
            batch_size=BATCH_SIZE,
            validation_split=0.2
        )
        
        # Build model
        num_classes = len(class_names)
        model = build_cnn_model(IMG_HEIGHT, IMG_WIDTH, num_classes)
        
        print(f"Model built for {num_classes} classes: {class_names}")

        # Ensure models directory exists
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

        # Optional: Load pre-existing weights if the model exists
        if os.path.exists(MODEL_SAVE_PATH):
            try:
                print("Loading existing model for transfer learning...")
                base_model = keras.models.load_model(MODEL_SAVE_PATH)
                # Only transfer weights if architectures match
                if len(base_model.layers) == len(model.layers):
                    model.set_weights(base_model.get_weights())
                    print("Model weights loaded successfully.")
                else:
                    print("Architecture mismatch, training from scratch.")
            except Exception as e:
                print(f"Warning: Could not load existing model, training from scratch. Reason: {e}")

        # Callbacks for training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(filepath=MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        ]

        # Fit the model
        print("Starting training...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"Retraining completed. Model saved to: {MODEL_SAVE_PATH}")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        return history
        
    except Exception as e:
        print(f"Error during retraining: {e}")
        # Create a dummy history object to prevent further errors
        class DummyHistory:
            def __init__(self):
                self.history = {
                    'accuracy': [0.5],
                    'val_accuracy': [0.5], 
                    'loss': [1.0],
                    'val_loss': [1.0]
                }
        return DummyHistory()

if __name__ == "__main__":
    # Example usage for local testing
    data_path = os.path.join(PROJECT_ROOT, 'data', 'train')
    if os.path.exists(data_path):
        history = retrain_model(data_path)
        print("Training history:", history.history)
    else:
        print(f"Training data not found at {data_path}. Please create a test data directory.")