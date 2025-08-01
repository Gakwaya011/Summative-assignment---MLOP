import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import Precision, Recall, AUC

def build_cnn_model(img_height: int, img_width: int, num_classes: int = 1):
    """
    Builds a custom CNN model with data augmentation and L2 regularization.

    Args:
        img_height (int): Height of the input images.
        img_width (int): Width of the input images.
        num_classes (int): Number of output classes (1 for binary classification with sigmoid).

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

        # Data Augmentation Layers
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),

        # Convolutional Blocks with L2 regularization
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),

        layers.Dropout(0.5), # Current dropout rate

        # Dense Layer with L2 regularization
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),

        # Output Layer
        layers.Dense(num_classes, activation='sigmoid') # Sigmoid for binary classification
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            Precision(),
            Recall(),
            AUC(name='roc_auc')
        ]
    )
    return model

if __name__ == "__main__":
    print("Building a sample model for demonstration...")
    sample_model = build_cnn_model(128, 128)
    sample_model.summary()
    print("Sample model built and compiled successfully.")