import tensorflow as tf
import os

# Define common parameters for data loading (adjust if your notebook uses different values)
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

def get_datasets(data_dir: str, img_height: int, img_width: int, batch_size: int, validation_split: float = 0.2):
    """
    Loads and prepares training and testing datasets using TensorFlow's image_dataset_from_directory.
    
    Args:
        data_dir (str): Path to the data directory.
        img_height (int): Height to resize images to.
        img_width (int): Width to resize images to.
        batch_size (int): Batch size for the datasets.
        validation_split (float): Fraction of data to reserve for validation. Defaults to 0.2.

    Returns:
        tuple: A tuple containing (train_ds, val_ds, class_names).
    """
    print(f"Loading data from: {data_dir}")

    # Load training data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        validation_split=validation_split,
        subset="training",
        seed=42,
        image_size=(img_height, img_width),
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=True
    )
    
    class_names = train_ds.class_names
    print(f"Found class names: {class_names}")

    # Load validation data
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        validation_split=validation_split,
        subset="validation",
        seed=42,
        image_size=(img_height, img_width),
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=False
    )
    
    # Optimize datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print(f"Number of batches in training set: {len(train_ds)}")
    print(f"Number of batches in validation set: {len(val_ds)}")

    return train_ds, val_ds, class_names

if __name__ == "__main__":
    # Example usage for testing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    data_path = os.path.join(parent_dir, 'data', 'train')
    if os.path.exists(data_path):
        train_dataset, val_dataset, names = get_datasets(data_path, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)
        print(f"Successfully loaded datasets with classes: {names}")
        print(f"Sample batch shape from training set: {next(iter(train_dataset))[0].shape}")
    else:
        print(f"Test data not found at {data_path}")