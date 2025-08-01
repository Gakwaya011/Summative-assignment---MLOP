import tensorflow as tf
import os

# Define common parameters for data loading (adjust if your notebook uses different values)
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

def get_datasets(train_data_dir: str, test_data_dir: str, img_height: int, img_width: int, batch_size: int):
    """
    Loads and prepares training and testing datasets using TensorFlow's image_dataset_from_directory.

    Args:
        train_data_dir (str): Path to the training data directory.
        test_data_dir (str): Path to the testing data directory.
        img_height (int): Height to resize images to.
        img_width (int): Width to resize images to.
        batch_size (int): Batch size for the datasets.

    Returns:
        tuple: A tuple containing (train_ds, test_ds, class_names).
    """
    print(f"Loading data from: {train_data_dir} and {test_data_dir}")

    # Load training data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_data_dir,
        labels='inferred',
        label_mode='int',
        image_size=(img_height, img_width),
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    # Correctly retrieve class names from the initial dataset object
    class_names = train_ds.class_names
    print(f"Found class names: {class_names}")

    # Load test data
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_data_dir,
        labels='inferred',
        label_mode='int',
        image_size=(img_height, img_width),
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=False, # No shuffling for test set
        seed=42
    )

    # Optimize datasets for performance AFTER getting class names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print(f"Number of batches in training set: {len(train_ds)}")
    print(f"Number of batches in test set: {len(test_ds)}")

    return train_ds, test_ds, class_names

if __name__ == "__main__":
    # Example usage for testing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    train_path = os.path.join(parent_dir, 'data', 'train')
    test_path = os.path.join(parent_dir, 'data', 'test')
    train_dataset, test_dataset, names = get_datasets(train_path, test_path, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)
    print(f"Successfully loaded datasets with classes: {names}")
    print(f"Sample batch shape from training set: {next(iter(train_dataset))[0].shape}")