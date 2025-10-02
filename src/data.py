
"""
Data loading and preprocessing utilities.
Default: TensorFlow/Keras datasets (Fashion-MNIST or CIFAR-10).

Switch to PyTorch by using torchvision in a similar structure if preferred.
"""
from typing import Tuple
import numpy as np

# TensorFlow as default backend
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASETS = {
    "fashion_mnist": tf.keras.datasets.fashion_mnist,
    "cifar10": tf.keras.datasets.cifar10,
}

CLASS_NAMES = {
    "fashion_mnist": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    "cifar10": ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"],
}

def load_dataset(name: str = "fashion_mnist", normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    if name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {name}. Choose from {list(DATASETS.keys())}")
    (x_train, y_train), (x_test, y_test) = DATASETS[name].load_data()
    # Ensure shape (N, H, W, C)
    if x_train.ndim == 3:  # (N, H, W) -> add channel
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
    if normalize:
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
    classes = CLASS_NAMES[name]
    return x_train, y_train.squeeze(), x_test, y_test.squeeze(), classes

def get_datagen(augment: bool = False):
    if augment:
        return ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
    else:
        return ImageDataGenerator()

def make_generators(x_train, y_train, x_val, y_val, batch_size: int = 64, augment: bool = False):
    train_gen = get_datagen(augment).flow(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_gen = ImageDataGenerator().flow(x_val, y_val, batch_size=batch_size, shuffle=False)
    return train_gen, val_gen
