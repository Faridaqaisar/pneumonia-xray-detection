"""
Turns a chest X-ray image into a fixed-length numeric feature vector
that a scikit-learn classifier can consume.

Pipeline: load -> convert to grayscale -> resize -> HOG features
HOG (Histogram of Oriented Gradients) captures edge/texture patterns,
which works well for X-rays since the diagnostic signal is largely
about texture and opacity patterns in the lungs.
"""
import numpy as np
from PIL import Image
from skimage.feature import hog

from src.config import IMAGE_SIZE, HOG_PARAMS


def load_image(path_or_bytes) -> Image.Image:
    """Load an image from a filepath, file-like object, or raw bytes."""
    img = Image.open(path_or_bytes)
    return img.convert("L")  # grayscale


def preprocess(img: Image.Image) -> np.ndarray:
    """Resize and normalize an already-grayscale PIL image to a numpy array."""
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def extract_features(path_or_bytes) -> np.ndarray:
    """
    Full pipeline: image (path, bytes, or file-like) -> 1D feature vector.
    This is the single function both training and inference must use,
    so features are always computed identically.
    """
    img = load_image(path_or_bytes)
    arr = preprocess(img)
    feats = hog(arr, **HOG_PARAMS)
    return feats
