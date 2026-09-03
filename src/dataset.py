"""
Walks a data directory (train/val/test) laid out as:
    <dir>/NORMAL/*.jpeg
    <dir>/PNEUMONIA/*.jpeg
and turns it into (X, y) arrays of HOG features and labels.
"""
import os
import numpy as np
from tqdm import tqdm

from src.config import CLASSES
from src.features import extract_features

VALID_EXTS = (".jpeg", ".jpg", ".png")


def load_split(split_dir: str):
    """
    Returns (X, y, paths) for one split directory (e.g. data/train).
    Label 0 = NORMAL, label 1 = PNEUMONIA (order comes from src.config.CLASSES).
    """
    X, y, paths = [], [], []

    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Expected folder not found: {class_dir}\n"
                f"Make sure your dataset follows the structure described in README.md"
            )

        filenames = [f for f in os.listdir(class_dir) if f.lower().endswith(VALID_EXTS)]
        if not filenames:
            print(f"[warning] No images found in {class_dir}")

        for fname in tqdm(filenames, desc=f"Loading {class_name}", leave=False):
            fpath = os.path.join(class_dir, fname)
            try:
                feats = extract_features(fpath)
            except Exception as e:
                print(f"[warning] Skipping unreadable image {fpath}: {e}")
                continue
            X.append(feats)
            y.append(label_idx)
            paths.append(fpath)

    return np.array(X), np.array(y), paths
