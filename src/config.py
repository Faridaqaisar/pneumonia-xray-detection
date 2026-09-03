"""
Central configuration for the pneumonia detection project.
Change paths/sizes here — everything else imports from this file.
"""
import os

# Root of the project (this file lives in src/, so go one level up)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset layout expected (standard Kaggle "Chest X-Ray Images (Pneumonia)" layout):
# data/
#   train/NORMAL/*.jpeg   train/PNEUMONIA/*.jpeg
#   val/NORMAL/*.jpeg     val/PNEUMONIA/*.jpeg
#   test/NORMAL/*.jpeg    test/PNEUMONIA/*.jpeg
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

CLASSES = ["NORMAL", "PNEUMONIA"]  # label 0 = NORMAL, label 1 = PNEUMONIA

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_svm.joblib")

# Image preprocessing settings
IMAGE_SIZE = (128, 128)  # (width, height) images are resized to before feature extraction

# HOG feature extractor settings
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)
