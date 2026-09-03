"""
Trains the pneumonia classifier and saves it to models/pneumonia_svm.joblib.

Usage:
    python -m src.train

Expects the dataset to already be downloaded into data/train, data/val, data/test
(see README.md for how to get it from Kaggle).
"""
import os
import time
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from src.config import TRAIN_DIR, VAL_DIR, TEST_DIR, MODEL_DIR, MODEL_PATH, CLASSES
from src.dataset import load_split


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading training data (extracting HOG features, this can take a while)...")
    t0 = time.time()
    X_train, y_train, _ = load_split(TRAIN_DIR)
    print(f"  train: {X_train.shape[0]} images, {X_train.shape[1]} features each "
          f"({time.time() - t0:.1f}s)")

    print("Loading validation data...")
    X_val, y_val, _ = load_split(VAL_DIR)
    print(f"  val: {X_val.shape[0]} images")

    # Handle class imbalance (the Kaggle dataset has ~3x more PNEUMONIA than NORMAL)
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"Class weights (to correct imbalance): {class_weight_dict}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                     class_weight=class_weight_dict, random_state=42)),
    ])

    print("Training SVM...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    print(f"  done in {time.time() - t0:.1f}s")

    print("\nValidation performance:")
    val_preds = pipeline.predict(X_val)
    print(classification_report(y_val, val_preds, target_names=CLASSES))

    print("Loading test data for final evaluation...")
    X_test, y_test, _ = load_split(TEST_DIR)
    test_preds = pipeline.predict(X_test)
    print("\nTest performance:")
    print(classification_report(y_test, test_preds, target_names=CLASSES))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, test_preds))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
