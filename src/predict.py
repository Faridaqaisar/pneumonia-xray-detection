"""
Loads the trained model once and exposes a simple predict() function.
Used by both the FastAPI backend and (optionally) direct CLI testing.
"""
import sys
import joblib

from src.config import MODEL_PATH, CLASSES
from src.features import extract_features

_model = None  # lazy-loaded singleton so we don't reload the model on every call


def get_model():
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. "
                f"Run `python -m src.train` first (see README.md)."
            )
    return _model


def predict_image(path_or_bytes) -> dict:
    """
    Runs the full pipeline on one image and returns a result dict:
    {
        "label": "PNEUMONIA" or "NORMAL",
        "confidence": float (0-1, confidence in the predicted label),
        "probabilities": {"NORMAL": float, "PNEUMONIA": float}
    }
    """
    model = get_model()
    feats = extract_features(path_or_bytes).reshape(1, -1)
    probs = model.predict_proba(feats)[0]
    pred_idx = int(probs.argmax())

    return {
        "label": CLASSES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
    }


if __name__ == "__main__":
    # Quick CLI test: python -m src.predict path/to/xray.jpg
    if len(sys.argv) != 2:
        print("Usage: python -m src.predict <image_path>")
        sys.exit(1)
    result = predict_image(sys.argv[1])
    print(result)
