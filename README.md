---
title: Pneumonia X-Ray Detector API
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Pneumonia X-Ray Detector — FastAPI Backend

This Space runs the FastAPI backend that serves pneumonia predictions from chest X-ray images.
It's called by a separate Streamlit frontend deployed on Streamlit Community Cloud.

## Endpoints

- `GET /health` — liveness check
- `POST /predict` — upload a chest X-ray image (JPEG/PNG), returns `{label, confidence, probabilities}`

## Model

HOG (Histogram of Oriented Gradients) features + SVM classifier, trained on the Kaggle
"Chest X-Ray Images (Pneumonia)" dataset (~5,800 images).

**Disclaimer:** This is a learning/demo project, not a certified medical device. Not for real diagnosis.