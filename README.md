# 🫁 Pneumonia X-Ray Detector

An end-to-end machine learning project: upload a chest X-ray, get a **PNEUMONIA / NORMAL** prediction.

## 🔗 Live Demo

- **Web App (Streamlit):** https://xray-pneumonia-app.streamlit.app
- **API (FastAPI, interactive docs):** https://pneumonia-xray-detection.onrender.com/docs
- **API health check:** https://pneumonia-xray-detection.onrender.com/health

> Note: the API is hosted on Render's free tier, which spins down after inactivity — the first request after idle time can take 30-60 seconds to wake up.

## 🧠 Model

HOG (Histogram of Oriented Gradients) feature extraction + SVM classifier, trained on the Kaggle
["Chest X-Ray Images (Pneumonia)"](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
dataset (~5,800 images). Achieves roughly 90% accuracy on the held-out test set, with corrected
class imbalance handling.
# live url ( https://xray-pneumonia-app.streamlit.app/) 
FastAPI live url (https://pneumonia-xray-detection.onrender.com/docs)
