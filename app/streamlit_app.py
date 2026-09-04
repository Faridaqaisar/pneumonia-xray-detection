"""
Streamlit UI for the pneumonia detector.

Run locally with:
    streamlit run app/streamlit_app.py

Locally this talks to the FastAPI backend over HTTP, so make sure that's running too:
    uvicorn api.main:app --reload --port 8000

When deployed to Streamlit Cloud, set a secret named API_URL pointing to your
live backend (e.g. https://your-app.onrender.com/predict) in the app's Settings > Secrets.
"""
import os
import streamlit as st
import requests


def get_api_url():
    try:
        return st.secrets["API_URL"]
    except Exception:
        return os.environ.get("API_URL", "http://localhost:8000/predict")


API_URL = get_api_url()

st.set_page_config(page_title="Pneumonia X-Ray Detector", page_icon="🫁", layout="centered")

st.title("🫁 Pneumonia X-Ray Detector")
st.write(
    "Upload a chest X-ray image and the model will predict whether it shows "
    "signs of pneumonia. This is a machine-learning demo, **not a medical diagnosis** — "
    "always consult a qualified doctor for real diagnoses."
)

uploaded_file = st.file_uploader("Upload a chest X-ray (JPEG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="Uploaded X-ray", use_container_width=True)

    with col2:
        with st.spinner("Analyzing X-ray... (first request after idle time can take up to a minute)"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files, timeout=90)
            except requests.exceptions.RequestException as e:
                st.error(
                    f"Can't reach the prediction API right now ({type(e).__name__}). "
                    "If this just deployed, the backend may still be starting up — try again in a minute."
                )
                st.stop()

        if response.status_code != 200:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            st.error(f"Error: {detail}")
        else:
            result = response.json()
            label = result["label"]
            confidence = result["confidence"]

            if label == "PNEUMONIA":
                st.error("### Result: PNEUMONIA detected")
            else:
                st.success("### Result: NORMAL")

            st.metric("Confidence", f"{confidence * 100:.1f}%")

            st.write("Class probabilities:")
            st.bar_chart(result["probabilities"])

st.divider()
st.caption(
    "Model: HOG features + SVM classifier trained on the Kaggle "
    "'Chest X-Ray Images (Pneumonia)' dataset."
)