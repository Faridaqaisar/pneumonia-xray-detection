"""
Streamlit UI for the pneumonia detector.

Run with:
    streamlit run app/streamlit_app.py

This talks to the FastAPI backend over HTTP, so make sure that's running too:
    uvicorn api.main:app --reload --port 8000
"""
import streamlit as st
import requests
import os
API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")

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
        with st.spinner("Analyzing X-ray..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files, timeout=30)
            except requests.exceptions.ConnectionError:
                st.error(
                    "Can't reach the prediction API. Make sure it's running:\n\n"
                    "`uvicorn api.main:app --reload --port 8000`"
                )
                st.stop()

        if response.status_code != 200:
            st.error(f"Error: {response.json().get('detail', response.text)}")
        else:
            result = response.json()
            label = result["label"]
            confidence = result["confidence"]

            if label == "PNEUMONIA":
                st.error(f"### Result: PNEUMONIA detected")
            else:
                st.success(f"### Result: NORMAL")

            st.metric("Confidence", f"{confidence * 100:.1f}%")

            st.write("Class probabilities:")
            st.bar_chart(result["probabilities"])

st.divider()
st.caption(
    "Model: HOG features + SVM classifier trained on the Kaggle "
    "'Chest X-Ray Images (Pneumonia)' dataset."
)
