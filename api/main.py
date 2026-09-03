"""
FastAPI backend that serves the trained pneumonia model.

Run with:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health         -> simple liveness check
    POST /predict         -> upload an X-ray image, get back a prediction

A plain REST endpoint (rather than a websocket) is used here because each
prediction is a single independent request/response — there's no ongoing
stream of data to justify a persistent connection. If you later want
live/streaming predictions (e.g. video frames), a websocket endpoint can
be added the same way.
"""
import sys
import os

# Allow running this file directly (uvicorn api.main:app) from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io

from src.predict import predict_image

app = FastAPI(title="Pneumonia X-Ray Detector API")

# Allow the Streamlit frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPEG or PNG X-ray image.",
        )

    contents = await file.read()
    try:
        result = predict_image(io.BytesIO(contents))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    return result
