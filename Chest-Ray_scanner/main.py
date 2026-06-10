"""
Chest X-Ray Pneumonia Detection API
====================================
POST /predict  — Receives an X-Ray image and returns the diagnosis
GET  /health   — Verification endpoint to check if the API is running
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import io
import os
import logging
from typing import Optional

# ─────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Constants (Matches Notebook Configurations)
# ─────────────────────────────────────────
IMG_SIZE      = 224
THRESHOLD     = 0.5
MODEL_PATH    = os.getenv("MODEL_PATH", "model/chest_xray_model.h5")
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}

# ─────────────────────────────────────────
# FastAPI App Initialization
# ─────────────────────────────────────────
app = FastAPI(
    title="Chest X-Ray Pneumonia Detection API",
    description="API for diagnosing pneumonia from chest X-ray images using CNN and Transfer Learning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Update this in production environments
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Load Model on Startup
# ─────────────────────────────────────────
model: Optional[keras.Model] = None

@app.on_event("startup")
async def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        logger.warning(
            f"⚠️  Model file not found at '{MODEL_PATH}'. "
            "Set MODEL_PATH env variable or place the .h5 file in model/"
        )
        return
    try:
        model = keras.models.load_model(MODEL_PATH)
        logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")

# ─────────────────────────────────────────
# Response Schemas
# ─────────────────────────────────────────
class PredictionResponse(BaseModel):
    diagnosis: str           # "Normal" | "Pneumonia"
    confidence: float        # 0.0 → 1.0
    confidence_percent: str  # "92.5%"
    raw_score: float         # Raw sigmoid output score
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# ─────────────────────────────────────────
# Helper: Preprocess Image
# ─────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Converts raw image bytes to a processed numpy array ready for model inference.
    - resize → 224×224
    - rescale → 1/255 (Matches val_datagen logic)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image. Please ensure the file is a valid image format.")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0      # Rescale
    img = np.expand_dims(img, axis=0)        # Shape: (1, 224, 224, 3)
    return img

# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Verifies that the API server is healthy and the ML model is successfully loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="Chest X-Ray image file in JPEG or PNG format")):
    """
    Accepts a chest X-Ray image and returns the diagnostic evaluation:
    - **Normal** → Healthy lungs
    - **Pneumonia** → Lungs infected with Pneumonia
    """

    # ── 1. Validate Model Availability ──────
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Verify the model file exists and restart the server."
        )

    # ── 2. Validate File Type ──────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type ({file.content_type}). Allowed formats: JPEG, PNG"
        )

    # ── 3. Read & Preprocess Image ─────────
    try:
        image_bytes = await file.read()
        img_array   = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during image preprocessing.")

    # ── 4. Model Inference ─────────────────
    try:
        raw_score = float(model.predict(img_array, verbose=0)[0][0])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during model prediction.")

    # ── 5. Interpret Output Results ────────
    # Sigmoid rules: score > 0.5 → Pneumonia (Class 1) | score <= 0.5 → Normal (Class 0)
    is_pneumonia = raw_score > THRESHOLD
    diagnosis    = "Pneumonia" if is_pneumonia else "Normal"
    confidence   = raw_score if is_pneumonia else (1.0 - raw_score)

    logger.info(f"Prediction: {diagnosis} | score={raw_score:.4f} | confidence={confidence:.4f}")

    return PredictionResponse(
        diagnosis         = diagnosis,
        confidence        = round(confidence, 4),
        confidence_percent= f"{confidence * 100:.1f}%",
        raw_score         = round(raw_score, 6),
        model_version     = "1.0.0",
    )