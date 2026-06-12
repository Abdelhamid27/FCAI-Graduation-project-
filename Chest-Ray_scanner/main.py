"""
Chest X-Ray Pneumonia Detection API
====================================
POST /predict  - Receives an X-Ray image and returns the diagnosis
GET  /health   - Verification endpoint to check if the API is running
GET  /         - Redirects to interactive Swagger UI
"""
 
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict
from huggingface_hub import hf_hub_download
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import os
import logging
from typing import Optional
 
# ─────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
IMG_SIZE      = 224
THRESHOLD     = 0.5
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
 
# HuggingFace Space repo info
HF_REPO_ID    = "Abdelhamid2004/chest-xray-pneumonia"
HF_FILENAME   = "model/chest_xray_model.h5"
 
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
    allow_origins=["*"],
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

    logger.info(f"📁 Working directory: {os.getcwd()}")
    logger.info(f"📁 Files in /app: {os.listdir('/app')}")

    # ── Strategy 1: Local disk (الملف اتحمل مع الـ container) ──
    local_path = "/app/model/chest_xray_model.h5"
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1_000_000:
        try:
            model = keras.models.load_model(local_path, compile=False)
            logger.info(f"✅ Model loaded from local disk: {local_path}")
            return
        except Exception as e:
            logger.warning(f"⚠️ Local load failed: {e} - trying HuggingFace Hub...")

    # ── Strategy 2: Download from HuggingFace Hub ──
    try:
        logger.info("⬇️  Downloading model from HuggingFace Hub...")
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="space",
            local_dir="/app",           # ✅ احفظه في /app مباشرة
        )
        logger.info(f"📥 Downloaded to: {model_path}")
        model = keras.models.load_model(model_path, compile=False)
        logger.info("✅ Model loaded successfully from HuggingFace Hub")
    except Exception as e:
        logger.error(f"❌ Failed to load model from Hub: {e}", exc_info=True)
 
# ─────────────────────────────────────────
# Response Schemas
# ─────────────────────────────────────────
class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    diagnosis: str
    confidence_percent: str
    clinical_note: str
 
class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
 
# ─────────────────────────────────────────
# Helper: Preprocess Image
# ─────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
 
    if img is None:
        raise ValueError("Could not decode image. Please ensure the file is a valid image format.")
 
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img
 
# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
 
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")
 
 
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(status="ok", model_loaded=model is not None)
 
 
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="Chest X-Ray image file in JPEG or PNG format")):
    """
    Accepts a chest X-Ray image and returns the diagnostic evaluation:
    - **Normal** → Healthy lungs
    - **Pneumonia** → Lungs infected with Pneumonia
    """
 
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Verify the model file exists and restart the server.")
 
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type ({file.content_type}). Allowed formats: JPEG, PNG")
 
    try:
        image_bytes = await file.read()
        img_array   = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during image preprocessing.")
 
    try:
        raw_score = float(model.predict(img_array, verbose=0)[0][0])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during model prediction.")
 
    is_pneumonia   = raw_score > THRESHOLD
    diagnosis      = "Pneumonia" if is_pneumonia else "Normal"
    confidence     = raw_score if is_pneumonia else (1.0 - raw_score)
    confidence_pct = confidence * 100

    if is_pneumonia:
        if confidence_pct >= 90:
            clinical_note = "High confidence detection of Pneumonia. Immediate medical consultation is strongly recommended."
        elif confidence_pct >= 70:
            clinical_note = "Moderate confidence detection of Pneumonia. Please consult a physician for further evaluation."
        else:
            clinical_note = "Low confidence detection of Pneumonia. The result is inconclusive. A radiologist review is advised."
    else:
        if confidence_pct >= 90:
            clinical_note = "High confidence result. Lungs appear Normal with no significant signs of Pneumonia detected."
        elif confidence_pct >= 70:
            clinical_note = "Moderate confidence result. Lungs appear mostly Normal. Follow-up may be considered if symptoms persist."
        else:
            clinical_note = "Low confidence result. No clear signs of Pneumonia, but the result is inconclusive. Further imaging may be needed."

    logger.info(f"Prediction: {diagnosis} | confidence={confidence_pct:.1f}%")

    return PredictionResponse(
        diagnosis          = diagnosis,
        confidence_percent = f"{confidence_pct:.1f}%",
        clinical_note      = clinical_note,
    )