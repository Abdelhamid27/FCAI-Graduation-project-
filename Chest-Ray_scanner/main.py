"""
Chest X-Ray Pneumonia Detection API
====================================
POST /predict  — يستقبل صورة X-Ray ويرجع التشخيص
GET  /health   — للتحقق أن الـ API شغال
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
# Logging
# ─────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Constants  (نفس القيم اللي في الـ notebook)
# ─────────────────────────────────────────
IMG_SIZE      = 224
THRESHOLD     = 0.5
MODEL_PATH    = os.getenv("MODEL_PATH", "model/chest_xray_model.h5")
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}

# ─────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────
app = FastAPI(
    title="Chest X-Ray Pneumonia Detection API",
    description="API لتشخيص الالتهاب الرئوي من صور الأشعة السينية باستخدام CNN و Transfer Learning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # عدّلها في الـ production
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
    raw_score: float         # الـ sigmoid output الخام
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# ─────────────────────────────────────────
# Helper: preprocess image  (نفس منطق الـ notebook)
# ─────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    تحويل الـ bytes لـ numpy array جاهز للـ model
    - resize → 224×224
    - rescale → 1/255  (نفس val_datagen)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("تعذّر قراءة الصورة، تأكد أن الملف صورة صحيحة")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0      # rescale
    img = np.expand_dims(img, axis=0)        # (1, 224, 224, 3)
    return img

# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """التحقق من أن الـ API والـ model شغالين"""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="صورة X-Ray بصيغة JPEG أو PNG")):
    """
    استقبال صورة X-Ray وإرجاع تشخيص:
    - **Normal**    → رئة سليمة
    - **Pneumonia** → التهاب رئوي
    """

    # ── 1. Validate model ──────────────────
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="الموديل مش محمّل. تأكد من وجود ملف الـ model وأعد تشغيل الـ server."
        )

    # ── 2. Validate file type ──────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"نوع الملف غير مدعوم ({file.content_type}). المسموح: JPEG, PNG"
        )

    # ── 3. Read & preprocess ───────────────
    try:
        image_bytes = await file.read()
        img_array   = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=500, detail="خطأ أثناء معالجة الصورة")

    # ── 4. Predict ─────────────────────────
    try:
        raw_score = float(model.predict(img_array, verbose=0)[0][0])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="خطأ أثناء التنبؤ")

    # ── 5. Interpret result ────────────────
    #  sigmoid > 0.5  →  Pneumonia (class 1)
    #  sigmoid ≤ 0.5  →  Normal    (class 0)
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
