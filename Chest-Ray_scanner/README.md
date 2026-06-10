# Chest X-Ray API 🫁

API لتشخيص الالتهاب الرئوي من صور الأشعة السينية.

---

## 🗂️ هيكل المشروع

```
chest_xray_api/
├── main.py            ← الـ API (FastAPI)
├── save_model.py      ← سكريبت حفظ الـ model
├── requirements.txt   ← المكتبات
└── model/
    └── chest_xray_model.h5   ← ملف الـ model (بتعمله أنت)
```

---

## ⚙️ خطوات الإعداد

### 1. احفظ الـ Model من الـ Notebook

في آخر الـ notebook بعد التدريب أضف:

```python
import os
os.makedirs("model", exist_ok=True)
model_pretrained.save("model/chest_xray_model.h5")
```

أو شغّل ملف `save_model.py` مباشرة.

---

### 2. ثبّت المكتبات

```bash
pip install -r requirements.txt
```

---

### 3. شغّل الـ Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔌 الـ Endpoints

### `GET /health`
للتحقق من أن الـ API والـ model شغالين.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### `POST /predict`
ترسل صورة X-Ray وتاخد التشخيص.

**Request:**
- `Content-Type: multipart/form-data`
- Field: `file` ← الصورة (JPEG أو PNG)

**Response:**
```json
{
  "diagnosis": "Pneumonia",
  "confidence": 0.9231,
  "confidence_percent": "92.3%",
  "raw_score": 0.923145,
  "model_version": "1.0.0"
}
```

| حقل | المعنى |
|-----|--------|
| `diagnosis` | `"Normal"` أو `"Pneumonia"` |
| `confidence` | نسبة الثقة (0 → 1) |
| `confidence_percent` | نفس الـ confidence بصيغة % |
| `raw_score` | الـ sigmoid output الخام |

---

## 🧪 اختبار الـ API

### باستخدام cURL:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@xray_image.jpg"
```

### باستخدام Python:
```python
import requests

with open("xray_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("xray.jpg", f, "image/jpeg")}
    )

print(response.json())
```

---

## 📝 Swagger UI

بعد تشغيل الـ server، افتح:

```
http://localhost:8000/docs
```

---

## متغيرات البيئة

| متغير | الافتراضي | الوصف |
|-------|-----------|-------|
| `MODEL_PATH` | `model/chest_xray_model.h5` | مسار ملف الـ model |
