# Chest X-Ray API 🫁

An API for pneumonia detection from chest X-ray images using a trained deep learning model.

---

## 🗂️ Project Structure

```

chest_xray_api/
├── main.py            # FastAPI application
├── save_model.py      # Model saving script
├── requirements.txt   # Project dependencies
└── model/
    └── chest_xray_model.h5   # Trained model file

```

---

## ⚙️ Setup Instructions

### 1. Save the Trained Model

At the end of your training notebook, add:

```python
import os
os.makedirs("model", exist_ok=True)
model_pretrained.save("model/chest_xray_model.h5")
```

Or simply run `save_model.py`.

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Start the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔌 API Endpoints

### `GET /health`

Checks whether the API and model are running correctly.

**Response:**

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### `POST /predict`

Upload a chest X-ray image and receive a diagnosis prediction.

**Request:**

- `Content-Type: multipart/form-data`
- Field: `file` (JPEG or PNG image)

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

| Field | Description |
|--------|------------|
| `diagnosis` | `"Normal"` or `"Pneumonia"` |
| `confidence` | Confidence score (0 → 1) |
| `confidence_percent` | Confidence score as a percentage |
| `raw_score` | Raw sigmoid output from the model |

---

## 🧪 API Testing

### Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@xray_image.jpg"
```

### Using Python

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

## 📝 Swagger Documentation

After starting the server, open:

```

http://localhost:8000/docs

```

---

## 🌍 Environment Variables

| Variable | Default Value | Description |
|----------|--------------|-------------|
| `MODEL_PATH` | `model/chest_xray_model.h5` | Path to the trained model file |
