# 🫁 Chest X-Ray Pneumonia Detection API

A deep learning-powered REST API that analyzes chest X-ray images and detects the presence of **Pneumonia** using Convolutional Neural Networks and Transfer Learning.

---

## 📌 Overview

This project builds an image classification system trained on chest X-ray images to distinguish between **Normal** lungs and lungs infected with **Pneumonia**.

Three modeling approaches were explored during training:
1. **Custom CNN** — a simple convolutional network built from scratch
2. **Transfer Learning** — using ResNet152V2 (pretrained on ImageNet) as a frozen feature extractor
3. **Fine Tuning** — unfreezing the last layers of ResNet152V2 and continuing training on the X-ray dataset

The final deployed model uses the **Fine Tuning** approach, which achieved the best performance.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | **90.7%** |
| ROC-AUC | **97.2%** |
| Precision (Normal) | 96% |
| Recall (Normal) | 79% |
| Precision (Pneumonia) | 88% |
| Recall (Pneumonia) | 98% |
| F1-Score (Overall) | 90% |

---

## 🗂️ Dataset

- **Source:** [Labeled Chest X-Ray Images — Kaggle](https://www.kaggle.com/tolgadincer/labeled-chest-xray-images)
- **Total Images:** 5,856 validated chest X-ray images
- **Patient Age:** 1 to 5 years old (pediatric cohort)
- **Source Hospital:** Guangzhou Women and Children's Medical Center, Guangzhou
- **Split:** 80% training / 20% validation / separate test set
- **Classes:** Normal, Pneumonia

---

## 🏗️ Model Architecture

- **Base Model:** ResNet152V2 (pretrained on ImageNet, fine-tuned)
- **Input Shape:** 224 × 224 × 3 (RGB)
- **Output:** Binary classification (Normal / Pneumonia)
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Data Augmentation:** Random flips, rotations, and zoom applied during training

---

## 🚀 API Endpoints

### `POST /predict`
Upload a chest X-ray image and receive a diagnosis.

**Request:**
- Content-Type: `multipart/form-data`
- Field: `file` — JPEG or PNG image

**Response:**
```json
{
  "diagnosis": "Pneumonia",
  "confidence_percent": "87.3%",
  "clinical_note": "Moderate confidence detection of Pneumonia. Please consult a physician for further evaluation."
}
```

**Clinical Note Logic:**

| Confidence | Note |
|------------|------|
| ≥ 90% | High confidence result |
| 70% – 90% | Moderate confidence result |
| < 70% | Low confidence — inconclusive, further review advised |

---

### `GET /health`
Check if the API is running and the model is loaded.

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `GET /`
Redirects to the interactive **Swagger UI** documentation.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI |
| Deep Learning | TensorFlow 2.15 / Keras |
| Base Model | ResNet152V2 |
| Image Processing | OpenCV |
| Server | Uvicorn |
| Deployment | HuggingFace Spaces (Docker) |

---

## 🐳 Running Locally with Docker

```bash
# Clone the repository
git clone https://huggingface.co/spaces/Abdelhamid2004/chest-xray-pneumonia
cd chest-xray-pneumonia

# Build the Docker image
docker build -t chest-xray-api .

# Run the container
docker run -p 7860:7860 chest-xray-api
```

Then open [http://localhost:7860/docs](http://localhost:7860/docs) in your browser.

---

## 📁 Project Structure

```
chest-xray-pneumonia/
├── main.py                                        # FastAPI application
├── requirements.txt                               # Python dependencies
├── Dockerfile                                     # Docker configuration
├── save_model.py                                  # Script to convert model format
├── chest-x-ray-pneumonia-cnn-transfer-learning.ipynb  # Training notebook
└── model/
    ├── chest_xray_model.h5                        # Deployed model (Keras H5)
    └── chest_xray_model.keras                     # Model in Keras format
```

---

## ⚠️ Medical Disclaimer

This tool is intended for **educational and research purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified radiologist or physician for any medical decisions.

---

## 👤 Author

**Abdelhamid** — Faculty of Computers and Artificial Intelligence
