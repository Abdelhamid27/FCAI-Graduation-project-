# 🫀 Heart Disease Prediction — Cardiovascular Risk Detection API

A machine learning system that predicts cardiovascular disease risk from patient health data, combining **KModes Clustering** with **Random Forest Classification** and served via a **FastAPI** REST API.

---

## 📌 Overview

This project analyzes a dataset of **70,000 patient records** to detect the presence of cardiovascular disease. The pipeline includes data cleaning, feature engineering, unsupervised clustering by gender, and supervised classification using multiple ML algorithms — with **Random Forest** selected as the final model.

---

## 🗂️ Project Structure

```
├── Heart.ipynb              # Full ML pipeline (EDA, clustering, training)
├── main.py                  # FastAPI prediction server
├── heart_model.pkl          # Trained Random Forest model (generated after training)
├── cardio_processed.csv     # Dataset (70,000 records)
└── README.md
```

---

## 📊 Dataset

**Source:** Cardiovascular Disease Dataset  
**Size:** 70,000 patient records × 13 features

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Patient age (years) | Numerical |
| `gender` | 1 = Female, 2 = Male | Categorical |
| `height` | Height (cm) | Numerical |
| `weight` | Weight (kg) | Numerical |
| `ap_hi` | Systolic blood pressure | Numerical |
| `ap_lo` | Diastolic blood pressure | Numerical |
| `cholesterol` | 1 = Normal, 2 = Above normal, 3 = Well above normal | Categorical |
| `gluc` | Glucose level (same scale as cholesterol) | Categorical |
| `smoke` | Smoker: 0 = No, 1 = Yes | Binary |
| `alco` | Alcohol intake: 0 = No, 1 = Yes | Binary |
| `active` | Physically active: 0 = No, 1 = Yes | Binary |
| `cardio` | **Target** — Cardiovascular disease: 0 = No, 1 = Yes | Binary |

---

## ⚙️ ML Pipeline

### 1. Data Cleaning
- Removed height/weight outliers (below 2.5% and above 97.5% quantile)
- Removed invalid blood pressure records (diastolic > systolic)
- Removed blood pressure outliers

### 2. Feature Engineering

| New Feature | Description |
|-------------|-------------|
| `age_bin` | Age grouped into 5-year intervals |
| `BMI_Class` | Body Mass Index classified into 6 levels (Underweight → Extreme Obesity) |
| `MAP_Class` | Mean Arterial Pressure classified into 6 levels (Low → Hypertensive Crisis) |

**MAP Formula:**  `MAP = (ap_hi + 2 × ap_lo) / 3`

### 3. KModes Clustering
- Separated data by gender (Male / Female)
- Applied **KModes with Huang initialization** to each group
- Optimal k = **2 clusters** per group (determined by Elbow Method)
- Merged clusters back into a single DataFrame as the `Cluster` feature

### 4. Classification Models Compared

| Model | Test Accuracy | AUC |
|-------|-------------|-----|
| Naive Bayes | — | — |
| Decision Tree | — | 0.90 |
| **Random Forest** ✅ | **Best** | — |
| Logistic Regression | — | 0.87 |

**Random Forest (GridSearchCV)** was selected as the final model and exported via `joblib`.

---

## 🚀 API — Running the Server

### Requirements

```bash
pip install fastapi uvicorn joblib pandas scikit-learn
```

### Start the Server

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive docs available at: **http://127.0.0.1:8000/docs**

---

## 🔌 API Endpoint

### `POST /predict`

Accepts patient data and returns a cardiovascular disease risk assessment.

**Request Body:**

```json
{
  "gender": 2,
  "age": 50,
  "height": 170,
  "weight": 80.0,
  "ap_hi": 130,
  "ap_lo": 85,
  "cholesterol": 2,
  "gluc": 1,
  "smoke": 0,
  "active": 1
}
```

| Field | Description |
|-------|-------------|
| `gender` | 1 = Female, 2 = Male |
| `age` | Age in years |
| `height` | Height in cm |
| `weight` | Weight in kg |
| `ap_hi` | Systolic blood pressure |
| `ap_lo` | Diastolic blood pressure |
| `cholesterol` | 1 / 2 / 3 |
| `gluc` | 1 / 2 / 3 |
| `smoke` | 0 or 1 |
| `active` | 0 or 1 |

**Response:**

```json
{
  "cardio_prediction": "Positive",
  "probability_percentage": "73.45%",
  "risk_status": "High Risk",
  "medical_analysis": {
    "age_group": "50-55 years",
    "weight_status": "Overweight",
    "blood_pressure_status": "Stage 1 Hypertension"
  },
  "recommendation": "High probability of heart disease. Please consult a specialist."
}
```

---

## 🧪 Quick Test

```python
import requests

data = {
    "gender": 2,
    "age": 52,
    "height": 168,
    "weight": 85.0,
    "ap_hi": 140,
    "ap_lo": 90,
    "cholesterol": 2,
    "gluc": 1,
    "smoke": 0,
    "active": 1
}

response = requests.post("http://127.0.0.1:8000/predict", json=data)
print(response.json())
```

Or with cURL:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"gender":2,"age":52,"height":168,"weight":85.0,"ap_hi":140,"ap_lo":90,"cholesterol":2,"gluc":1,"smoke":0,"active":1}'
```

---

## 🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data-lightgrey)

- **Python 3.10**
- **FastAPI** — REST API framework
- **scikit-learn** — Random Forest, Logistic Regression, Decision Tree, Naive Bayes
- **KModes** — Unsupervised clustering for categorical data
- **Pandas / NumPy** — Data manipulation
- **Seaborn / Matplotlib** — Data visualization
- **Joblib** — Model serialization

---

## 👨‍💻 Author

Built as part of FCAI graduation project.
