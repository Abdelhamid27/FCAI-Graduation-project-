from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# 1. Load the artifacts
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')
gender_encoder = joblib.load('gender_encoder.pkl')

class PredictionInput(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int

@app.post("/predict")
async def predict(data: PredictionInput):
    # A. Preprocessing
    g_encoded = gender_encoder.transform([data.gender])[0]
    smk_options = {'current': 0, 'ever': 0, 'former': 0, 'never': 0, 'not current': 0}
    if data.smoking_history in smk_options:
        smk_options[data.smoking_history] = 1

    # B. Scaling the 4 numerical features
    numerical_features = [[data.age, data.bmi, data.HbA1c_level, data.blood_glucose_level]]
    scaled_values = scaler.transform(numerical_features)[0]
    
    # C. Assemble all 12 features for the model
    final_features = [
        g_encoded, scaled_values[0], data.hypertension, data.heart_disease,
        scaled_values[1], scaled_values[2], scaled_values[3],
        smk_options['current'], smk_options['ever'], smk_options['former'], 
        smk_options['never'], smk_options['not current']
    ]
    
    # D. Get prediction and probability
    prediction = model.predict([final_features])[0]
    probability = model.predict_proba([final_features])[0][1] * 100

    # E. Risk Assessment Logic
    if probability < 20:
        risk_level = "Low Risk"
    elif 20 <= probability < 50:
        risk_level = "Moderate Risk"
    elif 50 <= probability < 80:
        risk_level = "Relatively High Risk"
    else:
        risk_level = "Very High Risk"

    # F. Clinical Insights (The Analysis part you liked)
    insights = []
    if data.HbA1c_level >= 6.5: insights.append("HbA1c level is in the diabetic range.")
    if data.bmi >= 30: insights.append("BMI indicates obesity, which is a significant risk factor.")
    if data.blood_glucose_level > 140: insights.append("Elevated blood glucose level detected.")
    if data.age > 60: insights.append("Age factor increases the risk of chronic complications.")
    
    if not insights:
        insights.append("All primary clinical markers are within stable limits.")

    return {
        "prediction": "Diabetic" if prediction == 1 else "Non-Diabetic",
        "risk_percentage": f"{probability:.2f}%",
        "risk_assessment": risk_level,
        "clinical_analysis": insights,
        "recommendation": "Consult a physician for clinical diagnosis." if prediction == 1 else "Maintain a healthy lifestyle and regular check-ups."
    }