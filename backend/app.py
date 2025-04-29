from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from pymongo import MongoClient

app = FastAPI()

# Load models
lstm_model = load_model('model/health_lstm_model.h5')  # LSTM Model for health status prediction
text_generator = pipeline('text-generation', model='distilgpt2')  # Text generation model for reports
diabetes_model = joblib.load('C:/Users/SUSHIL KUMAR/Desktop/diabetes_diagnosis_project/backend/model/diabetes_model.pkl')  # Sklearn model for diabetes prediction

# MongoDB Connection
client = MongoClient("mongodb+srv://sushil0303:skdb0303@cluster0.x5vy7oo.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["diabetes_db"]
collection = db["patient_records"]

# Pydantic models for data validation
class PhoneRequest(BaseModel):
    phone_number: str

class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Route to generate a health report based on phone number
@app.post("/generate_report")
def generate_health_report_api(data: PhoneRequest):
    patient = collection.find_one({"phone_number": data.phone_number})
    if not patient:
        return {"error": "Patient not found."}

    history = patient['history']

    # Prepare data for prediction (ensure valid fields)
    features = []
    for record in history:
        features.append([
            record.get('Glucose', 0),
            record.get('BloodPressure', 0),
            record.get('BMI', 0)
        ])

    X = np.array(features)
    X = X.reshape((1, X.shape[0], X.shape[1]))  # Reshape for LSTM

    # Predict health status
    health_status = lstm_model.predict(X)
    status = "stable" if health_status[0][0] < 0.5 else "worsening"

    # Generate a detailed report using text generation model
    prompt = f"Patient health status is {status}. Write a detailed medical report."
    generated = text_generator(prompt, max_length=150, do_sample=True)[0]['generated_text']

    return {"health_report": generated}

# Route to predict diabetes status based on input features
@app.post("/predict")
def predict(data: DiabetesInput):
    input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure,
                            data.SkinThickness, data.Insulin, data.BMI,
                            data.DiabetesPedigreeFunction, data.Age]])
    prediction = diabetes_model.predict(input_data)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    return {"prediction": result}
