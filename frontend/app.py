import streamlit as st
import requests
import re
from pymongo import MongoClient
import datetime

# MongoDB Connection (using MongoDB Atlas URL)
client = MongoClient("mongodb+srv://sushil0303:skdb0303@cluster0.x5vy7oo.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["diabetes_db"]
collection = db["patient_records"]

# Save patient data to MongoDB
def save_patient_data(patient_data):
    patient_data['timestamp'] = datetime.datetime.now()
    
    # Check if the patient already exists
    existing_patient = collection.find_one({"phone_number": patient_data["phone_number"]})
    
    if existing_patient:
        # If the patient exists, push the new record to the 'history' array
        collection.update_one(
            {"phone_number": patient_data["phone_number"]}, 
            {"$push": {"history": patient_data}}
        )
    else:
        # If the patient does not exist, create a new document
        collection.insert_one({
            "phone_number": patient_data["phone_number"],
            "history": [patient_data]  # Store records as a list
        })

# Retrieve patient history from MongoDB
def get_patient_history(phone_number):
    patient_history = collection.find_one({"phone_number": phone_number})
    if patient_history:
        return patient_history.get("history", [])
    return []

# Visualizing historical data using streamlit charts
def plot_patient_history(patient_history):
    import matplotlib.pyplot as plt

    glucose_levels = [record['Glucose'] for record in patient_history]
    timestamps = [record['timestamp'].strftime('%Y-%m-%d') for record in patient_history]

    plt.figure(figsize=(10,6))
    plt.plot(timestamps, glucose_levels, marker='o', color='b', label="Glucose Levels")
    plt.title('Glucose Levels Over Time')
    plt.xlabel('Date')
    plt.ylabel('Glucose Level')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    st.pyplot(plt)

# Streamlit UI
st.title("AI-Powered Diabetes Diagnosis")

# Adding a container to keep the title separated from the columns
st.markdown("""
    <style>
    .main-container {
        display: flex;
        justify-content: center;
        padding-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Create two columns: one for the left and one for the right side
left_column, right_column = st.columns([2, 3])  # You can adjust the width ratio (e.g., 2:3)

# Phone number input in the left column
with left_column:
    st.write("Please enter the patient's phone number:")
    phone_number = st.text_input("Phone Number")

# Validate phone number
if phone_number and not re.match(r'^\+?\d{10,15}$', phone_number):
    st.error("Please enter a valid phone number.")

if phone_number:
    # Check if the patient exists in the database
    patient_history = get_patient_history(phone_number)

    if len(patient_history) > 0:  # Patient exists
        with right_column:
            st.write("Patient found! Here is their previous history:")

            # Display previous records and graph
            for record in patient_history:
                st.write(f"Prediction: {record['prediction']}, Date: {record['timestamp']}, Glucose: {record['Glucose']}")

            # Plot historical glucose levels
            plot_patient_history(patient_history)

            # Predict current condition
            st.write("Enter the current data to make a new prediction:")

            # Current input form
            pregnancies = st.number_input("Pregnancies", 0, 20, step=1)
            glucose = st.number_input("Glucose", 0.0, 300.0)
            blood_pressure = st.number_input("Blood Pressure", 0.0, 200.0)
            skin_thickness = st.number_input("Skin Thickness", 0.0, 100.0)
            insulin = st.number_input("Insulin", 0.0, 900.0)
            bmi = st.number_input("BMI", 0.0, 70.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
            age = st.number_input("Age", 1, 120, step=1)

            if st.button("Predict"):
                # Sending data to the backend (FastAPI) for prediction
                payload = {
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": blood_pressure,
                    "SkinThickness": skin_thickness,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "DiabetesPedigreeFunction": dpf,
                    "Age": age
                }
                response = requests.post("http://localhost:8000/predict", json=payload)
                prediction = response.json()['prediction']

                st.success(f"The model predicts: {prediction}")

                # Save the new data to MongoDB along with prediction
                patient_data = {
                    "phone_number": phone_number,
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": blood_pressure,
                    "SkinThickness": skin_thickness,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "DiabetesPedigreeFunction": dpf,
                    "Age": age,
                    "prediction": prediction
                }

                save_patient_data(patient_data)

            # Add button to get health report
            if st.button("Get Health Report"):
                response = requests.post("http://localhost:8000/generate_report", json={"phone_number": phone_number})
                if response.status_code == 200:
                    report = response.json()['health_report']
                    st.subheader("🩺 Patient Health Report:")
                    st.success(report)
                else:
                    st.error("Failed to generate health report.")

    else:  # Patient does not exist, request data
        with right_column:
            st.write("Patient not found. Please enter the following details to register:")

            # Input form for new patient
            pregnancies = st.number_input("Pregnancies", 0, 20, step=1)
            glucose = st.number_input("Glucose", 0.0, 300.0)
            blood_pressure = st.number_input("Blood Pressure", 0.0, 200.0)
            skin_thickness = st.number_input("Skin Thickness", 0.0, 100.0)
            insulin = st.number_input("Insulin", 0.0, 900.0)
            bmi = st.number_input("BMI", 0.0, 70.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
            age = st.number_input("Age", 1, 120, step=1)

            if st.button("Register and Predict"):
                # Send data to FastAPI for prediction
                payload = {
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": blood_pressure,
                    "SkinThickness": skin_thickness,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "DiabetesPedigreeFunction": dpf,
                    "Age": age
                }
                response = requests.post("http://localhost:8000/predict", json=payload)
                prediction = response.json()['prediction']

                st.success(f"The model predicts: {prediction}")

                # Save new patient data to MongoDB
                patient_data = {
                    "phone_number": phone_number,
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": blood_pressure,
                    "SkinThickness": skin_thickness,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "DiabetesPedigreeFunction": dpf,
                    "Age": age,
                    "prediction": prediction
                }

                save_patient_data(patient_data)
