import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib  # For loading the saved model

# Title and description
st.title("Osteoporosis Risk Prediction Tool")

# Header with group and full member names in compact format (2 lines)
st.markdown("""
    <div style="background-color: #f1f1f1; padding: 3px 10px; text-align: center; font-size: 12px; color: #555; border-radius: 10px; margin-bottom: 10px">
        <strong>Group 4</strong><br>
        Chong Wai Fun, Dhiviyanandhini A/P V Kumar, Lisa Ho Yen Xin, Nur Maisarah Binti Jalalulail, Nurul Hafizah Binti Zaini, Tan Siao Shuen
    </div>
""", unsafe_allow_html=True)

# Add a brief intro with an impactful statement
st.markdown("""
    <h3 style="color: #FF6347;">What is Osteoporosis?</h3>
    <p style="font-size: 16px; text-align: justify;">Osteoporosis is a condition that weakens bones, making them fragile and more prone to fractures. Often referred to as the "silent disease," it progresses without noticeable symptoms until a fracture occurs. Early detection is crucial to preventing serious complications and improving bone health.</p>
""", unsafe_allow_html=True)

# Introduction to the application and its function with prevention message
st.markdown("""
    <h4 style="color: #20B2AA;">Predicting Osteoporosis</h4>
    <p style="font-size: 16px; text-align: justify;">This tool allows you to assess your risk of osteoporosis based on your personal information and medical history. Identifying risk factors early can help you take proactive steps toward better bone health. The application uses a <b>pre-trained K-Nearest Neighbors (KNN) model</b> to predict the likelihood of osteoporosis based on your input attributes. </p>
""", unsafe_allow_html=True)

# Custom CSS for styling
st.markdown("""
    <style>
        .age-title {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Function to convert input to binary (0 or 1)
def binary_choice(value):
    return 1 if value == "Yes" else 0

# Layout for the input fields
col1, col2 = st.columns([2, 2])  # Adjust the ratio as needed, ensuring space

# Personal Information (Age, Gender, Race/Ethnicity, Body Weight) - in first column
with col1:
    st.markdown("#### Personal Information")
    age = st.slider("Age", 0, 100, 20)
    gender = st.selectbox("Gender", ("Male", "Female"))
    race = st.selectbox("Race/Ethnicity", ("African American", "Caucasian", "Other"))
    body_weight = st.selectbox("Underweight", ("No", "Yes"))

# Lifestyle Factors (Physical Activity, Smoking, Alcohol Consumption) - in second column
with col2:
    st.markdown("#### Lifestyle Factors")
    physical_activity = st.selectbox("Sedentary Lifestyle", ("No", "Yes"))
    smoking = st.selectbox("Smoking", ("No", "Yes"))
    alcohol_consumption = st.selectbox("Moderate Alcohol Consumption", ("No", "Yes"))

# Health and Medical Conditions - Title Spanning Both Columns
st.markdown("### Health and Medical Conditions", unsafe_allow_html=True)

# Creating two columns for Health and Medical Conditions but splitting the content
col3, col4 = st.columns([2, 2])

# Health Conditions (Hyperthyroidism, Rheumatoid Arthritis, etc.) - in third column
with col3:
    medical_hyperthyroidism = st.selectbox("Hyperthyroidism", ("No", "Yes"))
    medical_rheumatoid_arthritis = st.selectbox("Rheumatoid Arthritis", ("No", "Yes"))
    hormonal_changes = st.selectbox("Postmenopausal", ("No", "Yes"))
    family_history = st.selectbox("Family History", ("No", "Yes"))

# Medical History (Calcium Intake, Vitamin D Intake, Medications, Prior Fractures) - in fourth column
with col4:
    calcium_intake = st.selectbox("Low Calcium Intake", ("No", "Yes"))
    vitamin_d_intake = st.selectbox("Insufficient Vitamin D Intake", ("No", "Yes"))
    medications_corticosteroids = st.selectbox("Corticosteroid Use", ("No", "Yes"))
    prior_fractures = st.selectbox("Prior Fractures", ("No", "Yes"))

# Organize data into a dictionary for model input
data = {
    'Age': age,
    'Race/Ethnicity-African American': 1 if race == "African American" else 0,
    'Race/Ethnicity-Caucasian': 1 if race == "Caucasian" else 0,
    'Medical Conditions-Hyperthyroidism': binary_choice(medical_hyperthyroidism),
    'Medical Conditions-Rheumatoid Arthritis': binary_choice(medical_rheumatoid_arthritis),
    'Gender-Female': 1 if gender == "Female" else 0,
    'Hormonal Changes-Postmenopausal': binary_choice(hormonal_changes),
    'Family History-Yes': binary_choice(family_history),
    'Body Weight-Underweight': binary_choice(body_weight),
    'Calcium Intake-Low': binary_choice(calcium_intake),
    'Vitamin D Intake-Insufficient': binary_choice(vitamin_d_intake),
    'Physical Activity-Sedentary': binary_choice(physical_activity),
    'Smoking-Yes': binary_choice(smoking),
    'Alcohol Consumption-Moderate': binary_choice(alcohol_consumption),
    'Medications-Corticosteroids': binary_choice(medications_corticosteroids),
    'Prior Fractures-Yes': binary_choice(prior_fractures)
}

# Convert input data into a DataFrame
input_df = pd.DataFrame(data, index=[0])

# Load the saved model and scaler
model = joblib.load("knn_model.pkl")  # Replace with the path to your saved model

# Prediction
prediction = model.predict(input_df)

# Display results in a clean layout with improved visibility
st.subheader("Prediction", anchor="prediction")

# Make the result highly visible with bold, colored text and background
prediction_result = "Yes" if prediction[0] == 1 else "No"

# Set colors based on prediction outcome
if prediction_result == "Yes":
    result_color = "red"
    icon = "⚠️"  # Warning icon
    background_color = "#ffcccc"  # Light red background
else:
    result_color = "green"
    icon = "✅"  # Checkmark icon
    background_color = "#d9f7d9"  # Light green background

# Display the result with the custom styling
st.markdown(f"""
    <div style="background-color: {background_color}; padding: 20px; border-radius: 10px; text-align: center;">
        <h3 style="color: {result_color}; font-size: 24px; font-weight: bold;">
            {icon} Osteoporosis Prediction: <span style="color: {result_color}; font-size: 28px;">{prediction_result}</span>
        </h3>
    </div>
""", unsafe_allow_html=True)