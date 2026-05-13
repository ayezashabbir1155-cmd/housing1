
import streamlit as st
import pickle
import joblib
import pandas as pd

# ============================================
# Load Model
# ============================================

model = joblib.load("linear_regression_model.pkl")

# Load Preprocessor / Scaler
preprocessor = joblib.load("preprocessor_scaler.pkl")

# ============================================
# Streamlit App Title
# ============================================

st.title("🏠 Housing Price Prediction App")

st.write("Enter house details below:")

# ============================================
# User Inputs
# Replace these fields according to your dataset columns
# ============================================

area = st.number_input("Area (sq ft)", min_value=0)

bedrooms = st.number_input("Bedrooms", min_value=1)

bathrooms = st.number_input("Bathrooms", min_value=1)

stories = st.number_input("Stories", min_value=1)

parking = st.number_input("Parking Spaces", min_value=0)

mainroad = st.selectbox("Main Road Access", ["yes", "no"])

guestroom = st.selectbox("Guest Room", ["yes", "no"])

basement = st.selectbox("Basement", ["yes", "no"])

hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])

airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])

prefarea = st.selectbox("Preferred Area", ["yes", "no"])

furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

# ============================================
# Create DataFrame
# Column names must match training dataset
# ============================================

input_data = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "stories": [stories],
    "mainroad": [mainroad],
    "guestroom": [guestroom],
    "basement": [basement],
    "hotwaterheating": [hotwaterheating],
    "airconditioning": [airconditioning],
    "parking": [parking],
    "prefarea": [prefarea],
    "furnishingstatus": [furnishingstatus]
})

# ============================================
# Preprocess Input
# ============================================

processed_data = preprocessor.transform(input_data)

# ============================================
# Prediction
# ============================================

if st.button("Predict House Price"):

    prediction = model.predict(processed_data)

    st.success(f"🏡 Estimated House Price: {prediction[0]:,.2f}")
