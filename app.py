import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("best_fire_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set page title and layout
st.set_page_config(page_title="Fire Type Classifier", layout="centered")

# App title
st.title("🔥 Fire Type Classification App")
st.markdown("Enter MODIS satellite readings to predict the fire type.")

# Input fields
brightness = st.number_input("Brightness", min_value=200.0, max_value=500.0, value=300.0)
bright_t31 = st.number_input("Brightness T31", min_value=200.0, max_value=400.0, value=290.0)
frp = st.number_input("Fire Radiative Power (FRP)", min_value=0.0, max_value=1000.0, value=15.0)
scan = st.number_input("Scan", min_value=0.0, max_value=5.0, value=1.0)
track = st.number_input("Track", min_value=0.0, max_value=5.0, value=1.0)
confidence = st.number_input("Confidence (0 to 100)", min_value=0, max_value=100, value=80)

# Prepare input for prediction
input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence]])
scaled_input = scaler.transform(input_data)

# Predict button
if st.button("Predict Fire Type"):
    prediction = model.predict(scaled_input)[0]

    fire_types = {
        0: "Vegetation Fire",
        2: "Other Static Land Source",
        3: "Offshore Fire"
    }

    result = fire_types.get(prediction, "Unknown")
    st.success(f"**Predicted Fire Type:** {result}")
