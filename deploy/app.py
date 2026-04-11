
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Predictive Maintenance App", layout="centered")
st.title("Predictive Maintenance Risk Predictor")
st.write("Enter engine sensor readings to predict whether maintenance is likely required.")

MODEL_PATH = Path("best_model.joblib")

if not MODEL_PATH.exists():
    st.error("Model file not found in the deployment folder.")
else:
    model = joblib.load(MODEL_PATH)

    engine_rpm = st.number_input("Engine rpm", value=750.0)
    lub_oil_pressure = st.number_input("Lub oil pressure", value=3.20)
    fuel_pressure = st.number_input("Fuel pressure", value=6.20)
    coolant_pressure = st.number_input("Coolant pressure", value=2.20)
    lub_oil_temp = st.number_input("lub oil temp", value=77.0)
    coolant_temp = st.number_input("Coolant temp", value=78.0)

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            "Engine rpm": engine_rpm,
            "Lub oil pressure": lub_oil_pressure,
            "Fuel pressure": fuel_pressure,
            "Coolant pressure": coolant_pressure,
            "lub oil temp": lub_oil_temp,
            "Coolant temp": coolant_temp
        }])

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        st.subheader("Prediction Result")
        st.write(f"Predicted Engine Condition: {int(pred)}")
        if prob is not None:
            st.write(f"Predicted Probability of Maintenance Required: {prob:.4f}")
