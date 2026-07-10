import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("PredictiveMaintenance.pkl")
scaler = joblib.load("PredictiveMaintenanceScaler.pkl")
encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️")

st.title("⚙️ Predictive Maintenance System")

st.write("Predict whether the machine will fail.")

st.sidebar.header("Machine Parameters")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temp = st.sidebar.number_input(
    "Air Temperature (K)",
    value=298.0
)

process_temp = st.sidebar.number_input(
    "Process Temperature (K)",
    value=308.0
)

rpm = st.sidebar.number_input(
    "Rotational Speed (rpm)",
    value=1500
)

torque = st.sidebar.number_input(
    "Torque (Nm)",
    value=40.0
)

tool_wear = st.sidebar.number_input(
    "Tool Wear (min)",
    value=20
)

type_encoded = encoder.transform([machine_type])[0]

input_df = pd.DataFrame({
    "Type":[type_encoded],
    "Air temperature [K]":[air_temp],
    "Process temperature [K]":[process_temp],
    "Rotational speed [rpm]":[rpm],
    "Torque [Nm]":[torque],
    "Tool wear [min]":[tool_wear]
})

scaled = scaler.transform(input_df)

prediction = model.predict(scaled)[0]
probability = model.predict_proba(scaled)[0][1]

if st.button("Predict"):

    if prediction == 1:
        st.error("⚠ Machine Failure Predicted")

    else:
        st.success("✅ Machine Healthy")

    st.write(f"Failure Probability : **{probability:.2%}**")

st.markdown("---")

st.write("Developed using Machine Learning & Streamlit")