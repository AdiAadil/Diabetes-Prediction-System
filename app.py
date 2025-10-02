import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- Custom CSS for Dashboard Colors ---
st.markdown("""
    <style>
    .header {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    .subheader {
        background-color: #f2f2f2;
        padding: 5px;
        border-radius: 5px;
    }
    .input-box {
        background-color: #e6f7ff;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header"><h2>City Hospital - Diabetes Prediction System</h2></div>', unsafe_allow_html=True)
st.markdown('<div class="subheader"><b>Patient Health Checkup System</b></div>', unsafe_allow_html=True)
st.write("")

# Patient Info
patient_name = st.text_input("Patient Name", "")

# Input Form with Colored Boxes
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    pregnancies = st.number_input("Pregnancies", 0, 20, 0)
    bloodpressure = st.number_input("Blood Pressure", 0, 140, 70)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    glucose = st.number_input("Glucose", 0, 200, 120)
    skinthickness = st.number_input("Skin Thickness", 0, 99, 20)
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    insulin = st.number_input("Insulin", 0, 900, 79)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    st.markdown('</div>', unsafe_allow_html=True)

dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

# Prepare input
input_data = pd.DataFrame({
    'Pregnancies':[pregnancies],
    'Glucose':[glucose],
    'BloodPressure':[bloodpressure],
    'SkinThickness':[skinthickness],
    'Insulin':[insulin],
    'BMI':[bmi],
    'DiabetesPedigreeFunction':[dpf],
    'Age':[age]
})
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Display prediction
st.subheader("Prediction Result")
if prediction[0]==1:
    st.markdown("<span style='color:red; font-weight:bold; font-size:20px;'>High Risk of Diabetes</span>", unsafe_allow_html=True)
    recommendation = "Consult doctor, maintain diet & exercise."
else:
    st.markdown("<span style='color:green; font-weight:bold; font-size:20px;'>Low Risk of Diabetes</span>", unsafe_allow_html=True)
    recommendation = "Keep healthy lifestyle."

# Probability chart
st.subheader("Prediction Probability")
labels = ['Low Risk','High Risk']
probs = prediction_proba[0]
colors = ['#00b050','#ff0000']  # Green and Red
fig, ax = plt.subplots()
ax.bar(labels, probs, color=colors)
ax.set_ylim(0,1)
ax.set_ylabel("Probability")
for i, v in enumerate(probs):
    ax.text(i, v + 0.02, f"{v*100:.2f}%", ha='center', fontweight='bold')
st.pyplot(fig)

# PDF generation button
if st.button("Submit & Generate PDF Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(0, 0, 128)  # Dark Blue Header
    pdf.cell(0, 10, "Mewar Hospital - Diabetes Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)  # Black text
    pdf.cell(0, 10, f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 10, f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    for key, value in input_data.iloc[0].items():
        pdf.cell(0, 10, f"{key}: {value}", ln=True)
    pdf.ln(5)
    # Prediction color
    if prediction[0]==1:
        pdf.set_text_color(255,0,0)  # Red
    else:
        pdf.set_text_color(0,128,0)  # Green
    pdf.cell(0, 10, f"Prediction: {'High Risk' if prediction[0]==1 else 'Low Risk'}", ln=True)
    pdf.set_text_color(0,0,0)
    pdf.cell(0, 10, f"Probability - Low Risk: {prediction_proba[0][0]*100:.2f}%", ln=True)
    pdf.cell(0, 10, f"Probability - High Risk: {prediction_proba[0][1]*100:.2f}%", ln=True)
    pdf.cell(0, 10, f"Recommendation: {recommendation}", ln=True)
    pdf.output("Diabetes_Report.pdf")
    st.success("✅ PDF report generated successfully!")