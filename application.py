import streamlit as st
import numpy as np
import joblib  

# Load the trained model
model = joblib.load(r"C:\Users\m8494\Downloads\Farm_Irrigation_System.pkl")  

# Page configuration
st.set_page_config(page_title="SMART IRRIGATION SYSTEM", layout="centered")

# Stylish title
st.markdown("<h1 style='text-align: center; color: green;'>🌱 SMART IRRIGATION SYSTEM  🌱</h1>", unsafe_allow_html=True)
st.subheader("Enter scaled sensor values (0 to 1) to predict sprinkler status")

# Collect sensor inputs in a 4-column layout
st.markdown("### 🌡️ Adjust Sensor Values:")
sensor_values = []
cols = st.columns(4)  # 4 sliders per row

for i in range(20):
    with cols[i % 4]:  # distribute sliders into 4 columns
        val = st.slider(f"Sensor {i}", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        sensor_values.append(val)

# Predict button
if st.button("🔍 Predict Sprinklers"):
    input_array = np.array(sensor_values).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    st.success("✅ Prediction Complete!")
    st.markdown("### 💡 Sprinkler Status by Parcel:")

    # Count ON/OFF sprinklers
    on_count = np.sum(prediction)
    off_count = len(prediction) - on_count

    for i, status in enumerate(prediction):
        emoji = "🟢 ON" if status == 1 else "🔴 OFF"
        st.write(f"**Sprinkler {i} (parcel_{i})**: {emoji}")

    # Display summary
    st.markdown("---")
    st.info(f"**Summary:** {on_count} Sprinklers ON | {off_count} Sprinklers OFF")

# Footer / Credit Section
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "📌 This project is part of the <b>AICTE Internship</b>.<br>"
    "<b>Developed by: Madhav M Nambiar</b>"
    "</p>",
    unsafe_allow_html=True
)
