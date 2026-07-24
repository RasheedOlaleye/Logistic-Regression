"""
Step 4: Streamlit App for Interactive Predictions
Run with: uv run streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ---- CHECK FOR REQUIRED FILES ----
required_files = {
    'logistic_model.pkl': 'Trained model',
    'scaler.pkl': 'Feature scaler',
    'le_embarked.pkl': 'Embarked encoder'
}

missing_files = []
for filename, desc in required_files.items():
    if not os.path.exists(filename):
        missing_files.append(f"• `{filename}` ({desc})")

if missing_files:
    st.error("⚠️ Model files not found!")
    st.write("Missing files:")
    for f in missing_files:
        st.write(f)
    st.info("💡 Run this command first, then come back:")
    st.code("uv run train_model.py", language="bash")
    st.stop()

# ---- LOAD MODEL & PREPROCESSORS ----
@st.cache_resource
def load_artifacts():
    model = joblib.load('logistic_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_embarked = joblib.load('le_embarked.pkl')
    return model, scaler, le_embarked

model, scaler, le_embarked = load_artifacts()

# ---- HEADER ----
st.title("🚢 Titanic Survival Predictor")
st.markdown("""
This app uses a **Logistic Regression** model trained on the Titanic dataset 
to predict whether a passenger would have survived.
""")

st.divider()

# ---- SIDEBAR INPUTS ----
st.sidebar.header("📝 Passenger Details")

pclass = st.sidebar.selectbox(
    "Passenger Class",
    options=[1, 2, 3],
    index=2,
    help="1 = 1st class, 2 = 2nd class, 3 = 3rd class"
)

sex = st.sidebar.radio(
    "Sex",
    options=["Female", "Male"],
    index=1
)

age = st.sidebar.slider(
    "Age",
    min_value=0,
    max_value=100,
    value=25,
    step=1
)

fare = st.sidebar.slider(
    "Fare (£)",
    min_value=0,
    max_value=500,
    value=30,
    step=5
)

embarked = st.sidebar.selectbox(
    "Port of Embarkation",
    options=["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"],
    index=0
)

sibsp = st.sidebar.number_input(
    "Siblings/Spouses Aboard",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

parch = st.sidebar.number_input(
    "Parents/Children Aboard",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

# ---- PREDICTION ----
sex_encoded = 1 if sex == "Male" else 0

embarked_map = {"Southampton (S)": "S", "Cherbourg (C)": "C", "Queenstown (Q)": "Q"}
embarked_encoded = le_embarked.transform([embarked_map[embarked]])[0]

# Create feature vector
input_data = pd.DataFrame([{
    'pclass': pclass,
    'sex': sex_encoded,
    'age': age,
    'fare': fare,
    'embarked': embarked_encoded,
    'sibsp': sibsp,
    'parch': parch
}])

# Scale
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0]

# ---- DISPLAY RESULTS ----
st.divider()
st.subheader("🔮 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    if prediction == 1:
        st.success("### ✅ SURVIVED")
        st.balloons()
    else:
        st.error("### ❌ DID NOT SURVIVE")

with col2:
    st.metric(
        label="Survival Probability",
        value=f"{probability[1]:.1%}",
        delta=f"{'+' if probability[1] > 0.5 else ''}{(probability[1] - 0.5):.1%} vs 50%"
    )

# Probability bar
st.progress(float(probability[1]), text=f"Survival Confidence: {probability[1]:.1%}")

# ---- INTERPRETATION ----
st.divider()
st.subheader("📊 What influenced this prediction?")

factors = []
if sex == "Female":
    factors.append("🟢 Being female significantly increases survival odds")
else:
    factors.append("🔴 Being male significantly decreases survival odds")

if pclass == 1:
    factors.append("🟢 1st class ticket improves survival chances")
elif pclass == 3:
    factors.append("🔴 3rd class ticket reduces survival chances")

if age < 10:
    factors.append("🟢 Young children had higher survival priority")
elif age > 60:
    factors.append("🔴 Elderly passengers had lower survival rates")

if fare > 100:
    factors.append("🟢 Higher fare suggests better class/location")
elif fare < 10:
    factors.append("🔴 Very low fare suggests lower class/deck")

for f in factors:
    st.write(f)

# ---- RAW DATA TABLE ----
with st.expander("🔍 View Raw Input Data"):
    st.dataframe(input_data, use_container_width=True)

# ---- FOOTER ----
st.divider()
st.caption("""
Built with ❤️ using Python, Scikit-Learn & Streamlit.  
Model: Logistic Regression | Dataset: Titanic (Seaborn)
""")